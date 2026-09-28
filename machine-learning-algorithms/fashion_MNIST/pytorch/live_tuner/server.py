"""Live tuning console for the Fashion-MNIST MLP from fashionMNIST.ipynb.

Trains in a background thread and streams progress to the browser over
Server-Sent Events. The HTTP layer is stdlib only; torch and numpy are the deps.

    python server.py [--data DIR] [--device cuda] [--host 127.0.0.1] [--port 8000]
"""
import argparse
import json
import math
import queue
import threading
import time
from collections import deque
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "sigmoid": nn.Sigmoid}
OPTIMIZERS = ("adam", "adamw", "sgd")

VAL_SIZE = 10_000     # held out of the 60k train set; the test set stays untouched while tuning
EMIT_EVERY = 25       # steps between loss updates pushed to the browser
GRAPH_SECONDS = 0.5   # min interval between weight snapshots for the topology view
GRAPH_NODES = 16      # nodes drawn per hidden layer
INPUT_PROBES = 24     # input pixels drawn as reticles
MAX_RUNS = 8          # archived runs kept for ghost curves

DEFAULTS = {  # mirrors the notebook
    "lr": 1e-3, "batch_size": 64, "epochs": 10, "optimizer": "adam", "momentum": 0.9,
    "weight_decay": 0.0, "dropout": 0.0, "hidden": [512, 512], "activation": "relu", "seed": 0,
}
LIVE_KEYS = {"lr", "batch_size", "epochs", "optimizer", "momentum", "weight_decay", "dropout"}
STRUCTURAL_KEYS = {"hidden", "activation", "seed"}


def num(x):
    """Float that survives JSON: NaN/inf become null."""
    x = float(x)
    return x if math.isfinite(x) else None


def validate(key, value):
    def clamp(v, lo, hi):
        return min(max(v, lo), hi)
    if key == "lr":
        return clamp(float(value), 1e-6, 1.0)
    if key == "batch_size":
        return int(clamp(int(value), 8, 4096))
    if key == "epochs":
        return int(clamp(int(value), 1, 500))
    if key == "momentum":
        return clamp(float(value), 0.0, 0.999)
    if key == "weight_decay":
        return clamp(float(value), 0.0, 1.0)
    if key == "dropout":
        return clamp(float(value), 0.0, 0.9)
    if key == "seed":
        return int(value)
    if key == "optimizer":
        if value not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
        return value
    if key == "activation":
        if value not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {tuple(ACTIVATIONS)}")
        return value
    if key == "hidden":
        widths = [int(clamp(int(w), 1, 4096)) for w in value]
        if not 1 <= len(widths) <= 6:
            raise ValueError("between 1 and 6 hidden layers")
        return widths
    raise ValueError(f"unknown config key {key!r}")


# ---------------------------------------------------------------- data

def find_dataset_dir(explicit):
    candidates = [Path(explicit)] if explicit else []
    candidates += [
        HERE.parent.parent / "dataset",                              # fashion_MNIST/dataset
        Path.cwd() / "dataset",                                      # run from fashion_MNIST/
        Path.cwd() / "machine-learning-algorithms/fashion_MNIST/dataset",  # run from repo root
    ]
    for c in candidates:
        if (c / "train-images-idx3-ubyte").exists():
            return c
    raise SystemExit("dataset not found; pass --data path/to/dir-with-idx-files")


def read_idx(path):
    raw = path.read_bytes()
    ndim = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


class Data:
    """Whole dataset resident on the device; batching is just indexing."""

    def __init__(self, root, device):
        def load(images, labels):
            x = torch.from_numpy(read_idx(root / images).reshape(-1, 784).copy())
            y = torch.from_numpy(read_idx(root / labels).astype(np.int64))
            return x.to(device).float().div_(255), y.to(device)

        x, y = load("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
        split = torch.randperm(len(x), generator=torch.Generator().manual_seed(0)).to(device)  # fixed so runs compare
        self.x_val, self.y_val = x[split[:VAL_SIZE]], y[split[:VAL_SIZE]]
        self.x_train, self.y_train = x[split[VAL_SIZE:]], y[split[VAL_SIZE:]]
        self.x_test, self.y_test = load("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")


# ---------------------------------------------------------------- model

class MLP(nn.Module):
    def __init__(self, hidden, activation, dropout):
        super().__init__()
        dims = [784, *hidden, 10]
        self.linears = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.act = ACTIVATIONS[activation]()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, keep=False):
        acts = []
        for lin in self.linears[:-1]:
            x = self.act(lin(x))
            if keep:
                acts.append(x)
            x = self.drop(x)
        logits = self.linears[-1](x)
        return (logits, acts) if keep else logits


def make_optimizer(name, params, cfg):
    if name == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    cls = torch.optim.Adam if name == "adam" else torch.optim.AdamW
    return cls(params, lr=cfg["lr"], betas=(cfg["momentum"], 0.999), weight_decay=cfg["weight_decay"])


# ---------------------------------------------------------------- trainer

class Trainer:
    def __init__(self, data, device):
        self.data, self.device = data, device
        self.lock = threading.RLock()  # guards model, optimizer, config, history
        self.waiting = 0               # HTTP threads queued on the lock; the train loop yields to them
        self.subscribers = []
        self.sub_lock = threading.Lock()
        self.config = dict(DEFAULTS, hidden=list(DEFAULTS["hidden"]))
        self.pending = {}              # structural edits held until reset
        self.runs = []
        self.logs = deque(maxlen=200)  # replayed to tabs that connect later
        self.run_id = 1                # bumps on reset; shown in the UI
        self.version = 0               # bumps on every rebuild; lets the UI drop stale probes
        self.status = "idle"           # idle | running | paused | stopped | done | diverged
        self.thread = None
        self.resume_evt = threading.Event()
        self.ctl_lock = threading.RLock()  # serialises start/stop/reset from concurrent requests
        self.stop_req = False
        self.build()

    @contextmanager
    def access(self):
        self.waiting += 1
        with self.lock:
            self.waiting -= 1
            yield

    # -- lifecycle

    def build(self):
        cfg = self.config
        torch.manual_seed(cfg["seed"])
        self.model = MLP(cfg["hidden"], cfg["activation"], cfg["dropout"]).to(self.device)
        self.opt = make_optimizer(cfg["optimizer"], self.model.parameters(), cfg)
        self.version += 1
        self.epoch = self.step = self.samples = 0
        self.sps = 0.0
        self.best_val = math.inf
        self.history = {"steps": [], "epochs": [], "markers": []}
        self.last_eval = None
        self.test_result = None
        self.started_at = None
        rng = np.random.default_rng(0)
        self.input_nodes = sorted(rng.choice([r * 28 + c for r in range(4, 24) for c in range(4, 24)],
                                             INPUT_PROBES, replace=False).tolist())
        self.hidden_nodes = [np.unique(np.linspace(0, h - 1, min(h, GRAPH_NODES)).round().astype(int)).tolist()
                             for h in cfg["hidden"]]

    def archive(self):
        if not self.history["epochs"]:
            return
        eps = self.history["epochs"]
        self.runs.append({
            "id": self.run_id, "config": json.loads(json.dumps(self.config)),
            "curve": [[e["epoch"], e["val_loss"], e["val_acc"]] for e in eps],
            "best_val_acc": max((e["val_acc"] or 0) for e in eps),
            "best_val_loss": num(self.best_val),
            "test_acc": self.test_result and self.test_result["acc"],
        })
        self.runs = self.runs[-MAX_RUNS:]

    def control(self, action):
        with self.ctl_lock:
            return self._control(action)

    def _control(self, action):
        if action == "start":
            if self.status == "paused":
                return self.control("resume")
            if self.status == "running":
                return {"ok": True}
            if self.status == "diverged":
                raise ValueError("weights are NaN; reset first")
            if self.epoch >= self.config["epochs"]:
                raise ValueError("epoch budget reached; raise EPOCHS to keep training")
            self.stop_req = False
            self.resume_evt.set()
            self.started_at = self.started_at or time.time()
            self.set_status("running")
            self.thread = threading.Thread(target=self.train, daemon=True)
            self.thread.start()
        elif action == "pause" and self.status == "running":
            self.resume_evt.clear()
            self.set_status("paused")
        elif action == "resume" and self.status == "paused":
            self.resume_evt.set()
            self.set_status("running")
        elif action == "stop":
            self.halt()
            if self.status in ("running", "paused"):
                self.set_status("stopped")
        elif action == "reset":
            self.halt()
            with self.lock:
                self.archive()
                self.config.update(self.pending)
                self.pending = {}
                self.run_id += 1
                self.build()
                self.status = "idle"
            self.log("RESET · RUN %02d · %s" % (self.run_id, arch_label(self.config)))
            self.publish("snapshot", self.snapshot())
        else:
            raise ValueError(f"cannot {action!r} while {self.status}")
        return {"ok": True}

    def halt(self):
        self.stop_req = True
        self.resume_evt.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join()
        self.thread = None

    def set_status(self, status):
        self.status = status
        self.publish("status", {"status": status, "progress": self.progress()})

    # -- config

    def update_config(self, body):
        changes = []
        with self.lock:
            fresh = self.samples == 0 and self.status == "idle"
            for key, raw in body.items():
                value = validate(key, raw)
                if key in STRUCTURAL_KEYS:
                    if fresh:
                        self.config[key] = value
                        self.pending.pop(key, None)
                    elif value == self.config[key]:
                        self.pending.pop(key, None)
                    else:
                        self.pending[key] = value
                    continue
                old = self.config[key]
                if value == old:
                    continue
                self.config[key] = value
                changes.append((key, old, value))
                self.apply_live(key)
            if fresh and STRUCTURAL_KEYS & body.keys():
                self.build()
                self.publish("snapshot", self.snapshot())
            x = self.samples / len(self.data.x_train)
            for key, old, value in changes:
                if self.samples and key != "epochs":
                    text = f"{LABELS[key]} {fmt(key, old)} → {fmt(key, value)}"
                    self.history["markers"].append({"x": x, "text": text})
                    self.log(text)
            self.publish("config", {"config": self.config, "pending": self.pending,
                                    "markers": self.history["markers"]})
        return {"ok": True}

    def apply_live(self, key):
        cfg = self.config
        if key == "optimizer":  # hot swap keeps weights, drops moment estimates
            self.opt = make_optimizer(cfg["optimizer"], self.model.parameters(), cfg)
        elif key == "dropout":
            self.model.drop.p = cfg["dropout"]
        elif key in ("lr", "momentum", "weight_decay"):
            for g in self.opt.param_groups:
                g["lr"], g["weight_decay"] = cfg["lr"], cfg["weight_decay"]
                if "betas" in g:
                    g["betas"] = (cfg["momentum"], g["betas"][1])
                else:
                    g["momentum"] = cfg["momentum"]

    # -- training

    def train(self):
        d = self.data
        n = len(d.x_train)
        try:
            while self.epoch < self.config["epochs"] and not self.stop_req:
                perm = torch.randperm(n, device=self.device)
                i, epoch_loss, epoch_batches = 0, torch.zeros((), device=self.device), 0
                win_loss, win_correct, win_seen, win_steps = 0.0, 0.0, 0, 0
                t0 = last_graph = time.perf_counter()
                while i < n and not self.stop_req:
                    if not self.resume_evt.is_set():
                        self.resume_evt.wait()
                        t0, win_seen = time.perf_counter(), 0
                        continue
                    with self.lock:
                        idx = perm[i: i + self.config["batch_size"]]
                        i += len(idx)
                        self.model.train()
                        logits = self.model(d.x_train[idx])
                        loss = F.cross_entropy(logits, d.y_train[idx])
                        self.opt.zero_grad(set_to_none=True)
                        loss.backward()
                        self.opt.step()
                        self.step += 1
                        self.samples += len(idx)
                        detached = loss.detach()
                        epoch_loss += detached
                        epoch_batches += 1
                        win_loss = win_loss + detached
                        win_correct = win_correct + (logits.argmax(1) == d.y_train[idx]).sum()
                        win_seen += len(idx)
                        win_steps += 1
                    if win_steps == EMIT_EVERY:
                        loss_v = float(win_loss) / win_steps
                        self.sps = win_seen / max(time.perf_counter() - t0, 1e-9)
                        point = [round(self.samples / n, 5), num(loss_v)]
                        self.history["steps"].append(point)
                        self.publish("step", {"point": point, "acc": num(float(win_correct) / win_seen),
                                              "progress": self.progress()})
                        if not math.isfinite(loss_v):
                            self.log("DIVERGED · LOSS IS NOT FINITE · LOWER LR AND RESET", alert=True)
                            self.set_status("diverged")
                            return
                        win_loss, win_correct, win_seen, win_steps = 0.0, 0.0, 0, 0
                        t0 = time.perf_counter()
                    if time.perf_counter() - last_graph > GRAPH_SECONDS:
                        self.publish("graph", self.graph())
                        last_graph = time.perf_counter()
                    if self.waiting:
                        time.sleep(0.0005)  # let queued HTTP readers take the lock
                if i < n:
                    break  # stopped mid-epoch; the partial epoch is not recorded
                self.epoch += 1
                self.finish_epoch(float(epoch_loss) / max(epoch_batches, 1))
            if not self.stop_req:
                self.set_status("done")
                self.log("DONE · %d EPOCHS" % self.epoch)
        except Exception as e:  # surface crashes in the UI rather than a silent dead thread
            self.log(f"ERROR · {type(e).__name__}: {e}", alert=True)
            self.set_status("stopped")
            raise

    def finish_epoch(self, train_loss):
        with self.lock:
            ev = self.evaluate(self.data.x_val, self.data.y_val)
            record = {"epoch": self.epoch, "train_loss": num(train_loss), "val_loss": ev["loss"],
                      "val_acc": ev["acc"], "lr": self.config["lr"], "t": round(time.time() - self.started_at, 1)}
            self.history["epochs"].append(record)
            self.last_eval = ev
            best = ev["loss"] is not None and ev["loss"] < self.best_val
            if best:
                self.best_val = ev["loss"]
                torch.save({"state_dict": self.model.state_dict(), "config": self.config,
                            "epoch": self.epoch, "val_loss": ev["loss"]}, HERE / "best_model.pth")
        self.publish("epoch", {"record": record, "eval": ev, "best": best, "progress": self.progress()})
        self.publish("graph", self.graph())
        self.log("EPOCH %02d · TRAIN %.4f · VAL %.4f · ACC %.2f%%%s" % (
            self.epoch, train_loss, ev["loss"] or math.nan, 100 * ev["acc"], " · SAVED BEST" if best else ""))

    @torch.no_grad()
    def evaluate(self, x, y):
        self.model.eval()
        loss = torch.zeros((), device=self.device)
        conf = torch.zeros(100, dtype=torch.long, device=self.device)
        for i in range(0, len(x), 4096):
            logits, yy = self.model(x[i: i + 4096]), y[i: i + 4096]
            loss += F.cross_entropy(logits, yy, reduction="sum")
            conf += torch.bincount(yy * 10 + logits.argmax(1), minlength=100)
        conf = conf.view(10, 10).cpu()
        return {"loss": num(loss.item() / len(x)), "acc": conf.diag().sum().item() / len(x),
                "confusion": conf.tolist(),
                "per_class": (conf.diag() / conf.sum(1).clamp(min=1)).tolist()}

    def run_test(self):
        with self.access():
            ev = self.evaluate(self.data.x_test, self.data.y_test)
            self.test_result = {"epoch": self.epoch, **ev}
        self.log("TEST SET · EPOCH %02d · LOSS %.4f · ACC %.2f%%" % (self.epoch, ev["loss"] or math.nan, 100 * ev["acc"]))
        self.publish("test", self.test_result)
        return self.test_result

    # -- views

    def graph(self):
        with self.access():
            layers = [self.input_nodes, *self.hidden_nodes, list(range(10))]
            edges, scale = [], []
            for li, lin in enumerate(self.model.linears):
                w = torch.nan_to_num(lin.weight.detach())
                sub = w[layers[li + 1]][:, layers[li]]  # (dst, src)
                edges.append(sub.T.cpu().numpy().round(4).tolist())  # [src][dst]
                scale.append(num(w.std()))
        sizes = [784, *self.config["hidden"], 10]
        return {"version": self.version, "sizes": sizes, "nodes": layers, "edges": edges, "scale": scale}

    @torch.no_grad()
    def probe(self, idx):
        x, y = self.data.x_val, self.data.y_val
        idx = int(idx) % len(x) if idx is not None else int(torch.randint(len(x), ()))
        with self.access():
            self.model.eval()
            logits, acts = self.model(x[idx: idx + 1], keep=True)
            probs = torch.nan_to_num(logits.softmax(1)[0]).cpu().tolist()
            sampled = []
            for a, nodes in zip(acts, self.hidden_nodes):
                a = torch.nan_to_num(a[0])
                peak = a.abs().max().item() or 1.0
                sampled.append([round(a[n].item() / peak, 4) for n in nodes])
        pixels = (x[idx] * 255).round().byte().cpu().tolist()
        return {"idx": idx, "label": int(y[idx]), "pred": int(np.argmax(probs)), "probs": probs,
                "pixels": pixels, "acts": sampled, "version": self.version}

    def field(self, neuron):
        with self.access():
            w = self.model.linears[0].weight
            neuron = int(neuron)
            if not 0 <= neuron < w.shape[0]:
                raise ValueError("neuron out of range")
            row = torch.nan_to_num(w[neuron].detach()).cpu()
        return {"neuron": neuron, "weights": row.numpy().round(5).tolist(), "version": self.version}

    def progress(self):
        n = len(self.data.x_train)
        return {"epoch": self.epoch, "epochs": self.config["epochs"], "step": self.step,
                "x": round(self.samples / n, 5), "sps": round(self.sps), "best_val": num(self.best_val),
                "elapsed": round(time.time() - self.started_at, 1) if self.started_at else 0}

    def snapshot(self):
        with self.access():
            params = sum(p.numel() for p in self.model.parameters())
            return json.loads(json.dumps({
                "status": self.status, "config": self.config, "pending": self.pending,
                "history": self.history, "eval": self.last_eval, "test": self.test_result,
                "runs": self.runs, "logs": list(self.logs), "run": self.run_id, "version": self.version, "progress": self.progress(),
                "meta": {"device": device_label(self.device), "params": params, "classes": CLASSES,
                         "train": len(self.data.x_train), "val": len(self.data.x_val),
                         "test": len(self.data.x_test)},
                "graph": self.graph(),
            }))

    # -- events

    def publish(self, kind, payload):
        msg = f"event: {kind}\ndata: {json.dumps(payload)}\n\n".encode()
        with self.sub_lock:
            for q in list(self.subscribers):
                try:
                    q.put_nowait(msg)
                except queue.Full:  # a stalled tab; drop it rather than grow without bound
                    self.subscribers.remove(q)

    def log(self, text, alert=False):
        entry = {"t": time.strftime("%H:%M:%S"), "text": text, "alert": alert}
        self.logs.append(entry)
        self.publish("log", entry)


LABELS = {"lr": "LR", "batch_size": "BATCH", "epochs": "EPOCHS", "optimizer": "OPT",
          "momentum": "β1/MOM", "weight_decay": "WD", "dropout": "DROPOUT"}


def fmt(key, v):
    if key in ("lr", "weight_decay"):
        return f"{v:.1e}" if v else "0"
    if key == "optimizer":
        return v.upper()
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def arch_label(cfg):
    return "784-" + "-".join(map(str, cfg["hidden"])) + "-10 " + cfg["activation"].upper()


def device_label(device):
    if device.type == "cuda":
        return f"CUDA · {torch.cuda.get_device_name(device)}"
    return "CPU"


# ---------------------------------------------------------------- http

class Handler(BaseHTTPRequestHandler):
    trainer: Trainer = None

    def log_message(self, *args):
        pass

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in ("/", "/index.html"):
                return self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/events":
                return self.events()
            if url.path == "/api/state":
                return self.json(self.trainer.snapshot())
            if url.path == "/api/probe":
                return self.json(self.trainer.probe(q.get("idx")))
            if url.path == "/api/field":
                return self.json(self.trainer.field(q.get("neuron", 0)))
            self.send(404, b"not found", "text/plain")
        except ValueError as e:
            self.json({"error": str(e)}, 400)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            path = urlparse(self.path).path
            if path == "/api/config":
                return self.json(self.trainer.update_config(body))
            if path == "/api/control":
                return self.json(self.trainer.control(body.get("action")))
            if path == "/api/test":
                return self.json(self.trainer.run_test())
            self.send(404, b"not found", "text/plain")
        except (ValueError, TypeError) as e:
            self.json({"error": str(e)}, 400)

    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj, code=200):
        self.send(code, json.dumps(obj).encode(), "application/json")

    def events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        q = queue.Queue(maxsize=2000)
        t = self.trainer
        with t.sub_lock:
            t.subscribers.append(q)
        try:
            self.wfile.write(f"event: snapshot\ndata: {json.dumps(t.snapshot())}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    msg = q.get(timeout=15)
                except queue.Empty:
                    msg = b": keepalive\n\n"
                self.wfile.write(msg)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with t.sub_lock:
                if q in t.subscribers:
                    t.subscribers.remove(q)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", help="directory holding the raw idx files")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to expose on the network")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    root = find_dataset_dir(args.data)
    print(f"dataset {root}\ndevice  {device_label(device)}")
    Handler.trainer = Trainer(Data(root, device), device)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"open    http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
