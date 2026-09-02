#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>

// MLP for Fashion-MNIST, pure C++, no libraries.
// Scaffold: data loading, evaluation and the training loop shape are done.
// The math (activations, loss, forward, backward) is left as TODO for you.
// Build and run from this directory:  g++ -O2 -std=c++17 main.cpp -o mlp && ./mlp

// --- Constants ---

constexpr int IMAGE_SIZE = 784;   // 28 * 28 pixels, flattened into one vector
constexpr int NUM_CLASSES = 10;   // ten clothing categories

const char* CLASS_NAMES[NUM_CLASSES] = {
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
};

// --- Data ---

struct Sample {
    std::vector<double> pixels;  // 784 values scaled to [0, 1]
    int label;                   // 0-9, index into CLASS_NAMES
};

// Parse one CSV row: the label, then 784 pixel values, all small non-negative ints.
// Hand-rolled rather than std::stod because that would run 47 million times.
static Sample parseLine(const std::string& line) {
    Sample s;
    s.label = -1;
    s.pixels.reserve(IMAGE_SIZE);
    int value = 0;
    bool haveLabel = false;
    for (size_t i = 0; i <= line.size(); i++) {
        char c = (i < line.size()) ? line[i] : ',';  // treat end-of-line as a final comma
        if (c >= '0' && c <= '9') {
            value = value * 10 + (c - '0');          // accumulate digits of the current field
        } else if (c == ',') {
            if (!haveLabel) {
                s.label = value;                     // first field on the row is the label
                haveLabel = true;
            } else {
                s.pixels.push_back(value / 255.0);   // scale pixel from 0-255 into [0, 1]
            }
            value = 0;                               // reset for the next field
        }
        // anything else (a trailing \r on Windows line endings) is ignored
    }
    return s;
}

// Read a whole CSV file, skipping the header row.
std::vector<Sample> loadCSV(const std::string& path, size_t expectedRows) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "could not open " << path << std::endl;
        std::exit(1);
    }
    std::string line;
    std::getline(file, line);       // discard header: label,pixel1,...,pixel784
    std::vector<Sample> data;
    data.reserve(expectedRows);
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        Sample s = parseLine(line);
        if (s.pixels.size() != static_cast<size_t>(IMAGE_SIZE)) {   // catch a malformed row early
            std::cerr << "row " << data.size() << " has " << s.pixels.size() << " pixels" << std::endl;
            std::exit(1);
        }
        data.push_back(std::move(s));
    }
    std::cout << "loaded " << data.size() << " samples from " << path << std::endl;
    return data;
}

// Print a sample as ASCII art, so you can confirm by eye that the loader is right.
void printAscii(const Sample& s) {
    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
            double p = s.pixels[r * 28 + c];
            std::cout << (p > 0.5 ? '#' : (p > 0.2 ? '+' : '.'));
        }
        std::cout << std::endl;
    }
    std::cout << "label " << s.label << " (" << CLASS_NAMES[s.label] << ")" << std::endl;
}

// Count how many samples of each class a split holds.
// A dev set missing a class, or badly skewed, makes its accuracy number meaningless.
void printClassCounts(const std::string& name, const std::vector<Sample>& data) {
    std::vector<int> counts(NUM_CLASSES, 0);
    for (const Sample& s : data) counts[s.label]++;
    std::cout << name << " (" << data.size() << "): ";
    for (int i = 0; i < NUM_CLASSES; i++) std::cout << i << "=" << counts[i] << " ";
    std::cout << std::endl;
}

// Move the last `devSize` samples out of `data` into a new split.
// The shuffle first is what makes this a random split rather than a slice that
// depends on however the CSV happened to be ordered.
std::vector<Sample> splitOffDev(std::vector<Sample>& data, size_t devSize, std::mt19937& rng) {
    std::shuffle(data.begin(), data.end(), rng);
    std::vector<Sample> dev(std::make_move_iterator(data.end() - devSize),
                            std::make_move_iterator(data.end()));
    data.resize(data.size() - devSize);   // drop the moved-from tail
    return dev;
}

// --- Activations and loss (TODO) ---

// TODO: relu(z) = max(0, z). Hidden layers use this instead of sigmoid.
double relu(double z) {
    (void)z;
    return 0.0;
}

// TODO: derivative expressed via the output a: 1 if a > 0, else 0.
double relu_derivative(double a) {
    (void)a;
    return 0.0;
}

// TODO: softmax turns raw scores into probabilities that sum to 1.
// Subtract the largest score before exp() or large scores overflow to inf.
void softmax(std::vector<double>& z) {
    (void)z;
}

// TODO: categorical cross-entropy is just -log(probability of the correct class).
double crossEntropy(const std::vector<double>& probs, int label) {
    (void)probs;
    (void)label;
    return 0.0;
}

// --- Layer ---

enum class Activation { ReLU, Softmax };

struct Layer {
    Activation activation;
    std::vector<std::vector<double>> W;   // W[j][i]: weight from input i into neuron j
    std::vector<double> b;                // b[j]: bias of neuron j
    std::vector<std::vector<double>> dW;  // weight gradients summed over the current batch
    std::vector<double> db;               // bias gradients summed over the current batch
    std::vector<double> input;            // cached input from the last forward pass
    std::vector<double> output;           // cached activations from the last forward pass
};

// He initialisation: normal(0, sqrt(2 / fan_in)).
// Uniform(-1, 1) would give pre-activations of magnitude ~15 with 784 inputs, and stall.
Layer makeLayer(int numInputs, int numNeurons, Activation act, std::mt19937& rng) {
    std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / numInputs));
    Layer layer;
    layer.activation = act;
    layer.W.resize(numNeurons, std::vector<double>(numInputs));
    layer.b.assign(numNeurons, 0.0);                                  // biases start at zero
    layer.dW.resize(numNeurons, std::vector<double>(numInputs, 0.0));
    layer.db.assign(numNeurons, 0.0);
    for (int j = 0; j < numNeurons; j++) {
        for (int i = 0; i < numInputs; i++) {
            layer.W[j][i] = dist(rng);                                // random weight breaks symmetry
        }
    }
    return layer;
}

// --- Forward pass (TODO) ---

// TODO: weighted sum + bias per neuron, then apply the layer's activation.
// Cache input and output on the layer; backprop needs both.
// Softmax layers apply softmax to the whole z vector at once, not per neuron.
std::vector<double> forwardLayer(Layer& layer, const std::vector<double>& input) {
    (void)input;
    layer.output.assign(layer.W.size(), 1.0 / layer.W.size());  // stub: uniform guess
    return layer.output;
}

// TODO: feed each layer's output into the next.
std::vector<double> forward(std::vector<Layer>& network, const std::vector<double>& x) {
    std::vector<double> a = x;
    for (Layer& layer : network) {
        a = forwardLayer(layer, a);
    }
    return a;
}

// --- Backward pass (TODO) ---

// TODO: compute gradients and ADD them into layer.dW / layer.db.
// Do not update weights here; applyGradients does that once per batch.
//
// Softmax + cross-entropy collapse the output delta to the same simple form
// your XOR net used: delta[j] = output[j] - (j == label ? 1 : 0).
// Route delta back through W, then scale by relu_derivative of the previous
// layer's output. Read the old weight before you change anything.
void backward(std::vector<Layer>& network, int label) {
    (void)network;
    (void)label;
}

// --- Gradient bookkeeping ---

// Clear the accumulators at the start of each mini-batch.
void zeroGradients(std::vector<Layer>& network) {
    for (Layer& layer : network) {
        for (std::vector<double>& row : layer.dW) {
            std::fill(row.begin(), row.end(), 0.0);
        }
        std::fill(layer.db.begin(), layer.db.end(), 0.0);
    }
}

// One gradient-descent step per batch, using the average gradient over the batch.
void applyGradients(std::vector<Layer>& network, double lr, size_t batchCount) {
    double scale = lr / static_cast<double>(batchCount);
    for (Layer& layer : network) {
        for (size_t j = 0; j < layer.W.size(); j++) {
            for (size_t i = 0; i < layer.W[j].size(); i++) {
                layer.W[j][i] -= scale * layer.dW[j][i];
            }
            layer.b[j] -= scale * layer.db[j];
        }
    }
}

// --- Evaluation ---

// The network's prediction is the class with the highest output probability.
int predict(std::vector<Layer>& network, const Sample& s) {
    std::vector<double> out = forward(network, s.pixels);
    return static_cast<int>(std::max_element(out.begin(), out.end()) - out.begin());
}

// Fraction of samples classified correctly.
double accuracy(std::vector<Layer>& network, std::vector<Sample>& data) {
    int correct = 0;
    for (Sample& s : data) {
        if (predict(network, s) == s.label) correct++;
    }
    return static_cast<double>(correct) / data.size();
}

// Confusion matrix: rows are true classes, columns are predictions.
// The off-diagonal entries tell you WHICH classes the net mixes up, which a
// single accuracy number hides. Expect the shirt/pullover/coat block to be worst.
void confusionMatrix(std::vector<Layer>& network, std::vector<Sample>& data) {
    std::vector<std::vector<int>> counts(NUM_CLASSES, std::vector<int>(NUM_CLASSES, 0));
    for (Sample& s : data) {
        counts[s.label][predict(network, s)]++;
    }
    std::cout << "\ntrue \\ pred";
    for (int j = 0; j < NUM_CLASSES; j++) std::cout << "  " << j;
    std::cout << "   per-class accuracy" << std::endl;
    for (int i = 0; i < NUM_CLASSES; i++) {
        int total = 0;
        for (int j = 0; j < NUM_CLASSES; j++) total += counts[i][j];
        std::cout << i << " " << CLASS_NAMES[i];
        for (int j = 0; j < NUM_CLASSES; j++) std::cout << "  " << counts[i][j];
        std::cout << "   " << (total ? static_cast<double>(counts[i][i]) / total : 0.0) << std::endl;
    }
}

int main() {
    std::cout << "YEAHHHH... OKAYYYY!!!! - lil Jon";
    const std::string dir = "../dataset/";
    std::vector<Sample> train = loadCSV(dir + "fashion-mnist_train.csv", 60000);
    std::vector<Sample> test  = loadCSV(dir + "fashion-mnist_test.csv", 10000);

    // Milestone 1: eyeball one image. Row 0 is a Pullover, so expect a sweater shape.
    printAscii(train[0]);

    // Three splits:
    //   train - the only data the weights are ever fitted to
    //   dev   - every tuning decision (lr, layer size, when to stop) is judged here
    //   test  - the separate CSV, opened once at the very end and never tuned against
    // The dev set gets its own rng so changing the training seed or any
    // hyperparameter leaves the split identical, keeping runs comparable.
    std::mt19937 splitRng(1234);
    std::vector<Sample> dev = splitOffDev(train, 5000, splitRng);
    printClassCounts("train", train);
    printClassCounts("dev  ", dev);
    printClassCounts("test ", test);

    std::mt19937 rng(42);                                                  // fixed seed = reproducible runs
    std::vector<Layer> network;
    network.push_back(makeLayer(IMAGE_SIZE, 128, Activation::ReLU, rng));  // hidden layer
    network.push_back(makeLayer(128, NUM_CLASSES, Activation::Softmax, rng));

    const double lr = 0.1;         // step size
    const size_t batchSize = 64;   // samples averaged into one weight update
    const int epochs = 20;         // full passes over the training set

    std::vector<size_t> order(train.size());
    std::iota(order.begin(), order.end(), 0);   // indices we shuffle instead of the data itself

    for (int e = 0; e < epochs; e++) {
        std::shuffle(order.begin(), order.end(), rng);   // new sample order every epoch
        double epochLoss = 0.0;

        for (size_t start = 0; start < order.size(); start += batchSize) {
            zeroGradients(network);                                    // fresh accumulators per batch
            size_t end = std::min(start + batchSize, order.size());    // last batch may be short
            for (size_t n = start; n < end; n++) {
                const Sample& s = train[order[n]];
                std::vector<double> out = forward(network, s.pixels);  // predict
                epochLoss += crossEntropy(out, s.label);               // measure error
                backward(network, s.label);                            // accumulate gradients
            }
            applyGradients(network, lr, end - start);                  // one update per batch
        }

        // Watch both numbers: train loss falling while dev accuracy stalls means overfitting.
        std::cout << "epoch " << e
                  << "  train loss " << epochLoss / train.size()
                  << "  dev acc " << accuracy(network, dev) << std::endl;
    }

    std::cout << "\nfinal test accuracy " << accuracy(network, test) << std::endl;
    confusionMatrix(network, test);
    return 0;
}
