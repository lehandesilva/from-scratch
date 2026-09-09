from math import tanh
import numpy as np
# Forward pass

# define x, w, z, a

x, y = 2.0, 1.0
w1, w2 = 0.5, -1.0
lr = 0.1

for epoch in range(50):

    z1 = w1*x
    a1 = tanh(z1)

    y_hat = w2 * a1
    L = (y_hat - y)**2

    # dL/dw. Using chain rule traverse back. 
    dL_dyhat = 2 * (y_hat - y)          # ∂L/∂ŷ
    dL_dw2   = dL_dyhat * a1           # ∂L/∂w2 = ∂L/∂ŷ · ∂ŷ/∂w2
    dL_da1   = dL_dyhat * w2          # ∂L/∂a1
    dL_dz1   = dL_da1 * (1 - a1**2)  # ∂L/∂z1  (tanh')
    dL_dw1   = dL_dz1 * x            # ∂L/∂w1 = ∂L/∂z1 · ∂z1/∂w1

    w1 = w1 - lr * dL_dw1
    w2 = w2 - lr * dL_dw2

    print(f"{epoch:3d}  L={L:.6f}  w1={w1:.4f}  w2={w2:.4f}  ŷ={y_hat:.4f}")