# Streamlit Optimizer Comparison Dashboard

"""
Instructions to run in VS Code
--------------------------------
1. Create project folder and open it in VS Code.
2. Create virtual environment:
   Windows:
       python -m venv .venv
       .venv\\Scripts\\activate
   Mac/Linux:
       python3 -m venv .venv
       source .venv/bin/activate

3. Install required packages:
       pip install streamlit numpy matplotlib

4. Save this file as: optimizer_dashboard.py
5. Run Streamlit app:
       streamlit run optimizer_dashboard.py

This will open the dashboard automatically in your browser.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Simple MLP Model ---
class SimpleMLP:
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def backward(self, X, y, y_pred):
        m = len(y)
        dz2 = (y_pred - y) / m

        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - np.tanh(self.z1) ** 2)

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

# --- Optimizers ---
def sgd_update(params, grads, lr):
    for p, g in zip(params, grads):
        p -= lr * g


def rmsprop_update(params, grads, cache, lr, beta=0.9, eps=1e-8):
    for i, (p, g) in enumerate(zip(params, grads)):
        cache[i] = beta * cache[i] + (1-beta) * g**2
        p -= lr * g / (np.sqrt(cache[i]) + eps)


def adam_update(params, grads, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    for i, (p, g) in enumerate(zip(params, grads)):
        m[i] = beta1 * m[i] + (1-beta1) * g
        v[i] = beta2 * v[i] + (1-beta2) * (g**2)

        m_hat = m[i] / (1 - beta1**t)
        v_hat = v[i] / (1 - beta2**t)

        p -= lr * m_hat / (np.sqrt(v_hat) + eps)


# Train function
def train(optimizer_name, lr, iters):
    # Generate data
    X = np.linspace(-1, 1, 200).reshape(-1, 1)
    y = np.sin(2*np.pi*X) + 0.1 * np.random.randn(*X.shape)

    model = SimpleMLP()
    params = [model.W1, model.b1, model.W2, model.b2]
    losses = []

    if optimizer_name == "RMSProp":
        cache = [np.zeros_like(p) for p in params]
    if optimizer_name == "Adam":
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]

    for t in range(1, iters+1):
        y_pred = model.forward(X)
        loss = np.mean((y_pred - y)**2)
        losses.append(loss)

        grads = model.backward(X, y, y_pred)

        if optimizer_name == "SGD":
            sgd_update(params, grads, lr)
        elif optimizer_name == "RMSProp":
            rmsprop_update(params, grads, cache, lr)
        else:  # Adam
            adam_update(params, grads, m, v, t, lr)

    return X, y, model, losses

# --- Streamlit UI ---
st.title("Optimizer Convergence Comparison Dashboard")
st.write("Compare how SGD, RMSProp, and Adam converge on a regression task.")

optimizer = st.selectbox("Select Optimizer", ["SGD", "RMSProp", "Adam"])
lr = st.slider("Learning Rate", 0.0001, 0.05, 0.01, step=0.0001)
iters = st.slider("Iterations", 100, 3000, 1000, step=100)

if st.button("Run Training"):
    X, y, model, losses = train(optimizer, lr, iters)

    # --- Plot Loss ---
    fig1, ax1 = plt.subplots()
    ax1.plot(losses)
    ax1.set_title(f"Loss Curve ({optimizer})")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("MSE Loss")
    st.pyplot(fig1)

    # --- Plot Fit ---
    fig2, ax2 = plt.subplots()
    X_test = np.linspace(-1, 1, 200).reshape(-1, 1)
    y_pred = model.forward(X_test)

    ax2.scatter(X, y, s=10, label="Data")
    ax2.plot(X_test, y_pred, label="Prediction", linewidth=2)
    ax2.set_title(f"Model Fit ({optimizer})")
    ax2.legend()
    st.pyplot(fig2)
