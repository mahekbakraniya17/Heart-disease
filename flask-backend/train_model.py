import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("../data/pre_proccesed_data.csv")

X = df.drop(columns=["cardio"]).values   # (m, 11)
y = df["cardio"].values.reshape(-1, 1)

# ✅ Mean & std BEFORE bias
X_mean = X.mean(axis=0)   # (11,)
X_std = X.std(axis=0)     # (11,)

# Standardize
X = (X - X_mean) / X_std

# ✅ Add bias ONCE
X = np.c_[np.ones((X.shape[0], 1)), X]   # (m, 12)

# Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient Descent
def train(X, y, lr=0.01, epochs=1000):
    w = np.zeros((X.shape[1], 1))
    for _ in range(epochs):
        h = sigmoid(X @ w)
        w -= (lr / len(y)) * (X.T @ (h - y))
    return w

weights = train(X, y)

# ✅ Save model artifacts
np.save("weights.npy", weights)
np.save("X_mean.npy", X_mean)
np.save("X_std.npy", X_std)

print("✅ Model retrained correctly")
print("weights:", weights.shape)
print("X_mean:", X_mean.shape)
