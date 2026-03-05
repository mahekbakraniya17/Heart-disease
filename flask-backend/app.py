from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load trained model data
weights = np.load("weights.npy")   # shape: (12,1)
X_mean = np.load("X_mean.npy")     # shape: (11,)
X_std = np.load("X_std.npy")       # shape: (11,)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Home route (important for Render test)
@app.route("/")
def home():
    return "Heart Disease ML Backend Running"

# Order of features as in training
FEATURE_ORDER = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validate JSON keys
        for key in FEATURE_ORDER:
            if key not in data:
                return jsonify({"error": f"Missing key '{key}'"}), 400

        # Prepare feature array
        features = np.array(
            [data[key] for key in FEATURE_ORDER], dtype=float
        ).reshape(1, -1)

        # Add intercept column
        features = np.hstack([np.ones((features.shape[0], 1)), features])

        # Standardize features except intercept
        features[:, 1:] = (features[:, 1:] - X_mean) / X_std

        # Compute probability
        probability = sigmoid(features @ weights)[0][0]
        prediction = int(probability >= 0.5)

        return jsonify({
            "prediction": prediction,
            "probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)