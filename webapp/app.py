from flask import Flask, request, jsonify, render_template
import pandas as pd
import torch
import joblib
import os
import numpy as np

# Import preprocessing function from preprocessing.py
from preprocessing import preprocess_data  # This should be defined inside preprocessing.py

# Initialize Flask app
app = Flask(__name__)

# Define available models
MODEL_PATHS = {
    "Random Forest": "../data/best_rf_model.pkl",
    "XGBoost": "../data/best_xgb_model.pkl",
    "Neural Network": "../data/best_nn_model.pth"
}

# Load Scaler (used in preprocessing)
scaler = joblib.load("./data/scaler.pkl")

# Function to load selected model
def load_model(model_name):
    model_path = MODEL_PATHS.get(model_name)
    if model_path.endswith(".pkl"):
        return joblib.load(model_path)  # Load Random Forest & XGBoost
    elif model_path.endswith(".pth"):
        import torch.nn as nn  # Import the neural network architecture
        input_size = 20  # Adjust based on actual input size
        num_classes = 3  # Adjust based on actual classes
        model = nn(input_size, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        return model
    return None

# Home route (renders UI)
@app.route('/')
def home():
    return render_template('index.html', models=list(MODEL_PATHS.keys()))

# Prediction route (Manual Input)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        input_data = {key: request.form[key] for key in request.form}
        input_df = pd.DataFrame([input_data])

        # Apply preprocessing
        processed_data = preprocess_data(input_df)

        # Load selected model
        model_name = request.form.get("model")
        model = load_model(model_name)

        # Make prediction
        if model_name in ["Random Forest", "XGBoost"]:
            prediction = model.predict(processed_data)[0]
        elif model_name == "Neural Network":
            input_tensor = torch.tensor(processed_data.values, dtype=torch.float32)
            prediction = model(input_tensor).argmax(dim=1).item()

        return jsonify({"model": model_name, "prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})

# Batch Prediction route (CSV Upload)
@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        file = request.files['file']
        data = pd.read_csv(file)

        # Apply preprocessing
        processed_data = preprocess_data(data)

        # Load selected model
        model_name = request.form.get("model")
        model = load_model(model_name)

        # Make predictions
        if model_name in ["Random Forest", "XGBoost"]:
            predictions = model.predict(processed_data)
        elif model_name == "Neural Network":
            input_tensor = torch.tensor(processed_data.values, dtype=torch.float32)
            predictions = model(input_tensor).argmax(dim=1).numpy()

        # Return predictions
        results = pd.DataFrame({'Prediction': predictions})
        return results.to_json(orient='records')

    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
