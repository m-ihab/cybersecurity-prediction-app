import os
import pandas as pd
import joblib
import torch
from flask import Flask, request, render_template, send_file
from preprocessing import preprocess_samples

app = Flask(__name__)

MODEL_PATHS = {
    "Random Forest": "./data/best_rf_model.pkl",
    "XGBoost": "./data/best_xgb_model.pkl",
    "Neural Network": "./data/best_nn_model.pth"
}

expected_columns = [
    "Timestamp", "Source IP Address", "Destination IP Address", "Source Port",
    "Destination Port", "Protocol", "Packet Length", "Packet Type",
    "Traffic Type", "Payload Data", "Malware Indicators", "Anomaly Scores",
    "Alerts/Warnings", "Attack Type", "Attack Signature", "Action Taken",
    "Severity Level", "User Information", "Device Information",
    "Network Segment", "Geo-location Data", "Proxy Information",
    "Firewall Logs", "IDS/IPS Alerts", "Log Source"
]

def load_csv(file):
    """Loads a CSV and assigns headers if missing."""
    df = pd.read_csv(file, header=None)

    if df.shape[1] == len(expected_columns):  
        df.columns = expected_columns
    else:
        print("⚠️ Warning: Mismatch in column count! Assigning expected headers.")
        df = pd.read_csv(file, names=expected_columns, header=None)

    return df

def load_model(model_name):
    if model_name == "Neural Network":
        model = torch.load(MODEL_PATHS[model_name], map_location=torch.device('cpu'))
        model.eval()
    else:
        model = joblib.load(MODEL_PATHS[model_name])
    return model

@app.route("/")
def index():
        return render_template("index.html", models=list(MODEL_PATHS.keys()))


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if "file" not in request.files:
        return "No file part"

    file = request.files["file"]
    if file.filename == "":
        return "No selected file"

    raw_data = load_csv(file)
    processed_data = preprocess_samples(raw_data)

    true_attack_type = None
    if "Attack Type" in processed_data.columns:
        true_attack_type = processed_data["Attack Type"].copy()
        processed_data = processed_data.drop(columns=["Attack Type"]) 

    model_name = request.form["model"]
    model = load_model(model_name)

    if model_name == "Neural Network":
        inputs = torch.tensor(processed_data.values, dtype=torch.float32)
        predictions = model(inputs).argmax(dim=1).numpy()
    else:
        predictions = model.predict(processed_data)

    attack_mapping = {0: "Malware", 1: "Intrusion", 2: "DDoS"}
    predictions = [attack_mapping[pred] for pred in predictions]

    if true_attack_type is not None:
        true_attack_type = [attack_mapping[label] for label in true_attack_type]

    return render_template("index.html", 
                           predictions=zip(predictions, true_attack_type) if true_attack_type is not None else predictions, 
                           models=list(MODEL_PATHS.keys()))
if __name__ == "__main__":
    app.run(debug=True)
