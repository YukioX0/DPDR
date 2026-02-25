import pickle
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='.', static_url_path='')

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Extract symptoms array from the training dataset columns
try:
    df = pd.read_csv("Training.csv", nrows=0)
    symptoms = list(df.columns)[:-1]
except Exception as e:
    print(f"Error loading Training.csv: {e}")
    symptoms = []

@app.route("/")
def home():
    return app.send_static_file("index.html")

@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    return jsonify({"symptoms": symptoms})

@app.route("/predict", methods=["POST"])
def predict():
    selected_symptoms = request.json.get("symptoms", [])
    
    # Create input vector of 0s
    input_vector = np.zeros(len(symptoms))
    
    # Set to 1 the symptoms selected by the user
    for symptom in selected_symptoms:
        if symptom in symptoms:
            idx = symptoms.index(symptom)
            input_vector[idx] = 1
            
    # Predict
    prediction = model.predict([input_vector])
    return jsonify({"prediction": str(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True, port=5000)