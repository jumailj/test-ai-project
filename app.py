# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 10:46:47 2025

@author: ARJUN
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Load trained model and preprocessors
model = joblib.load("failure_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Machine Failure Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        df = pd.DataFrame([data])

        # Encode categorical variables
        for col in label_encoders:
            if col in df.columns:
                df[col] = label_encoders[col].transform(df[col])

        # Scale numerical features
        df_scaled = scaler.transform(df)

        # Make prediction
        prediction = model.predict(df_scaled)

        # Return result
        return jsonify({"Failure Prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    print("🚀 Starting Flask Server...")  # Explicit message
    app.run(debug=True)
