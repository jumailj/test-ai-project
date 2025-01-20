# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 10:33:52 2025

@author: ARJUN
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("machine_maintenance.csv")  # Update with actual filename

# Fix column name issues (strip spaces)
df.columns = df.columns.str.strip()

# Print column names to verify
print("Available columns:", df.columns)

# Drop unnecessary columns
df.drop(columns=["UDI", "Product ID"], errors="ignore", inplace=True)

# Define target column
target_column = "Target"

if target_column in df.columns:
    X = df.drop(columns=[target_column])  # Features
    y = df[target_column]  # Target variable (0/1)
else:
    print(f"Column '{target_column}' not found. Please check dataset.")
    quit()  # Stop execution in Spyder

# Encode categorical variables
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col])

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model and preprocessing tools
joblib.dump(model, "failure_prediction_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("Model training and saving completed successfully!")
