import streamlit as st
import joblib
import pandas as pd

# Load trained model and preprocessors
model = joblib.load("failure_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("🔧 Machine Failure Prediction App")

# Collect user input
type_value = st.selectbox("Machine Type", ["M", "L", "H"])
air_temp = st.number_input("Air Temperature (K)", min_value=290.0, max_value=310.0, value=298.1)
process_temp = st.number_input("Process Temperature (K)", min_value=300.0, max_value=320.0, value=308.6)
rot_speed = st.number_input("Rotational Speed (rpm)", min_value=1000, max_value=3000, value=1551)
torque = st.number_input("Torque (Nm)", min_value=10.0, max_value=100.0, value=42.8)
tool_wear = st.number_input("Tool Wear (minutes)", min_value=0, max_value=500, value=0)

if st.button("Predict"):
    # Prepare input data
    input_data = {
        "Type": type_value,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Debugging: Check columns in df
    st.write("Columns in input data:", df.columns)

    # Encode categorical variables
    for col in label_encoders:
        if col in df.columns:
            df[col] = label_encoders[col].transform(df[col])

    # Debugging: Check columns after encoding
    st.write("Columns after encoding:", df.columns)

    # Ensure the columns match the scaler's expected features
    expected_columns = scaler.feature_names_in_

    # Remove or add missing columns to match the scaler's expected features
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        for col in missing_columns:
            df[col] = 0  # or use an appropriate placeholder

    # Scale numerical features
    df_scaled = scaler.transform(df)

    # Debugging: Check if scaling is applied correctly
    st.write("Columns expected by scaler:", expected_columns)

    # Predict
    prediction = model.predict(df_scaled)

    # Display results
    if prediction[0] == 1:
        st.error("⚠️ Failure is predicted!")
    else:
        st.success("✅ No failure detected!")
