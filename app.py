import pickle
import pandas as pd
import streamlit as st

# Load Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load Scaler
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None


def preprocess_and_predict(features):
    input_data = pd.DataFrame([features])

    required_columns = [
        "Age",
        "Gender",
        "AnnualIncome",
        "SpendingScore",
        "MaritalStatus"
    ]

    input_data = input_data[required_columns]

    if scaler is not None:
        input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]

    return prediction[0], probability[0]


# Streamlit App

st.title("Customer Purchase Prediction")

st.write("Enter customer details to predict whether the customer will purchase.")


age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)


gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


annual_income = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=50000.0
)


spending_score = st.number_input(
    "Spending Score",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)


marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married"]
)


# Encoding

gender = 1 if gender == "Male" else 0

marital_status = 1 if marital_status == "Single" else 0


features = {
    "Age": age,
    "Gender": gender,
    "AnnualIncome": annual_income,
    "SpendingScore": spending_score,
    "MaritalStatus": marital_status
}


if st.button("Predict"):

    prediction, probability = preprocess_and_predict(features)

    if prediction == 1:
        st.success(
            f"HIGH chance of purchase (Probability: {probability:.2f})"
        )
    else:
        st.error(
            f"LOW chance of purchase (Probability: {probability:.2f})"
        )