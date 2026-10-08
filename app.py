import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="Customer Purchase Prediction",
    page_icon="🛒",
    layout="centered"
)

st.title("Customer Purchase Prediction App")
st.write("Predict whether a customer will purchase a product.")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Streamli_task.csv")

df = load_data()

# Display dataset
st.subheader("Dataset Preview")
st.dataframe(df)

# Features and target
X = df.drop("Purchased", axis=1)
y = df["Purchased"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["Gender", "MaritalStatus"]
        )
    ],
    remainder="passthrough"
)

# Machine learning model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

model.fit(X, y)

# User inputs
st.subheader("Enter Customer Details")

age = st.number_input(
    "Age", min_value=1, max_value=100, value=25
)

gender = st.selectbox(
    "Gender", ["Male", "Female"]
)

income = st.number_input(
    "Annual Income", min_value=0, value=35000
)

score = st.slider(
    "Spending Score", 0, 100, 50
)

marital_status = st.selectbox(
    "Marital Status", ["Single", "Married"]
)

# Prediction
if st.button("Predict Purchase"):
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "AnnualIncome": [income],
        "SpendingScore": [score],
        "MaritalStatus": [marital_status]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("Prediction: Customer may purchase!")
    else:
        st.warning("Prediction: Customer may not purchase.")

st.info("Demo only: predictions use a very small dataset.")