import streamlit as st
import pickle
import numpy as np

# Load the model (cached so it loads only once)
@st.cache_resource
def load_model():
    with open('catboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Define your features and their explanations
feature_info = {
    'Year': 'The calendar year of the data point (e.g., 2023).',
    'DE1': 'Economic indicator DE1 - specific to dataset domain.',
    'DE2': 'Economic indicator DE2 - specific to dataset domain.',
    'DGS5': '5-Year Treasury Constant Maturity Rate (percent).',
    'DTB6': '6-Month Treasury Bill Rate (percent).',
    'DTB4WK': '4-Week Treasury Bill Rate (percent).',
    'Month': 'Month of the year as a number (1-12).',
    'DE5': 'Economic indicator DE5 - specific to dataset domain.',
    'DE4': 'Economic indicator DE4 - specific to dataset domain.',
    'DGS10': '10-Year Treasury Constant Maturity Rate (percent).'
}

st.title("Stock Market Prediction")

st.write("Enter values below (Tap on the info icon for description).")

features = []
for feature_name, explanation in feature_info.items():
    val = st.number_input(
        label=f"{feature_name} ℹ️",  # Add info icon to label
        value=0.0,
        help=explanation  # Tooltip shown on hover
    )
    features.append(val)

input_array = np.array(features).reshape(1, -1)

if st.button("Predict"):
    prediction = model.predict(input_array)
    st.success(f"Predicted value: {prediction[0]:.4f}")
