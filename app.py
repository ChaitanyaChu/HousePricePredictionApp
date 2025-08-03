import streamlit as st
import pandas as pd
import pickle

# Load model and columns
with open("pricepred.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# App UI
st.set_page_config(page_title="🏠 USA Housing Price Predictor", layout="centered")

st.title("🏠 USA Housing Price Predictor")
st.markdown("Predict housing prices based on home features using a trained ML model.")

# Sidebar input features
st.sidebar.header("Input Features")
sqft_living = st.sidebar.slider("Living Area (sqft)", 200, 10000, 2000)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)
floors = st.sidebar.slider("Floors", 1, 3, 1)
waterfront = st.sidebar.selectbox("Waterfront View", [0, 1])
view = st.sidebar.slider("View Score", 0, 4, 1)
condition = st.sidebar.slider("Condition (1–5)", 1, 5, 3)
grade = st.sidebar.slider("Grade (1–13)", 1, 13, 7)

# Dynamic city options
city_options = [col for col in model_columns if col.startswith("city_grouped_")]
city_names = [c.replace("city_grouped_", "") for c in city_options]
city = st.sidebar.selectbox("City", ["Other"] + city_names)

# Build input dictionary
input_dict = {
    'sqft_living': sqft_living,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'floors': floors,
    'waterfront': waterfront,
    'view': view,
    'condition': condition,
    'grade': grade,
}

# Add city dummy variables
for col in city_options:
    input_dict[col] = 1 if col == f"city_grouped_{city}" else 0

# Create input DataFrame and fill missing columns if needed
input_df = pd.DataFrame([input_dict])
missing_cols = set(model_columns) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0

# Ensure correct column order
input_df = input_df[model_columns]

# Prediction
try:
    prediction = model.predict(input_df)[0]
    st.success(f"💵 Predicted Price: ${prediction:,.2f}")
except Exception as e:
    st.error(f"⚠️ Prediction failed: {e}")
