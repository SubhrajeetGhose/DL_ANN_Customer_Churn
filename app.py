import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")

# ------------------------------
# Helper function to load pickle safely
def load_pickle_file(filename, description):
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"⚠️ Failed to load {description}: {e}")
            return None
    else:
        st.error(f"⚠️ {description} file not found: {filename}")
        return None

# ------------------------------
# Load model
model = None
with st.spinner("Loading model..."):
    try:
        if os.path.exists("model.h5"):
            model = tf.keras.models.load_model("model.h5")
        else:
            st.error("❌ Model file `model.h5` not found. Please add it to the repo.")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")

# ------------------------------
# Load encoders & scaler
label_encoder_gender = load_pickle_file("label_encoder_gender.pkl", "Label Encoder (Gender)")
onehot_encoder_geo = load_pickle_file("onehot_encoder_geo.pkl", "OneHot Encoder (Geography)")
scaler = load_pickle_file("scaler.pkl", "Scaler")

# ------------------------------
# Run only if everything is loaded
if model and label_encoder_gender and onehot_encoder_geo and scaler:

    st.subheader("Enter Customer Information")

    geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
    gender = st.selectbox("Gender", label_encoder_gender.classes_)
    age = st.slider("Age", 18, 92, 30)
    balance = st.number_input("Balance", min_value=0.0, step=100.0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1)
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=100.0)
    tenure = st.slider("Tenure", 0, 10, 3)
    num_of_products = st.slider("Number of Products", 1, 4, 1)
    has_cr_card = st.selectbox("Has Credit Card", [0, 1])
    is_active_member = st.selectbox("Is Active Member", [0, 1])

    if st.button("🔮 Predict"):
        try:
            # Prepare input
            input_data = pd.DataFrame({
                "CreditScore": [credit_score],
                "Gender": [label_encoder_gender.transform([gender])[0]],
                "Age": [age],
                "Tenure": [tenure],
                "Balance": [balance],
                "NumOfProducts": [num_of_products],
                "HasCrCard": [has_cr_card],
                "IsActiveMember": [is_active_member],
                "EstimatedSalary": [estimated_salary]
            })

            # One-hot encode Geography
            geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
            geo_encoded_df = pd.DataFrame(
                geo_encoded,
                columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
            )

            # Combine
            input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

            # Scale
            input_data_scaled = scaler.transform(input_data)

            # Predict
            with st.spinner("Predicting..."):
                prediction = model.predict(input_data_scaled, verbose=0)
                prediction_proba = float(prediction[0][0])

            # Display result
            st.success(f"Churn Probability: **{prediction_proba:.2f}**")
            if prediction_proba > 0.5:
                st.error("⚠️ The customer is **likely to churn**.")
            else:
                st.info("✅ The customer is **not likely to churn**.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.warning("⚠️ App is not fully functional because some files failed to load.")
