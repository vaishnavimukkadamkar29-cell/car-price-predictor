import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.title("🚗 Used Car Price Predictor")

# Load model artifacts directly
model = joblib.load("car_price_model.pkl")
ohe = joblib.load("ohe_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# User Inputs
year = st.number_input("Year of Purchase", 2000, 2026, 2018)
present_price = st.number_input("Present Showroom Price (in Lakhs)", 0.5, 50.0, 6.5)
kms = st.number_input("Kilometers Driven", 0, 500000, 25000)
owner = st.selectbox("Previous Owners", [0, 1, 2, 3])
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
trans = st.selectbox("Transmission", ["Manual", "Automatic"])

if st.button("Predict Price"):
    # 1. Format input DataFrame
    raw_df = pd.DataFrame(
        [{
            "Year": year,
            "Present_Price": present_price,
            "Kms_Driven": kms,
            "Owner": owner,
            "Fuel_Type": fuel,
            "Seller_Type": seller,
            "Transmission": trans,
        }]
    )

    num_cols = ["Year", "Present_Price", "Kms_Driven", "Owner"]
    cat_cols = ["Fuel_Type", "Seller_Type", "Transmission"]

    # 2. Process inputs
    num_df = raw_df[num_cols]
    enc_df = pd.DataFrame(
        ohe.transform(raw_df[cat_cols]),
        columns=ohe.get_feature_names_out(cat_cols),
    )
    final_df = pd.concat([num_df, enc_df], axis=1)

    # 3. Scale and predict
    scaled_df = scaler.transform(final_df)
    raw_pred = model.predict(scaled_df)[0]
    final_price = np.clip(raw_pred, a_min=0.05, a_max=None)

    # Calculate price range using MAE (1.21 Lakhs)
    mae = 1.21
    min_price = max(0.05, final_price - mae)
    max_price = final_price + mae

    # --- Trust & Confidence UI Output ---
    st.markdown("---")
    st.subheader("💡 Estimated Market Price")
    st.metric(
        label="Predicted Selling Price", value=f"₹{final_price:.2f} Lakhs"
    )

    st.info(
        f"📊 **Expected Price Range:** ₹{min_price:.2f} Lakhs – ₹{max_price:.2f} Lakhs\n\n"
        f"*(Based on model Mean Absolute Error of ±₹1.21 Lakhs with an R² accuracy score of 85%)*"
    )