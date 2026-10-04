import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Initialize FastAPI app
app = FastAPI(title="Car Price Prediction API")

# 2. Load the saved pipeline artifacts from disk
model = joblib.load("car_price_model.pkl")
ohe = joblib.load("ohe_encoder.pkl")
scaler = joblib.load("scaler.pkl")

cat_cols = ["Fuel_Type", "Seller_Type", "Transmission"]
num_cols = ["Year", "Present_Price", "Kms_Driven", "Owner"]


# 3. Define the input data schema using Pydantic
class CarInput(BaseModel):
    Year: int
    Present_Price: float
    Kms_Driven: int
    Owner: int
    Fuel_Type: str  # e.g., 'Petrol', 'Diesel', 'CNG'
    Seller_Type: str  # e.g., 'Dealer', 'Individual'
    Transmission: str  # e.g., 'Manual', 'Automatic'


@app.get("/")
def home():
    return {
        "message": "Car Price Prediction API is running! Go to /docs to test it."
    }


# 4. Define the prediction endpoint
@app.post("/predict")
def predict_price(data: CarInput):
    # Convert incoming JSON request into a single-row DataFrame
    input_dict = data.model_dump()
    input_df = pd.DataFrame([input_dict])

    # Separate numerical and categorical features
    numeric_df = input_df[num_cols]
    categorical_df = input_df[cat_cols]

    # One-Hot Encode categorical features using saved encoder
    encoded_array = ohe.transform(categorical_df)
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=ohe.get_feature_names_out(cat_cols),
        index=input_df.index,
    )

    # Combine numeric and encoded features
    final_df = pd.concat([numeric_df, encoded_df], axis=1)

    # Scale features using saved scaler
    scaled_features = scaler.transform(final_df)

    # Make raw prediction & apply floor limit (minimum 0.05 Lakhs / ₹5,000)
    raw_pred = model.predict(scaled_features)[0]
    final_price = float(np.clip(raw_pred, a_min=0.05, a_max=None))

    return {
        "predicted_price_lakhs": round(final_price, 2),
        "predicted_price_inr": round(final_price * 100000),
    }