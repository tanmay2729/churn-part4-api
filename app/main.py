import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os

# ── Load model and encoders at startup ──
# Loading once at startup is efficient — not on every request
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), '..', 'label_encoders.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(ENCODER_PATH, 'rb') as f:
    label_encoders = pickle.load(f)

# ── FastAPI app ──
app = FastAPI(
    title="D2C Churn Scoring API",
    description="Predicts customer churn probability for a D2C personal-care brand",
    version="1.0.0"
)

# ── Feature lists (must match Part 3 exactly) ──
CATEGORICAL_FEATURES = [
    'city_tier', 'age_group', 'acquisition_channel',
    'loyalty_tier', 'preferred_category', 'marketing_consent'
]

NUMERICAL_FEATURES = [
    'recency_days', 'frequency_180d', 'monetary_180d',
    'return_rate_180d', 'avg_discount_pct_180d', 'avg_rating_180d',
    'category_diversity_180d', 'ticket_count_90d',
    'negative_ticket_rate_90d', 'avg_resolution_hours_90d',
    'days_since_signup', 'sessions_30d', 'product_views_30d',
    'cart_adds_30d', 'wishlist_adds_30d', 'abandoned_carts_30d',
    'email_opens_30d', 'campaign_clicks_30d', 'last_visit_days_ago'
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

THRESHOLD = 0.5

# ── Pydantic input schema ──
# Pydantic validates every incoming request automatically
# If a required field is missing or wrong type — FastAPI returns 422 error
class CustomerFeatures(BaseModel):
    # Categorical features
    city_tier: str = Field(..., example="Tier 1",
                           description="City classification: Tier 1, Tier 2, Tier 3")
    age_group: str = Field(..., example="25-34",
                           description="Age bracket: 18-24, 25-34, 35-44, 45+")
    acquisition_channel: str = Field(..., example="Instagram",
                                     description="Marketing channel")
    loyalty_tier: str = Field(default="Not Enrolled", example="Silver",
                              description="Loyalty tier: Silver, Gold, Platinum, Not Enrolled")
    preferred_category: str = Field(..., example="Skin Care",
                                    description="Product category preference")
    marketing_consent: str = Field(..., example="Yes",
                                   description="Marketing consent: Yes or No")

    # RFM features
    recency_days: int = Field(..., ge=0,
                              description="Days since last purchase")
    frequency_180d: int = Field(..., ge=0, example=3,
                                description="Number of orders in last 180 days")
    monetary_180d: float = Field(..., ge=0, example=1500.0,
                                 description="Total spend in last 180 days (INR)")

    # Order behaviour
    return_rate_180d: float = Field(..., ge=0, le=1, example=0.1,
                                    description="Proportion of orders returned")
    avg_discount_pct_180d: float = Field(..., ge=0, le=1, example=0.2,
                                         description="Average discount used")
    avg_rating_180d: float = Field(default=3.0, ge=1, le=5, example=4.0,
                                   description="Average order rating")
    category_diversity_180d: int = Field(..., ge=0, example=2,
                                         description="Number of distinct categories purchased")

    # Support features
    ticket_count_90d: int = Field(..., ge=0, example=1,
                                  description="Support tickets in last 90 days")
    negative_ticket_rate_90d: float = Field(..., ge=0, le=1, example=0.0,
                                            description="Proportion of negative tickets")
    avg_resolution_hours_90d: float = Field(..., ge=0, example=5.0,
                                            description="Average ticket resolution time")

    # Engagement features
    days_since_signup: int = Field(..., ge=0, example=300,
                                   description="Days since account creation")
    sessions_30d: int = Field(..., ge=0, example=5,
                              description="App sessions in last 30 days")
    product_views_30d: int = Field(..., ge=0, example=10,
                                   description="Product pages viewed in last 30 days")
    cart_adds_30d: int = Field(..., ge=0, example=2,
                               description="Items added to cart in last 30 days")
    wishlist_adds_30d: int = Field(..., ge=0, example=1,
                                   description="Items added to wishlist in last 30 days")
    abandoned_carts_30d: int = Field(..., ge=0, example=1,
                                     description="Abandoned cart sessions in last 30 days")
    email_opens_30d: int = Field(..., ge=0, example=3,
                                 description="Marketing emails opened in last 30 days")
    campaign_clicks_30d: int = Field(..., ge=0, example=1,
                                     description="Campaign clicks in last 30 days")
    last_visit_days_ago: int = Field(..., ge=0, example=7,
                                     description="Days since last app/website visit")

# ── Pydantic output schema ──
class PredictionResponse(BaseModel):
    customer_id: Optional[str] = None
    churn_probability: float
    predicted_class: int
    risk_level: str
    risk_explanation: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int

# ── Helper function ──
def preprocess_and_predict(customer: CustomerFeatures, customer_id: str = None):
    """
    Preprocesses customer features and returns prediction.
    Steps:
    1. Convert to DataFrame
    2. Apply label encoders to categorical columns
    3. Run model prediction
    4. Generate risk explanation
    """
    # Convert input to DataFrame
    data = pd.DataFrame([customer.model_dump()])

    # Apply label encoders — same encoding used during training
    for col in CATEGORICAL_FEATURES:
        le = label_encoders[col]
        val = data[col].astype(str).values[0]
        # Handle unseen categories gracefully
        if val in le.classes_:
            data[col] = le.transform([val])
        else:
            # Use most common class (index 0) for unknown categories
            data[col] = 0

    # Select features in correct order
    X = data[ALL_FEATURES]

    # Get churn probability
    # predict_proba returns [prob_no_churn, prob_churn]
    proba = model.predict_proba(X)[0][1]
    predicted_class = int(proba >= THRESHOLD)

    # Risk level based on probability
    if proba >= 0.75:
        risk_level = "high"
    elif proba >= 0.45:
        risk_level = "medium"
    else:
        risk_level = "low"

    # Generate risk explanation based on top signals
    explanations = []
    if customer.recency_days > 90:
        explanations.append(f"no purchase in {customer.recency_days} days")
    if customer.sessions_30d < 3:
        explanations.append("low app engagement")
    if customer.return_rate_180d > 0.3:
        explanations.append("high return rate")
    if customer.negative_ticket_rate_90d > 0.5:
        explanations.append("negative support interactions")
    if customer.frequency_180d < 2:
        explanations.append("low purchase frequency")
    if customer.last_visit_days_ago > 30:
        explanations.append(f"not visited in {customer.last_visit_days_ago} days")

    if explanations:
        risk_explanation = f"{', '.join(explanations).capitalize()} indicate {risk_level} churn risk."
    else:
        risk_explanation = "Customer shows healthy engagement patterns with low churn risk."

    return PredictionResponse(
        customer_id=customer_id,
        churn_probability=round(float(proba), 4),
        predicted_class=predicted_class,
        risk_level=risk_level,
        risk_explanation=risk_explanation
    )

# ── Endpoints ──

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    Returns API status and model info.
    """
    return {
        "status": "ok",
        "model": "Logistic Regression",
        "version": "1.0.0",
        "threshold": THRESHOLD,
        "message": "D2C Churn Scoring API is running"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures, customer_id: Optional[str] = None):
    """
    Single customer churn prediction.
    Accepts one customer feature payload and returns churn risk response.
    """
    try:
        return preprocess_and_predict(customer, customer_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict(customers: List[CustomerFeatures]):
    """
    Batch prediction for multiple customers.
    Accepts a list of customer feature payloads.
    Returns predictions for each customer.
    """
    if len(customers) == 0:
        raise HTTPException(status_code=400, detail="No customers provided")
    if len(customers) > 1000:
        raise HTTPException(status_code=400, detail="Batch size exceeds limit of 1000")

    try:
        predictions = []
        for i, customer in enumerate(customers):
            pred = preprocess_and_predict(customer, customer_id=f"batch_{i+1}")
            predictions.append(pred)

        high_risk = sum(1 for p in predictions if p.risk_level == "high")

        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(predictions),
            high_risk_count=high_risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
