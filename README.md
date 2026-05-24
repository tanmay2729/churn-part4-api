# Part 4: FastAPI Churn Scoring Service
## D2C Customer Churn Intelligence & Retention API — Capstone

---

## Project Overview

This is a REST API service that loads a trained churn prediction model and exposes endpoints for the CRM and retention teams to get real-time churn risk scores for customers.

**Business purpose:** Allow non-technical teams to get churn predictions for any customer by sending a simple HTTP request — no Python or ML knowledge required.

---

## Repository Structure

```
churn-part4-api/
│
├── app/
│   ├── __init__.py
│   └── main.py              ← FastAPI application
├── tests/
│   └── test_api.py          ← 7 API test cases
├── model.pkl                ← Trained Logistic Regression model
├── label_encoders.pkl       ← Categorical feature encoders
├── monitoring_plan.md       ← Post-deployment monitoring plan
├── requirements.txt         ← Python dependencies
└── README.md
```

---

## Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/tanmay2729/churn-part4-api.git
cd churn-part4-api

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the API

```bash
uvicorn app.main:app --reload
```

The API will start at: `http://127.0.0.1:8000`

**Interactive API docs:** `http://127.0.0.1:8000/docs`

---

## API Endpoints

### GET /health
Health check — confirms API is running.

**Response:**
```json
{
  "status": "ok",
  "model": "Logistic Regression",
  "version": "1.0.0",
  "threshold": 0.5,
  "message": "D2C Churn Scoring API is running"
}
```

---

### POST /predict
Single customer churn prediction.

**Sample Request:**
```json
{
  "city_tier": "Tier 2",
  "age_group": "25-34",
  "acquisition_channel": "Instagram",
  "loyalty_tier": "Silver",
  "preferred_category": "Skin Care",
  "marketing_consent": "Yes",
  "recency_days": 45,
  "frequency_180d": 2,
  "monetary_180d": 1500.0,
  "return_rate_180d": 0.1,
  "avg_discount_pct_180d": 0.2,
  "avg_rating_180d": 3.5,
  "category_diversity_180d": 2,
  "ticket_count_90d": 1,
  "negative_ticket_rate_90d": 0.0,
  "avg_resolution_hours_90d": 5.0,
  "days_since_signup": 300,
  "sessions_30d": 5,
  "product_views_30d": 10,
  "cart_adds_30d": 2,
  "wishlist_adds_30d": 1,
  "abandoned_carts_30d": 1,
  "email_opens_30d": 3,
  "campaign_clicks_30d": 1,
  "last_visit_days_ago": 7
}
```

**Sample Response:**
```json
{
  "customer_id": null,
  "churn_probability": 0.3821,
  "predicted_class": 0,
  "risk_level": "low",
  "risk_explanation": "Customer shows healthy engagement patterns with low churn risk."
}
```

---

### POST /batch_predict
Batch prediction for multiple customers.

**Request:** Array of customer objects (same format as /predict)

**Sample Response:**
```json
{
  "predictions": [...],
  "total_customers": 3,
  "high_risk_count": 1
}
```

---

## Running Tests

```bash
pip install pytest httpx
pytest tests/test_api.py -v
```

**Expected output:**
```
tests/test_api.py::test_health_check              PASSED
tests/test_api.py::test_predict_low_risk          PASSED
tests/test_api.py::test_predict_high_risk         PASSED
tests/test_api.py::test_batch_predict             PASSED
tests/test_api.py::test_invalid_input_missing_field PASSED
tests/test_api.py::test_invalid_input_bad_range   PASSED
tests/test_api.py::test_empty_batch               PASSED
7 passed
```

---

## Model & Source Data Notes

| Field | Details |
|---|---|
| Model | Logistic Regression (sklearn Pipeline) |
| Trained in | Part 3 — churn-part3-model |
| Training data | rfm_modeling_snapshot.csv (2,400 customers) |
| Snapshot date | 2025-09-30 |
| Target | churn_next_60d (60-day churn window) |
| Threshold | 0.5 |
| Test ROC-AUC | 0.8856 |
| Test F1 | 0.8871 |

**Loading the model manually:**
```python
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
```

---

## Risk Level Logic

| Churn Probability | Risk Level | Recommended Action |
|---|---|---|
| >= 0.75 | High | Immediate personal outreach |
| 0.45 - 0.74 | Medium | Targeted retention campaign |
| < 0.45 | Low | Standard engagement only |
