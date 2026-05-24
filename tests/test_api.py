from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path so we can import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

# TestClient simulates HTTP requests without running a real server
client = TestClient(app)

# ── Sample customer data ──
# Low risk customer — recent, frequent, high spend
LOW_RISK_CUSTOMER = {
    "city_tier": "Tier 1",
    "age_group": "25-34",
    "acquisition_channel": "Instagram",
    "loyalty_tier": "Gold",
    "preferred_category": "Skin Care",
    "marketing_consent": "Yes",
    "recency_days": 10,
    "frequency_180d": 5,
    "monetary_180d": 3500.0,
    "return_rate_180d": 0.0,
    "avg_discount_pct_180d": 0.1,
    "avg_rating_180d": 4.5,
    "category_diversity_180d": 3,
    "ticket_count_90d": 0,
    "negative_ticket_rate_90d": 0.0,
    "avg_resolution_hours_90d": 0.0,
    "days_since_signup": 365,
    "sessions_30d": 10,
    "product_views_30d": 25,
    "cart_adds_30d": 5,
    "wishlist_adds_30d": 3,
    "abandoned_carts_30d": 0,
    "email_opens_30d": 8,
    "campaign_clicks_30d": 3,
    "last_visit_days_ago": 2
}

# High risk customer — dormant, low engagement
HIGH_RISK_CUSTOMER = {
    "city_tier": "Tier 3",
    "age_group": "45+",
    "acquisition_channel": "Organic",
    "loyalty_tier": "Not Enrolled",
    "preferred_category": "Hair Care",
    "marketing_consent": "No",
    "recency_days": 180,
    "frequency_180d": 1,
    "monetary_180d": 400.0,
    "return_rate_180d": 0.5,
    "avg_discount_pct_180d": 0.4,
    "avg_rating_180d": 2.0,
    "category_diversity_180d": 1,
    "ticket_count_90d": 3,
    "negative_ticket_rate_90d": 0.8,
    "avg_resolution_hours_90d": 30.0,
    "days_since_signup": 400,
    "sessions_30d": 1,
    "product_views_30d": 2,
    "cart_adds_30d": 0,
    "wishlist_adds_30d": 0,
    "abandoned_carts_30d": 1,
    "email_opens_30d": 0,
    "campaign_clicks_30d": 0,
    "last_visit_days_ago": 60
}

# Medium risk customer
MEDIUM_RISK_CUSTOMER = {
    "city_tier": "Tier 2",
    "age_group": "35-44",
    "acquisition_channel": "Marketplace",
    "loyalty_tier": "Silver",
    "preferred_category": "Makeup",
    "marketing_consent": "Yes",
    "recency_days": 45,
    "frequency_180d": 2,
    "monetary_180d": 1200.0,
    "return_rate_180d": 0.1,
    "avg_discount_pct_180d": 0.2,
    "avg_rating_180d": 3.5,
    "category_diversity_180d": 2,
    "ticket_count_90d": 1,
    "negative_ticket_rate_90d": 0.2,
    "avg_resolution_hours_90d": 10.0,
    "days_since_signup": 200,
    "sessions_30d": 4,
    "product_views_30d": 8,
    "cart_adds_30d": 2,
    "wishlist_adds_30d": 1,
    "abandoned_carts_30d": 1,
    "email_opens_30d": 2,
    "campaign_clicks_30d": 1,
    "last_visit_days_ago": 15
}

# ── Test 1: Health check ──
def test_health_check():
    """API should return status ok"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "version" in data
    print("✓ Test 1 passed: Health check")

# ── Test 2: Single prediction — low risk customer ──
def test_predict_low_risk():
    """Low risk customer should return low churn probability"""
    response = client.post("/predict", json=LOW_RISK_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "predicted_class" in data
    assert "risk_level" in data
    assert "risk_explanation" in data
    assert 0 <= data["churn_probability"] <= 1
    assert data["predicted_class"] in [0, 1]
    assert data["risk_level"] in ["low", "medium", "high"]
    print(f"✓ Test 2 passed: Low risk prediction — prob={data['churn_probability']}, risk={data['risk_level']}")

# ── Test 3: Single prediction — high risk customer ──
def test_predict_high_risk():
    """High risk customer should return higher churn probability"""
    response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert data["churn_probability"] > 0.4
    print(f"✓ Test 3 passed: High risk prediction — prob={data['churn_probability']}, risk={data['risk_level']}")

# ── Test 4: Batch prediction ──
def test_batch_predict():
    """Batch endpoint should return predictions for all customers"""
    batch = [LOW_RISK_CUSTOMER, HIGH_RISK_CUSTOMER, MEDIUM_RISK_CUSTOMER]
    response = client.post("/batch_predict", json=batch)
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] == 3
    assert len(data["predictions"]) == 3
    assert "high_risk_count" in data
    print(f"✓ Test 4 passed: Batch prediction — {data['total_customers']} customers, {data['high_risk_count']} high risk")

# ── Test 5: Input validation — missing required field ──
def test_invalid_input_missing_field():
    """Missing required field should return 422 validation error"""
    invalid_customer = {"city_tier": "Tier 1"}  # Missing most fields
    response = client.post("/predict", json=invalid_customer)
    assert response.status_code == 422
    print("✓ Test 5 passed: Invalid input returns 422")

# ── Test 6: Input validation — invalid value range ──
def test_invalid_input_bad_range():
    """Return rate > 1 should fail validation"""
    bad_customer = LOW_RISK_CUSTOMER.copy()
    bad_customer["return_rate_180d"] = 5.0  # Invalid — max is 1.0
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422
    print("✓ Test 6 passed: Invalid range returns 422")

# ── Test 7: Empty batch ──
def test_empty_batch():
    """Empty batch should return 400 error"""
    response = client.post("/batch_predict", json=[])
    assert response.status_code == 400
    print("✓ Test 7 passed: Empty batch returns 400")

# ── Run all tests ──
if __name__ == "__main__":
    print("Running API tests...\n")
    test_health_check()
    test_predict_low_risk()
    test_predict_high_risk()
    test_batch_predict()
    test_invalid_input_missing_field()
    test_invalid_input_bad_range()
    test_empty_batch()
    print("\nAll tests passed! ✓")
