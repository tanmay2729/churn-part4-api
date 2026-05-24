# Monitoring Plan
## D2C Churn Scoring API — Post-Deployment Monitoring

---

## Overview

Once the churn scoring API is deployed, it must be actively monitored to ensure predictions remain accurate, the service stays healthy, and business outcomes improve over time.

---

## 1. Data Drift Monitoring

**What is data drift?**
Data drift occurs when the statistical distribution of incoming customer data changes over time compared to the training data. If drift is detected, model predictions become unreliable.

**What to monitor:**

| Feature | Method | Alert Threshold |
|---|---|---|
| `recency_days` | Compare monthly mean vs training mean | >20% change |
| `frequency_180d` | Distribution comparison (KS test) | p-value < 0.05 |
| `monetary_180d` | Compare monthly median vs training median | >25% change |
| `sessions_30d` | Rolling 30-day average vs baseline | >30% change |
| Categorical distributions | Chi-square test on city_tier, age_group | p-value < 0.05 |

**How to monitor:**
- Log all incoming feature values to a database
- Run weekly statistical tests comparing recent inputs vs training data
- Alert the data science team when drift is detected

---

## 2. Prediction Distribution Monitoring

**What to monitor:**

| Metric | Description | Alert Threshold |
|---|---|---|
| Daily churn rate | % of customers predicted as churners | >60% or <20% on any day |
| Avg churn probability | Mean probability across all predictions | Shifts by >10% week-over-week |
| Risk level distribution | % high/medium/low per day | High risk > 50% of predictions |
| Score distribution | Histogram of churn probabilities | Bimodal collapse or spike at extremes |

**Why this matters:**
If suddenly 90% of customers are predicted as high risk, either the model is broken or something unusual happened in the business (e.g. a bad product launch).

**How to monitor:**
- Log every prediction with timestamp to a database
- Build a simple dashboard (e.g. in Metabase or Grafana) showing daily prediction distributions
- Set up automated alerts when thresholds are breached

---

## 3. Business Outcomes Monitoring

**What to monitor:**

| Metric | Description | Target |
|---|---|---|
| Retention rate | % of predicted churners who stayed after intervention | >30% improvement vs no intervention |
| Campaign ROI | Revenue saved vs cost of retention campaign | Positive ROI within 60 days |
| False negative cost | Revenue lost from missed churners | Track monthly |
| False positive cost | Wasted retention spend on non-churners | < 15% of total campaign budget |
| Actual vs predicted churn | Compare model predictions vs actual churn after 60 days | Within 5% of predicted rate |

**How to monitor:**
- Join prediction logs with actual churn outcomes after 60 days
- Calculate precision and recall on live data monthly
- Report to business stakeholders monthly

---

## 4. API Health & Error Monitoring

**What to monitor:**

| Metric | Description | Alert Threshold |
|---|---|---|
| Response time | Time to return prediction | >2 seconds average |
| Error rate | % of requests returning 4xx or 5xx | >1% error rate |
| Uptime | API availability | <99.5% uptime |
| Request volume | Number of API calls per hour | Sudden spike or drop >50% |
| 422 validation errors | Invalid input rate | >5% of requests |

**How to monitor:**
- Use uvicorn access logs to track response times and error rates
- Set up health check pings every 5 minutes
- Alert on-call engineer if API returns 500 errors

---

## 5. Retraining Triggers

The model should be retrained when ANY of the following conditions are met:

| Trigger | Condition |
|---|---|
| Performance degradation | Live ROC-AUC drops below 0.80 |
| Data drift detected | KS test p-value < 0.05 on 2+ features |
| Prediction distribution shift | Daily predicted churn rate shifts >15% for 7+ days |
| Business calendar event | Major product launch, price change, or seasonal shift |
| Scheduled retraining | Every 3 months regardless of performance |
| New data available | >500 new labelled customers available |

**Retraining process:**
1. Collect new labelled data (customers with known churn outcomes)
2. Retrain model using same pipeline as Part 3
3. Evaluate on held-out test set — must beat current model's F1
4. If better → deploy new model.pkl to API
5. Keep old model.pkl as backup for rollback

---

## 6. Responsible Use Note

### How the retention team SHOULD use this API:

- Use churn probability as **one signal** among many — not the sole decision maker
- Prioritise outreach for customers with probability > 0.70 (high risk)
- Combine API output with human judgment for high-value customers
- Use batch predictions weekly to plan campaign calendars
- Respect `marketing_consent = No` — never trigger campaigns for these customers
- Document all interventions so outcomes can be tracked

### How the retention team should NOT use this API:

- Do NOT use churn probability to deny customers any service or benefit
- Do NOT use predictions to discriminate based on city_tier or age_group
- Do NOT treat the model as infallible — always allow for human override
- Do NOT use the API output to make permanent account decisions (e.g. cancellation)
- Do NOT share raw prediction scores with customers
- Do NOT use stale predictions (>30 days old) for campaign decisions
- Do NOT skip the monitoring plan — unmonitored models degrade silently

---

## Summary Dashboard (Recommended)

Build a simple weekly monitoring dashboard with:
- Total predictions made this week
- % high / medium / low risk
- API uptime and error rate
- Feature drift alerts
- Actual vs predicted churn rate (60-day lag)
- Campaign ROI from previous cycle
