# Design Document: Lead Scoring Weight Pipeline

## 1. System Architecture

The pipeline uses **Azure ML** for model lifecycle management and **Azure SQL Database** as the deterministic scoring engine. Weights are extracted from a Logistic Regression model, transformed into raw-feature space, and written to a SQL table — the production scoring query is a single weighted sum with no preprocessing logic.

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  SMS/MMS Platform ──► Azure Data Lake Storage Gen2 (ADLS)           │
│  CRM / NLP Pipeline     /raw/yyyy/mm/dd/                            │
│                         /processed/features/                         │
│                                                                      │
│       Campaign Outcomes ──► /labels/yyyy/mm/dd/  ◄── Feedback Loop  │
│                                                                      │
│                              │                                       │
│                              ▼                                       │
│                    Azure ML Workspace                                │
│                    ┌──────────────────────┐                          │
│                    │  ML Pipeline         │                          │
│                    │  1. Data Prep        │                          │
│                    │  2. Train LR         │                          │
│                    │  3. Evaluate & Cal.  │                          │
│                    │  4. Extract Weights  │                          │
│                    └──────────┬───────────┘                          │
│                               │                                      │
│                    Model Registry (versioned artifacts)              │
│                               │                                      │
│                               ▼                                      │
│                    Azure SQL Database                                │
│                    ┌──────────────────────┐                          │
│                    │ scoring_weights      │                          │
│                    │ (feature, weight,    │                          │
│                    │  offset, version)    │                          │
│                    └──────────┬───────────┘                          │
│                               │                                      │
│                               ▼                                      │
│                    Production Scoring (SQL weighted sum)             │
│                               │                                      │
│                               ▼                                      │
│                    Campaign Targeting                                │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Azure Monitor + ML Data Drift                                 │  │
│  │  • Feature PSI tracking (threshold: 0.2)                       │  │
│  │  • AUC monitoring on recent labeled data                       │  │
│  │  • Retraining triggers: drift / schedule / manual              │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

**Azure services:** ADLS Gen2 (partitioned storage), Azure Data Factory (data movement & feature engineering), Azure ML Workspace (compute, MLflow tracking, model registry), Azure SQL Database (`scoring_weights` table), Azure Monitor + ML Data Drift (alerting).

## 2. Data Flow & Feedback Loop

1. **Ingestion.** Raw campaign data (sends, clicks, replies, opt-outs) lands in ADLS Gen2 daily. NLP sentiment scores are pre-computed by a separate pipeline. Azure Data Factory computes the 7 features per contact and writes to `/processed/features/`.
2. **Label collection (feedback loop).** Campaign outcomes (positive reaction within N days of send) are written back to ADLS at `/labels/yyyy/mm/dd/` on a rolling basis. This closes the loop: each training run uses the latest labeled data, so the model adapts to changing contact behavior.
3. **Training.** The Azure ML Pipeline (Section 3) reads features + labels, trains, evaluates, and exports weights.
4. **Serving.** Azure SQL reads the `scoring_weights` table and scores contacts with a single `SELECT` — no model server, no feature preprocessing in SQL.

## 3. Training Pipeline (Azure ML)

Four steps, orchestrated as an Azure ML Pipeline. Each step is a registered component with tracked inputs/outputs.

### Step 1 — Data Prep & Scaling

- Pull latest labeled data from ADLS. Time-based split: train on older data, validate on recent.
- Fit `StandardScaler` on train split only. Record μ (mean) and σ (std) per feature — these are needed in Step 4.
- Features: `recency_days`, `fatigue_score`, `sentiment_score`, `engagement_rate_30d`, `reply_rate_90d`, `opt_out_risk`, `mms_affinity`.

### Step 2 — Train Logistic Regression

- `LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)`.
- **Why LR:** The requirement is deterministic weights for a SQL scoring engine that computes a weighted sum. LR minimizes log-loss while producing exactly this structure: `score = w · x + b`. Tree models require approximations (e.g., SHAP values) that disconnect the "weight" from the actual prediction.

### Step 3 — Evaluate & Calibrate

- **Metrics:** AUC-ROC, PR-AUC, Brier score, Precision@100, Precision@200.
- **Calibration check:** Plot calibration curve (10 bins). If Brier score exceeds 0.20, apply Platt scaling (`CalibratedClassifierCV`) and re-evaluate. LR is typically well-calibrated; this step acts as a safety net.
- **Safety gate:** If AUC drops more than 0.02 below the current production baseline → abort, alert the team, do **not** update weights.

### Step 4 — Extract & Transform Weights

Standard scaling is essential for training (ensures coefficients are comparable), but the SQL scoring engine receives **raw, unscaled** feature values. Rather than embedding normalization logic in every SQL query, we transform the model coefficients back into raw-feature space:

```
W_raw_i  = β_i / σ_i
Offset   = β_0 − Σ (β_i × μ_i / σ_i)
```

where β_i are the fitted coefficients on scaled features, β_0 is the intercept, and μ_i / σ_i are the scaler parameters.

**Why this matters:** The SQL scoring query becomes a single expression with no per-feature normalization:

```sql
SELECT
    (recency_days    * w_recency)
  + (fatigue_score   * w_fatigue)
  + (sentiment_score * w_sentiment)
  + (engagement_rate_30d * w_engagement)
  + (reply_rate_90d  * w_reply)
  + (opt_out_risk    * w_optout)
  + (mms_affinity    * w_mms)
  + offset
  AS lead_score
FROM contacts c
JOIN scoring_weights w ON w.is_active = 1;
```

No hardcoded feature ranges, no division by 60 for recency, no `(sentiment + 2) / 4` — the math is baked into the weights. This also preserves the exact logistic regression prediction: applying sigmoid to `lead_score` yields calibrated probabilities.

**Artifacts:** `weights.csv` (feature, raw_weight, offset, model_version) is logged to the AML Model Registry and synced to the `scoring_weights` table in Azure SQL.

## 4. SQL Integration

| Column | Type | Description |
|---|---|---|
| `model_version` | `INT` | Auto-incremented version |
| `feature` | `VARCHAR(50)` | Feature name |
| `weight` | `FLOAT` | Raw-space weight (W_raw) |
| `offset` | `FLOAT` | Combined intercept (same for all rows in a version) |
| `is_active` | `BIT` | Only one version active at a time |
| `created_at` | `DATETIME` | Timestamp |

- **Deployment** = insert new rows + flip `is_active`.
- **Rollback** = flip `is_active` back to the previous version. No query changes needed.

## 5. Monitoring & Retraining

| Trigger | Condition | Action |
|---|---|---|
| **Scheduled** | Weekly (Sunday 02:00 UTC) | Full pipeline run with latest labeled data |
| **Drift-based** | PSI > 0.2 on any feature | Immediate retraining via Azure Monitor alert |
| **Performance-based** | AUC drops > 5% from training baseline | Immediate retraining + team alert |
| **Manual** | New campaign type or seasonal shift | On-demand pipeline trigger |

**Drift detection:** Azure ML Data Drift Monitor compares weekly inference-time feature distributions against the training baseline using Population Stability Index (PSI). Alerts route through Azure Monitor.

**Reproducibility:** Every pipeline run logs the dataset version (AML Data Assets), code snapshot, scaler parameters, model artifact, and all metrics to MLflow. Fixed random seed (`42`) ensures deterministic splits and training.

