"""Configuration constants for the lead scoring pipeline."""

SEED = 42

FEATURES = [
    "recency_days",
    "fatigue_score",
    "sentiment_score",
    "engagement_rate_30d",
    "reply_rate_90d",
    "opt_out_risk",
    "mms_affinity",
]

# Logistic Regression hyperparameters
LR_PARAMS = {
    "C": 0.1,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": SEED,
}

