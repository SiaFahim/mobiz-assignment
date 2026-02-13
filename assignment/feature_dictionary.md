# Feature Dictionary

## SMS/MMS lead scoring features

- `recency_days`: Days since last positive response (lower is better).
- `fatigue_score`: Contact fatigue from over-messaging, 0-1 (lower is better).
- `sentiment_score`: Recent sentiment score from NLP, -2 to 2 (higher is better).
- `engagement_rate_30d`: Click/engage frequency in last 30 days, 0-1.
- `reply_rate_90d`: Reply frequency in last 90 days, 0-1.
- `opt_out_risk`: Predicted unsubscribe risk, 0-1 (lower is better).
- `mms_affinity`: Likelihood to engage with MMS rich media, 0-1.
- `label`: Binary target where 1 means positive reaction.
