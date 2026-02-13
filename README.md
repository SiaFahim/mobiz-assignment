# ML-Driven Lead Scoring Weights

An ML solution that learns optimal feature weights from labeled campaign data for a deterministic SQL-based lead scoring pipeline. The weights drop directly into a `SELECT` statement — no model serving required.

## Notebook Experimentation

`notebook.ipynb` contains the full exploration and analysis:

1. **Data exploration** — 3,500 training / 1,200 validation rows across 7 features (`recency_days`, `fatigue_score`, `sentiment_score`, `engagement_rate_30d`, `reply_rate_90d`, `opt_out_risk`, `mms_affinity`). No missing values, moderate class imbalance (41% positive).
2. **Visualization** — Feature distributions, correlation heatmap, and feature-label correlations. Key finding: features are largely independent (no multicollinearity), with `recency_days` and `sentiment_score` showing the strongest label correlations.
3. **Three approaches compared:**

   - **Equal Weights** — baseline, every feature weighted the same
   - **Logistic Regression** — L2-regularized, coefficients map directly to SQL weights
   - **Gradient Boosting** — non-linear benchmark to check if we're leaving signal on the table
4. **Weight extraction** — Logistic Regression coefficients normalized so absolute values sum to 1.0, preserving direction and relative magnitude.

### Results

| Approach            | AUC-ROC | PR-AUC | Brier Score | P@100 | P@200 |
| ------------------- | ------- | ------ | ----------- | ----- | ----- |
| Equal Weights       | 0.7964  | 0.7193 | 0.2193      | 0.85  | 0.815 |
| Logistic Regression | 0.8244  | 0.7472 | 0.1664      | 0.86  | 0.835 |
| Gradient Boosting   | 0.7989  | 0.7114 | 0.1788      | 0.84  | 0.790 |

**Logistic Regression wins across all metrics.** It outperforms the equal-weight baseline by a clear margin and matches (or beats) the non-linear Gradient Boosting model — confirming that the feature-label relationships are mostly linear and a simple weighted sum captures the signal well.

## Production Module (`src/`)

The `src/` directory contains a lean, reproducible Python module extracted from the notebook — covering preprocessing, training, evaluation, and weight export. It's designed to run locally with identical results and deploy directly into an Azure ML pipeline.

See [`src/README.md`](src/README.md) for local run instructions and Azure ML integration guide.

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline
```

Weights are exported to `outputs/weights/weights.csv`.

## Project Structure

```
├── notebook.ipynb          # Full experimentation notebook
├── requirements.txt        # Python dependencies
├── src/                    # Production training pipeline
│   ├── config.py           # Feature list, seed, hyperparameters
│   ├── preprocess.py       # Data loading + StandardScaler
│   ├── train.py            # Training, evaluation, weight export
│   └── pipeline.py         # CLI entry point
├── data/
│   ├── train.csv           # Training set (3,500 rows)
│   └── valid.csv           # Validation set (1,200 rows)
├── outputs/weights/        # Exported weights
└── weights.csv             # Notebook-generated weights (reference)
```

Additional experiment: `alternative_solutions_ebm/ebm.ipynb` adds an **Explainable Boosting Machine (EBM)** benchmark (plus a tuned Gradient Boosting model) on the same validation split; the comparison table shows **Logistic Regression still best** (AUC-ROC **0.8244**, PR-AUC **0.7472**, Brier **0.1664**), with **EBM close behind** (AUC-ROC **0.8146**, PR-AUC **0.7355**) and tuned GB trailing (AUC-ROC **0.8035**). This indicates only mild non-linear lift, so LR remains the recommended deployable model while EBM is most valuable for its shape-function interpretability.
