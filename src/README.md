# Lead Scoring — Training Pipeline

Logistic Regression pipeline that learns feature weights from labeled campaign data and exports them as a CSV for SQL-based lead scoring.

## Project Structure

```
src/
├── config.py       # Feature list, seed, LR hyperparameters
├── preprocess.py   # Data loading + StandardScaler
├── train.py        # Model training, evaluation, weight export
└── pipeline.py     # CLI entry point — chains the full flow
outputs/
└── weights/
    └── weights.csv # Exported normalized weights
data/
├── train.csv
└── valid.csv
```

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# From the project root (where data/ lives)
python -m src.pipeline
```

Custom paths:

```bash
python -m src.pipeline \
  --train data/train.csv \
  --valid data/valid.csv \
  --output outputs/weights
```

**Expected output:**

| Metric  | Value  |
|---------|--------|
| AUC-ROC | 0.8244 |
| PR-AUC  | 0.7472 |
| Brier   | 0.1664 |
| P@100   | 0.8600 |
| P@200   | 0.8350 |

Weights are saved to `outputs/weights/weights.csv`.

## Requirements

- Python 3.9+
- scikit-learn, pandas, numpy (pinned versions in `requirements.txt`)

```bash
pip install -r requirements.txt
```

## Azure ML Pipeline Integration

The module is designed to plug directly into an Azure ML pipeline. Each function maps to a pipeline step.

### 1. Register as an Azure ML Component

```yaml
# azure_ml/train_component.yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandComponent.schema.json
name: lead_scoring_train
display_name: Lead Scoring — Train & Export Weights
type: command
inputs:
  train_data:
    type: uri_file
  valid_data:
    type: uri_file
outputs:
  weights:
    type: uri_folder
command: >-
  python -m src.pipeline
  --train ${{inputs.train_data}}
  --valid ${{inputs.valid_data}}
  --output ${{outputs.weights}}
environment:
  image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04
  conda_file: environment.yaml
```

### 2. Define the Pipeline

```python
from azure.ai.ml import MLClient, Input, load_component
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<sub-id>",
    resource_group_name="<rg>",
    workspace_name="<workspace>",
)

train_component = load_component(source="azure_ml/train_component.yaml")

@pipeline(description="Lead scoring weight training")
def lead_scoring_pipeline(train_data, valid_data):
    train_step = train_component(
        train_data=train_data,
        valid_data=valid_data,
    )
    return {"weights": train_step.outputs.weights}

# Submit
pipeline_job = lead_scoring_pipeline(
    train_data=Input(type="uri_file", path="azureml://datastores/data/paths/train.csv"),
    valid_data=Input(type="uri_file", path="azureml://datastores/data/paths/valid.csv"),
)
ml_client.jobs.create_or_update(pipeline_job, experiment_name="lead-scoring")
```

### 3. Schedule Retraining

```python
from azure.ai.ml.entities import RecurrenceTrigger, JobSchedule

schedule = JobSchedule(
    name="weekly-retrain",
    trigger=RecurrenceTrigger(frequency="week", interval=1),
    create_job=pipeline_job,
)
ml_client.schedules.begin_create_or_update(schedule)
```

### 4. Post-Training: Push Weights to Azure SQL

After the pipeline runs, read `outputs/weights/weights.csv` and insert into the `scoring_weights` table that the production SQL scoring query reads from. This can be a downstream pipeline step or a simple script triggered on pipeline completion.

