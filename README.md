# Flight delay prediction model

Machine learning pipeline for predicting credit applications using transaction data and behavioral patterns of clients.

## Data Exploration and modelling

Jupyter notebooks were used for initial data exploration, feature construction and modelling. The notebooks can be found in `notebooks/`.

### Features

- Automated feature engineering pipeline
- Multiple model implementations:
  - Random Forest
  - Logistic Regression
  - LightGBM
- MLflow experiment tracking and model versioning
- Production-ready prediction pipeline
- Type hints and error handling
- Code quality enforcement via pre-commit hooks

## Repository Structure
```
abnamro_casestudy/
├── src/
│ ├── models.py # Model classes and pipeline definitions
│ ├── features.py # Feature engineering functions
│ ├── processing.py # Data preprocessing utilities
│ ├── train.py # Training script
│ ├── predict.py # Prediction script
│ ├── evaluation.py # logging evaluation metrics
│ ├── model_visualizations.py # plotting functions for model performance evaluations
│ └── constants.py # feature names and model parameters
├── dataset/ # Data files (not tracked)
│ ├── credit_applications.csv
│ └── customers.csv
├── configs/ # Configuration files
│ └── mlflow_config.yml
├── notebooks/ # Jupyter notebooks for exploration
│ ├── EDA.ipynb
│ └── model_exploration.ipynb
├── pyproject.toml # Poetry dependencies
└── .pre-commit-config.yaml # pre commit config
```


## Setup

1. Clone the repository:
```bash
git clone git@github.com:username/abnamro_casestudy.git
cd abnamro_casestudy
```

We use Poetry for dependency management under a `Python 3.11` environment. You can follow the steps below to install Poetry. Alternatively the dependencies are listed in `pyproject.toml` and you can install them in your own environment in your preferred manner.


2. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies (run from root):
```bash
poetry install
```

4. Setup pre-commit hooks:
```bash
poetry run pre-commit install
```

## Usage

### Taining
Train the model
```bash
poetry run python src/train.py --model-type logistic_regression
```
Available model types:
`random_forest`, `logistic_regression` (default), and `lightgbm` (check `poetry run python src/train.py --h` for more)

### Prediction
```bash
poetry run python src/predict.py predict \
    --app_data_path PATH/TO/DATA.csv \
    --cust_data_path PATH/TO/DATA.csv
```

## Development and dependency management

Run pre-commit hooks
```bash
poetry run pre-commit run --all-files
```

Add dependencies if needed
```bash
poetry add package_name
```

## Model tracking

We also setup a simple MLFlow server for model tracking. After training the models, you can access the metrics and other artifacts from the web UI.

1. Start UI
```
mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5000
```

2. Access Dashboard:
- Navigate to http://localhost:5000
- View experiments, metrics, and artifacts
- Compare model runs
- Download saved models

## Data Requirements

This repository assumes that the dataset fed to the models has the same schema as the one present in `/dataset` folder. The codebase does not contain schema checks hence it must be manually ensured that the input data contain these columns:

- `client_nr`: Client identifier
- `yearmonth`: Transaction month
- `total_nr_trx`: Total number of transactions
- `nr_debit_trx`: Number of debit transactions
- `volume_debit_trx`: Volume of debit transactions
- `nr_credit_trx`: Number of credit transactions
- `volume_credit_trx`: Volume of credit transactions
- `min_balance`: Minimum balance
- `max_balance`: Maximum balance
- `CRG`: Credit rating grade (this is an assumption from my side, I assumed CRG as an indicator of the client risk group)
- `credit_application`: Target variable (0/1)
