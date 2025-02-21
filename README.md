# Flight delay prediction model

Machine learning pipeline for predicting whether a given flight will be delayed or not 2 hours before the scheduled departure. The analysis first defines what delay in flights are, and subsequently defines the target for modelling as a binary variable with the label being 1 when the calculated delay is more than 10 minutes, 0 zero for the delays less than or equal to 10 minutes. All the models are designed to predict whether a flight will be delayed or not 2 hours using information available at-most 2 hours before the scheduled departure of the flight.

We calculated different features taking into various factors into account such that the past delays, the number of flights being flown earlier, static flight features such as the airplane type, terminal, destination region, scheduled day and hour of the flight.


## Data Exploration and modelling

Jupyter notebooks were used for initial data exploration, feature construction and modelling. The notebooks can be found in `notebooks/`.

1. In `EDA.ipynb` we explore the data, and experiment with various features.
2. In `model_exploration.ipynb` we explore different modelling strategies by trying different features, and hyperparameters for 3 classifier models.


In the end, the best performing model had these metrics for the test set:

```
=== LightGBM Classification Report (Test) ===
              precision    recall  f1-score   support

           0      0.786     0.072     0.132     21457
           1      0.358     0.964     0.523     11547
```

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
schiphol_casestudy/
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
│ ├── flights.csv # not present in the repo cause it is larrrge
│ └── airports.csv # not present in the repo cause it is larrrge
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
git clone git@github.com:vvaibhav08/schiphol_assignment.git
cd schiphol_assignment
```

### Data
Place the airports and flights data as csv files in the dataset folder.


### Environment
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
poetry run python src/train.py --model-type "random_forest"
```
Available model types:
`random_forest`, `logistic_regression` (default), and `lightgbm` (check `poetry run python src/train.py --h` for more)

### Prediction
```bash
poetry run python src/predict.py predict \
    --flights_data_path PATH/TO/DATA.csv \
    --airports_data_path PATH/TO/DATA.csv
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

This repository assumes that the dataset fed to the models has the same schema as the one present in `/dataset` folder. The codebase does not contain schema checks hence it must be manually ensured that the input data contain columns defined in the [schiphol flights api](https://developer.schiphol.nl/apis/flight-api/overview?version=latest).
