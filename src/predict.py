from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from src.constants import CLASSIFICATION_THRESHOLD, SELECTED_BEST_MODEL_FEATURES
from src.features import create_features
from src.processing import load_csv_data, merge_data


def get_latest_run_id(experiment_name: str) -> str:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment {experiment_name} not found")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        raise ValueError("No runs found in the experiment")

    return runs.iloc[0].run_id


def predict(
    flights_data_path: Path = Path("../data/flights.csv"),
    airports_data_path: Path = Path("../data/airports.csv"),
    config_path: Path = Path("../config/mlflow_config.yml"),
    run_id: str | None = None,
) -> pd.Series:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    if run_id is None:
        run_id = get_latest_run_id(config["mlflow"]["experiment_name"])

    df_flights = load_csv_data(flights_data_path)
    df_airports = load_csv_data(airports_data_path)
    df = merge_data(df_flights, df_airports)
    df = create_features(df)
    features = SELECTED_BEST_MODEL_FEATURES
    df = df.dropna(subset=features)
    X = df[features]

    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= CLASSIFICATION_THRESHOLD).astype(int)
    return pd.Series(y_pred, index=df.index)


if __name__ == "__main__":
    import argparse

    base_dir = Path(__file__).parent.parent
    flights_data_path = Path("dataset/flights.csv")
    airports_data_path = Path("dataset/airports.csv")
    config_path = Path("config/mlflow_config.yaml")

    parser = argparse.ArgumentParser(description="Generate credit application predictions")
    parser.add_argument("--flights-data-path", type=Path, default=flights_data_path, help="Path to flights data CSV")
    parser.add_argument("--airports-data-path", type=Path, default=airports_data_path, help="Path to airports data CSV")
    parser.add_argument("--config-path", type=Path, default=config_path, help="Path to MLflow config")

    args = parser.parse_args()

    predictions = predict(
        flights_data_path=Path(args.flights_data_path),
        airports_data_path=Path(args.airports_data_path),
        config_path=Path(args.config_path),
        run_id=args.run_id,
    )

    print(f"Total predictions: {len(predictions)}")
    print(f"Positive predictions: {(predictions == 1).sum()}")
