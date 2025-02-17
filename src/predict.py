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
    app_data_path: Path = Path("../data/application_data.csv"),
    cust_data_path: Path = Path("../data/customer_data.csv"),
    config_path: Path = Path("../config/mlflow_config.yml"),
    run_id: str | None = None,
) -> pd.Series:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    if run_id is None:
        run_id = get_latest_run_id(config["mlflow"]["experiment_name"])

    df_app = load_csv_data(app_data_path)
    df_cust = load_csv_data(cust_data_path)
    df = merge_data(df_app, df_cust)
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
    app_data_path = Path("dataset/credit_applications.csv")
    cust_data_path = Path("dataset/customers.csv")
    config_path = Path("config/mlflow_config.yaml")

    parser = argparse.ArgumentParser(description="Generate credit application predictions")
    parser.add_argument("--app-data-path", type=Path, default=app_data_path, help="Path to application data CSV")
    parser.add_argument("--cust-data-path", type=Path, default=cust_data_path, help="Path to customer data CSV")
    parser.add_argument("--config-path", type=Path, default=config_path, help="Path to MLflow config")

    args = parser.parse_args()

    predictions = predict(
        app_data_path=Path(args.app_data_path),
        cust_data_path=Path(args.cust_data_path),
        config_path=Path(args.config_path),
        run_id=args.run_id,
    )

    print(f"Total predictions: {len(predictions)}")
    print(f"Positive predictions: {(predictions == 1).sum()}")
