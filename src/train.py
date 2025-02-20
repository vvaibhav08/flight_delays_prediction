from pathlib import Path

import mlflow
import mlflow.sklearn
import yaml
from sklearn.model_selection import train_test_split

from src.constants import SELECTED_BEST_MODEL_FEATURES, RawFeatures
from src.evaluation import evaluate_and_log_metrics
from src.features import create_features
from src.models import LightGBMModel, LogisticRegressionModel, RandomForestModel
from src.processing import load_csv_data, merge_data


def _get_model_pipeline(model_type: str):
    model_map = {
        "logistic_regression": LogisticRegressionModel,
        "random_forest": RandomForestModel,
        "lightgbm": LightGBMModel,
    }
    model_class = model_map.get(model_type)
    if not model_class:
        raise ValueError(f"Unknown model type: {model_type}")
    return model_class().get_pipeline()


def train_model(
    flights_data_path: Path = Path("../data/flights.csv"),
    airports_data_path: Path = Path("../data/airports.csv"),
    config_path: Path = Path("../config/mlflow_config.yml"),
    model_type: str = "logistic_regression",
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Load and merge data
    df_flights = load_csv_data(flights_data_path)
    df_airports = load_csv_data(airports_data_path)

    df_flights_with_regions = merge_data(df_flights, df_airports)

    df = create_features(df_flights_with_regions)
    if RawFeatures._DELAY_TARGET not in df_flights_with_regions.columns:
        raise ValueError(f"No target '{RawFeatures._DELAY_TARGET}' column found in dataset.")
    features = SELECTED_BEST_MODEL_FEATURES
    df = df.dropna(subset=features + [RawFeatures._DELAY_TARGET])

    X = df[features]
    y = df[RawFeatures._DELAY_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"starting training run for model: {model_type}")
    with mlflow.start_run():
        pipeline = _get_model_pipeline(model_type)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("features", features)
        pipeline.fit(X_train, y_train)
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        evaluate_and_log_metrics(y_train, y_pred_train, prefix="train_")
        evaluate_and_log_metrics(y_test, y_pred_test, prefix="test_")
        mlflow.sklearn.log_model(pipeline, artifact_path="model")


if __name__ == "__main__":
    import argparse

    base_dir = Path(__file__).parent
    flights_data_path = Path("dataset/flights.csv")
    airports_data_path = Path("dataset/airports.csv")
    config_path = Path("config/mlflow_config.yaml")
    model_type = "logistic_regression"

    mlflow_dir = Path("mlruns")
    mlflow_dir.mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="train flight delays models")
    parser.add_argument(
        "--flights-data-path", type=Path, default=flights_data_path, help="Path to flight data CSV"
    )
    parser.add_argument("--airports-data-path", type=Path, default=airports_data_path, help="Path to airports data CSV")
    parser.add_argument("--config-path", type=Path, default=config_path, help="Path to MLflow config")
    parser.add_argument(
        "--model-type",
        type=str,
        default=model_type,
        help="Type of model to train (logistic_regression, random_forest, lightgbm)",
    )

    args = parser.parse_args()

    train_model(
        flights_data_path=args.flights_data_path,
        airports_data_path=args.airports_data_path,
        config_path=args.config_path,
        model_type=args.model_type,
    )
