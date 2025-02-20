from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.constants import RawFeatures


def load_csv_data(dataset_path: Path, sep: str = ",") -> pd.DataFrame:
    return pd.read_csv(dataset_path, sep=sep)


def merge_data(df_flights: pd.DataFrame, df_airports: pd.DataFrame) -> pd.DataFrame:
    """
        Merge the processed flight data with the airport data using a left join,
       matching df_flights['first_destination'] to df_airports['IATA']. Drop rows with missing IATA codes.
        Extract the top-level region from the 'Tz' column in the merged DataFrame.
       For example, "Europe/Paris" will be split to "Europe".

    Returns:
        Merged DataFrame with additional columns: 'num_destinations', 'first_destination' and 'region'.
    """
    df_flights["num_destinations"] = df_flights[RawFeatures.ROUTE_DESTINATIONS].apply(
        lambda x: len(eval(x)) if isinstance(x, str) else 0
    )
    df_flights["first_destination"] = df_flights[RawFeatures.ROUTE_DESTINATIONS].apply(
        lambda x: eval(x)[0] if isinstance(x, str) and len(eval(x)) > 0 else None
    )

    # Merge flights with airports on the first destination and IATA, dropping rows with missing IATA.
    df_merged = pd.merge(
        df_flights, df_airports, how="left", left_on="first_destination", right_on="IATA", suffixes=("", "_airport")
    ).dropna(subset=["IATA"])

    # Extract top-level region from the Tz column (e.g., "Europe/Paris" -> "Europe")
    df_merged["region"] = df_merged["Tz"].str.split("/").str.get(0)

    return df_merged


class IQRClipper(BaseEstimator, TransformerMixin):
    """
    Clips numeric features to [Q1 - factor*IQR, Q3 + factor*IQR].
    This is done column by column, calculated on the training data.
    By default, factor=3.0 means fairly lenient outlier clipping.
    """

    def __init__(self, factor: float = 3.0, quartile_range: list[int] = [25, 75]):
        self.factor = factor
        self.quartile_range = quartile_range
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        self.quartile1_ = self.quartile_range[0]
        self.quartile2_ = self.quartile_range[1]

    def fit(self, X, y=None):
        # We compute the Q1, Q3 for each numeric column, and store them
        # as well as the lower/upper clipping bounds
        X_ = pd.DataFrame(X).copy()
        for col in X_.select_dtypes(include=[np.number]).columns:
            q1 = np.percentile(X_[col], self.quartile1_)
            q3 = np.percentile(X_[col], self.quartile2_)
            iqr = q3 - q1
            self.lower_bounds_[col] = q1 - self.factor * iqr
            self.upper_bounds_[col] = q3 + self.factor * iqr
        return self

    def transform(self, X, y=None):
        X_ = pd.DataFrame(X).copy()
        numeric_cols = X_.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            lb = self.lower_bounds_.get(col, None)
            ub = self.upper_bounds_.get(col, None)
            if lb is not None and ub is not None:
                X_[col] = np.clip(X_[col], lb, ub)
        return X_.values
