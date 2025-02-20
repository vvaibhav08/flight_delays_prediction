import numpy as np
import pandas as pd

from src.constants import AggregatedFeatures, RawFeatures


def create_features(df: pd.DataFrame, lags: list = [3, 12, 1]) -> pd.DataFrame:
    """
    Creates flight delay features based on the EDA steps.

    Features created:

    1. Temporal Features:
       - Hour of scheduled departure, day of week.
       - A combined datetime "scheduledDeparture" from RawFeatures.SCHEDULE_DATE and RawFeatures.SCHEDULE_TIME.

    2. Lag Features (Overall):
       For each flight, using 60-minute bins, compute a reference average delay per bin.
       Then for each flight, compute target times for:
         - 2h lag: using scheduledDeparture - 3 hours (per your example)
         - 12h lag: using scheduledDeparture - 12 hours
         - 1d lag: using scheduledDeparture - 1 day
       Round these target times to the nearest 60 minutes, then map the corresponding bin’s average delay.
       Results are stored using AggregatedFeatures.AGG_LAG_DELAY_2H_OVERALL,
       AggregatedFeatures.AGG_LAG_DELAY_12H_OVERALL, and AggregatedFeatures.AGG_LAG_DELAY_1D_OVERALL.

    3. Aggregated (Cumulative) Features (Daily):
       For flights on the same day (using RawFeatures.SCHEDULE_DATE),
       calculate the cumulative average delay and flight count (using only flights that departed
       at least 2 hours before the current flight’s scheduledDeparture).
       Analogous features are computed per terminal (RawFeatures.TERMINAL)
       and, if available, per region (RawFeatures.DEPARTURE_REGION).
       Results are stored using keys from AggregatedFeatures.

    4. Conversion of Categorical Features and Target:
       Converts RawFeatures.SERVICE_TYPE and RawFeatures.DEPARTURE_REGION to numeric codes.
       Creates a binary target (RawFeatures._DELAY_TARGET): 0 if delay_minutes ≤ 10, else 1.

    Input:
      df: DataFrame containing flight data.

    Returns:
      DataFrame with newly created features.
    """
    df = df.copy()

    df[RawFeatures.SCHEDULED_DEPARTURE] = pd.to_datetime(
        df[RawFeatures.SCHEDULE_DATE].astype(str) + " " + df[RawFeatures.SCHEDULE_TIME].astype(str)
    )
    df[RawFeatures.SCHEDULED_DEPARTURE] = df[RawFeatures.SCHEDULED_DEPARTURE].dt.tz_localize("UTC+01:00")
    df[RawFeatures.ACTUAL_OFF_BLOCK_TIME] = pd.to_datetime(df[RawFeatures.ACTUAL_OFF_BLOCK_TIME], errors="coerce")
    df[RawFeatures.SCHEDULED_DEPARTURE] = pd.to_datetime(
        df[RawFeatures.SCHEDULE_DATE].astype(str) + " " + df[RawFeatures.SCHEDULE_TIME].astype(str)
    )
    df[RawFeatures.SCHEDULED_DEPARTURE] = df[RawFeatures.SCHEDULED_DEPARTURE].dt.tz_localize("UTC+01:00")
    df[RawFeatures.ACTUAL_OFF_BLOCK_TIME] = pd.to_datetime(df[RawFeatures.ACTUAL_OFF_BLOCK_TIME], errors="coerce")
    df[RawFeatures.DELAY_MINUTES] = (
        df[RawFeatures.ACTUAL_OFF_BLOCK_TIME] - df[RawFeatures.SCHEDULED_DEPARTURE]
    ).dt.total_seconds() / 60

    raw_data_len = len(df)
    print(f"Total flights before dropping NA delays: {raw_data_len}")
    df.dropna(subset=[RawFeatures.DELAY_MINUTES], inplace=True)
    print(f"Total flights after dropping NA delays: {raw_data_len - len(df)}")
    print(f"Percentage of entries dropped: {(1 - len(df) / raw_data_len) * 100:.2f}%")

    df[RawFeatures.DELAY_MINUTES] = df[RawFeatures.DELAY_MINUTES].apply(lambda x: x if x > 0 else 0)

    # Create "scheduledDeparture" if not already present.
    if RawFeatures.SCHEDULED_DEPARTURE not in df.columns:
        df[RawFeatures.SCHEDULED_DEPARTURE] = pd.to_datetime(
            df[RawFeatures.SCHEDULE_DATE].astype(str) + " " + df[RawFeatures.SCHEDULE_TIME].astype(str)
        )

    df = df.sort_values(RawFeatures.SCHEDULED_DEPARTURE).reset_index(drop=True)

    # Create temporal features.
    df[RawFeatures.SCHEDULED_HOUR] = df[RawFeatures.SCHEDULED_DEPARTURE].dt.hour
    df[RawFeatures.SCHEDULED_DAY_OF_WEEK] = df[RawFeatures.SCHEDULED_DEPARTURE].dt.dayofweek
    df[RawFeatures.SCHEDULE_DATE] = df[RawFeatures.SCHEDULED_DEPARTURE].dt.date  # overwrite with date only

    # Create a bin by rounding scheduledDeparture to the nearest 60 minutes.
    df["departure_bin"] = df[RawFeatures.SCHEDULED_DEPARTURE].dt.round("60min")

    bin_avg = df.groupby("departure_bin")[RawFeatures.DELAY_MINUTES].mean()

    df["target_time_2h"] = df[RawFeatures.SCHEDULED_DEPARTURE] - pd.Timedelta(hours=lags[0])
    df["target_time_12h"] = df[RawFeatures.SCHEDULED_DEPARTURE] - pd.Timedelta(hours=lags[1])
    df["target_time_1d"] = df[RawFeatures.SCHEDULED_DEPARTURE] - pd.Timedelta(days=lags[2])

    df["target_bin_2h"] = df["target_time_2h"].dt.round("60min")
    df["target_bin_12h"] = df["target_time_12h"].dt.round("60min")
    df["target_bin_1d"] = df["target_time_1d"].dt.round("60min")

    df[AggregatedFeatures.AGG_LAG_DELAY_2H_OVERALL] = df["target_bin_2h"].map(bin_avg)
    df[AggregatedFeatures.AGG_LAG_DELAY_12H_OVERALL] = df["target_bin_12h"].map(bin_avg)
    df[AggregatedFeatures.AGG_LAG_DELAY_1D_OVERALL] = df["target_bin_1d"].map(bin_avg)

    cols_to_drop = [
        "departure_bin",
        "target_time_2h",
        "target_time_12h",
        "target_time_1d",
        "target_bin_2h",
        "target_bin_12h",
        "target_bin_1d",
    ]
    df.drop(columns=cols_to_drop, inplace=True)

    print(f"Calculating temporal lag and aggregate features. This takes a few minutes...")

    # These functions compute the cumulative (aggregated) average delay and count
    # for flights on the same day up to (scheduledDeparture - 2 hours).
    def compute_cumulative_avg(group):
        group = group.sort_values(RawFeatures.SCHEDULED_DEPARTURE)
        avg_delays = []
        for _, row in group.iterrows():
            start_of_day = row[RawFeatures.SCHEDULED_DEPARTURE].normalize()
            cutoff = row[RawFeatures.SCHEDULED_DEPARTURE] - pd.Timedelta(hours=2)
            earlier = group[
                (group[RawFeatures.SCHEDULED_DEPARTURE] >= start_of_day)
                & (group[RawFeatures.SCHEDULED_DEPARTURE] < cutoff)
            ]
            avg_delay = np.nanmean(earlier[RawFeatures.DELAY_MINUTES]) if not earlier.empty else np.nan
            avg_delays.append(avg_delay)
        return pd.Series(avg_delays, index=group.index)

    def compute_cumulative_count(group):
        group = group.sort_values(RawFeatures.SCHEDULED_DEPARTURE)
        counts = []
        for _, row in group.iterrows():
            start_of_day = row[RawFeatures.SCHEDULED_DEPARTURE].normalize()
            cutoff = row[RawFeatures.SCHEDULED_DEPARTURE] - pd.Timedelta(hours=2)
            earlier = group[
                (group[RawFeatures.SCHEDULED_DEPARTURE] >= start_of_day)
                & (group[RawFeatures.SCHEDULED_DEPARTURE] < cutoff)
            ]
            counts.append(len(earlier))
        return pd.Series(counts, index=group.index)

    df[AggregatedFeatures.AGG_DAILY_AVG_DELAY_OVERALL] = df.groupby(RawFeatures.SCHEDULE_DATE, group_keys=False).apply(
        compute_cumulative_avg
    )
    df[AggregatedFeatures.AGG_DAILY_COUNT_OVERALL] = df.groupby(RawFeatures.SCHEDULE_DATE, group_keys=False).apply(
        compute_cumulative_count
    )

    df[AggregatedFeatures.AGG_DAILY_AVG_DELAY_TERMINAL] = df.groupby(
        [RawFeatures.SCHEDULE_DATE, RawFeatures.TERMINAL], group_keys=False
    ).apply(compute_cumulative_avg)
    df[AggregatedFeatures.AGG_DAILY_COUNT_TERMINAL] = df.groupby(
        [RawFeatures.SCHEDULE_DATE, RawFeatures.TERMINAL], group_keys=False
    ).apply(compute_cumulative_count)

    if RawFeatures.DEPARTURE_REGION in df.columns:
        df[AggregatedFeatures.AGG_DAILY_AVG_DELAY_REGION] = df.groupby(
            [RawFeatures.SCHEDULE_DATE, RawFeatures.DEPARTURE_REGION], group_keys=False
        ).apply(compute_cumulative_avg)
        df[AggregatedFeatures.AGG_DAILY_COUNT_REGION] = df.groupby(
            [RawFeatures.SCHEDULE_DATE, RawFeatures.DEPARTURE_REGION], group_keys=False
        ).apply(compute_cumulative_count)

    df[RawFeatures.SERVICE_TYPE] = pd.Categorical(df[RawFeatures.SERVICE_TYPE]).codes
    if RawFeatures.DEPARTURE_REGION in df.columns:
        df[RawFeatures.DEPARTURE_REGION] = pd.Categorical(df[RawFeatures.DEPARTURE_REGION]).codes

    # Create the binary target: 0 if delay_minutes <= 10, else 1.
    df[RawFeatures._DELAY_TARGET] = df[RawFeatures.DELAY_MINUTES].apply(lambda x: 0 if x <= 10 else 1)

    return df
