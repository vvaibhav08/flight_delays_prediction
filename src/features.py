import pandas as pd
import numpy as np

from src.constants import AggregatedFeatures, RawFeatures


def create_features(
    df: pd.DataFrame, lags_hours: list = [2, 4, 8]
) -> pd.DataFrame:
    """
    Creates flight delay features based on the EDA steps.

    Features created:

    1. Temporal Features:
       - Hour of scheduled departure (RawFeatures.HOUR)
       - Day of week (RawFeatures.DAY_OF_WEEK)
       - Schedule date (extracted from RawFeatures.SCHEDULED_DEPARTURE)

    2. Lag Features:
       - Overall lag features for each lag value in lags_hours:
         For each flight, computes the average delay (from RawFeatures.DELAY_MINUTES)
         over a 1‐hour window from (lag+1) to lag hours before scheduled departure,
         restricted to flights on the same day.
       - Terminal-specific lag (for a 2-hour lag) computed similarly using RawFeatures.TERMINAL.

    3. Aggregated (Cumulative) Features:
       - Overall aggregated features: For flights on the same day (RawFeatures.SCHEDULE_DATE),
         calculate the cumulative average delay and flight count from the beginning of the day
         (i.e. midnight) until 2 hours before the flight’s scheduled departure.
         Results are stored using the keys from AggregatedFeatures.
       - Analogous aggregated features are computed per terminal and, if available, per region.

    Input:
      df: DataFrame containing flight data with at least the following columns:
          • RawFeatures.SCHEDULED_DEPARTURE (datetime)
          • RawFeatures.DELAY_MINUTES (numeric)
          • RawFeatures.TERMINAL
          • Optionally, a region column (RawFeatures.REGION or "region")
      lags_hours: List of lag hours for computing lag features (default: [2, 4, 8])

    Returns:
      DataFrame with newly created features.
    """
    df = df.copy()
    df = df.sort_values(RawFeatures.SCHEDULED_DEPARTURE).reset_index(drop=True)

    # Ensure scheduledDeparture is datetime
    if not np.issubdtype(df[RawFeatures.SCHEDULED_DEPARTURE].dtype, np.datetime64):
        df[RawFeatures.SCHEDULED_DEPARTURE] = pd.to_datetime(df[RawFeatures.SCHEDULED_DEPARTURE])

    # Create temporal features.
    df[RawFeatures.HOUR] = df[RawFeatures.SCHEDULED_DEPARTURE].dt.hour
    df[RawFeatures.DAY_OF_WEEK] = df[RawFeatures.SCHEDULED_DEPARTURE].dt.dayofweek
    # Create scheduleDate from scheduledDeparture
    df[RawFeatures.SCHEDULE_DATE] = df[RawFeatures.SCHEDULED_DEPARTURE].dt.date

    # Overall lag features: average delay in a 1-hour window from (lag+1) to lag hours before departure.
    def avg_delay_window_overall(current_time, lag_hours):
        window_start = current_time - pd.Timedelta(hours=lag_hours + 1)
        window_end = current_time - pd.Timedelta(hours=lag_hours)
        mask = (
            (df[RawFeatures.SCHEDULED_DEPARTURE].dt.date == current_time.date()) &
            (df[RawFeatures.SCHEDULED_DEPARTURE] >= window_start) &
            (df[RawFeatures.SCHEDULED_DEPARTURE] < window_end)
        )
        return df.loc[mask, RawFeatures.DELAY_MINUTES].mean()

    for lag in lags_hours:
        col_name = f"lag_avg_delay_{lag}h_overall"
        df[col_name] = df[RawFeatures.SCHEDULED_DEPARTURE].apply(lambda t: avg_delay_window_overall(t, lag))

    # Terminal-specific lag feature for 2-hour window (from 3h to 2h before departure)
    def avg_delay_window_terminal(row):
        current_time = row[RawFeatures.SCHEDULED_DEPARTURE]
        terminal = row[RawFeatures.TERMINAL]
        window_start = current_time - pd.Timedelta(hours=3)
        window_end = current_time - pd.Timedelta(hours=2)
        mask = (
            (df[RawFeatures.SCHEDULED_DEPARTURE].dt.date == current_time.date()) &
            (df[RawFeatures.TERMINAL] == terminal) &
            (df[RawFeatures.SCHEDULED_DEPARTURE] >= window_start) &
            (df[RawFeatures.SCHEDULED_DEPARTURE] < window_end)
        )
        return df.loc[mask, RawFeatures.DELAY_MINUTES].mean()

    df["lag_avg_delay_2h_terminal"] = df.apply(avg_delay_window_terminal, axis=1)

    # Aggregated features: computed from start of day until 2 hours before scheduled departure.
    def compute_cumulative_avg(group):
        group = group.sort_values(RawFeatures.SCHEDULED_DEPARTURE)
        avg_delays = []
        for _, row in group.iterrows():
            start_of_day = row[RawFeatures.SCHEDULED_DEPARTURE].normalize()
            cutoff = row[RawFeatures.SCHEDULED_DEPARTURE] - pd.Timedelta(hours=2)
            earlier = group[
                (group[RawFeatures.SCHEDULED_DEPARTURE] >= start_of_day) &
                (group[RawFeatures.SCHEDULED_DEPARTURE] < cutoff)
            ]
            if earlier.empty:
                avg_delay = np.nan
            else:
                avg_delay = np.nanmean(earlier[RawFeatures.DELAY_MINUTES])
            avg_delays.append(avg_delay)
        return pd.Series(avg_delays, index=group.index)

    def compute_cumulative_count(group):
        group = group.sort_values(RawFeatures.SCHEDULED_DEPARTURE)
        counts = []
        for _, row in group.iterrows():
            start_of_day = row[RawFeatures.SCHEDULED_DEPARTURE].normalize()
            cutoff = row[RawFeatures.SCHEDULED_DEPARTURE] - pd.Timedelta(hours=2)
            earlier = group[
                (group[RawFeatures.SCHEDULED_DEPARTURE] >= start_of_day) &
                (group[RawFeatures.SCHEDULED_DEPARTURE] < cutoff)
            ]
            counts.append(len(earlier))
        return pd.Series(counts, index=group.index)

    # Overall aggregated features
    df[AggregatedFeatures.AGG_AVG_DELAY_OVERALL] = df.groupby(RawFeatures.SCHEDULE_DATE, group_keys=False).apply(compute_cumulative_avg)
    df[AggregatedFeatures.AGG_COUNT_OVERALL] = df.groupby(RawFeatures.SCHEDULE_DATE, group_keys=False).apply(compute_cumulative_count)

    # Terminal aggregated features
    df[AggregatedFeatures.AGG_AVG_DELAY_TERMINAL] = df.groupby([RawFeatures.SCHEDULE_DATE, RawFeatures.TERMINAL], group_keys=False).apply(compute_cumulative_avg)
    df[AggregatedFeatures.AGG_COUNT_TERMINAL] = df.groupby([RawFeatures.SCHEDULE_DATE, RawFeatures.TERMINAL], group_keys=False).apply(compute_cumulative_count)

    # Region aggregated features, if available.
    region_col = RawFeatures.REGION if RawFeatures.REGION in df.columns else "region"
    if region_col in df.columns:
        df[AggregatedFeatures.AGG_AVG_DELAY_REGION] = df.groupby([RawFeatures.SCHEDULE_DATE, region_col], group_keys=False).apply(compute_cumulative_avg)
        df[AggregatedFeatures.AGG_COUNT_REGION] = df.groupby([RawFeatures.SCHEDULE_DATE, region_col], group_keys=False).apply(compute_cumulative_count)

    return df