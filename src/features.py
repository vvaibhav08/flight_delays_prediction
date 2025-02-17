import pandas as pd

from src.constants import AggregatedFeatures, RawFeatures


def create_features(
    df: pd.DataFrame, lags_months: list = [1, 2], rolling_windows_months: list = [2, 4]
) -> pd.DataFrame:
    """
    Creates time-based features (lag, rolling average) for monthly transaction data.
    Assumes df has columns: client_nr, yearmonth, volume_debit_trx, volume_credit_trx,
                            nr_debit_trx, nr_credit_trx, total_nr_trx, min_balance, max_balance.

    1. Combined columns are created:
       - net_volume = volume_credit_trx - volume_debit_trx
       - debit_total_trx_ratio = nr_debit_trx / total_nr_trx
       - debit_total_vol_ratio = volume_debit_trx / (volume_debit_trx + volume_credit_trx)
       - balance_range = max_balance - min_balance

    2. Sort by (client_nr, yearmonth) to ensure chronological order.

    3. For each client_nr group:
       - For each column in [net_volume, min_balance, debit_total_trx_ratio, debit_total_vol_ratio]:
         * Create lag features for each value in 'lags' (e.g. col_lag1, col_lag2).
         * Create rolling average features for each value in 'rolling_windows' (e.g. col_rolling3, col_rolling6).
           Uses a rolling window on each group's time series.
    4. Return df with newly added columns.

    Input
        df: Input DataFrame with necessary columns.
        lags_months: List of integers for lag offsets (e.g. [1,2]).
        rolling_windows_months: List of integers for rolling window sizes (e.g. [2,4]).
    Returns
        DataFrame with newly created lag & rolling average features.
    """

    df = df.copy()

    df[AggregatedFeatures.MONTHLY_NET_VOLUME] = df[RawFeatures.VOLUME_CREDIT_TRX] - df[RawFeatures.VOLUME_DEBIT_TRX]
    df[AggregatedFeatures.MONTHLY_DEBIT_TOTAL_TRX_RATIO] = df[RawFeatures.NR_DEBIT_TRX] / df[RawFeatures.TOTAL_NR_TRX]
    df[AggregatedFeatures.MONTHLY_DEBIT_TOTAL_VOL_RATIO] = df[RawFeatures.VOLUME_DEBIT_TRX] / (
        df[RawFeatures.VOLUME_DEBIT_TRX] + df[RawFeatures.VOLUME_CREDIT_TRX]
    )
    df[AggregatedFeatures.MONTHLY_BALANCE_RANGE] = df[RawFeatures.MAX_BALANCE] - df[RawFeatures.MIN_BALANCE]

    df = df.sort_values([RawFeatures.CLIENT_NR, RawFeatures.YEARMONTH]).reset_index(drop=True)

    # Rolling average features
    time_series_cols = [
        RawFeatures.MIN_BALANCE,
        AggregatedFeatures.MONTHLY_NET_VOLUME,
        AggregatedFeatures.MONTHLY_BALANCE_RANGE,
        AggregatedFeatures.MONTHLY_DEBIT_TOTAL_TRX_RATIO,
        AggregatedFeatures.MONTHLY_DEBIT_TOTAL_VOL_RATIO,
    ]

    # Group by client_nr for time-based transformations
    def _create_features_for_group(g: pd.DataFrame) -> pd.DataFrame:
        g["cumulative_applications"] = (
            g["credit_application"].shift(1).cumsum()
        )  # cumulative applications up to the previous month
        g["cumulative_nr_applications"] = (
            g["nr_credit_applications"].shift(1).cumsum()
        )  # cumulative number of applications up to the previous month
        for col in time_series_cols:
            # LAGS
            for lag_val in lags_months:
                g[f"{col}_lag{lag_val}"] = g[col].shift(lag_val)
            # ROLLING AVERAGES
            for w in rolling_windows_months:
                g[f"{col}_rolling{w}"] = g[col].shift(1).rolling(window=w, min_periods=1).mean()
        return g

    # Apply this function to each client
    return df.groupby(RawFeatures.CLIENT_NR, group_keys=False).apply(_create_features_for_group)
