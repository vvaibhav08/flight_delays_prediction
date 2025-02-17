from enum import StrEnum


class RawFeatures(StrEnum):
    TOTAL_NR_TRX = "total_nr_trx"
    NR_DEBIT_TRX = "nr_debit_trx"
    VOLUME_DEBIT_TRX = "volume_debit_trx"
    NR_CREDIT_TRX = "nr_credit_trx"
    VOLUME_CREDIT_TRX = "volume_credit_trx"
    MIN_BALANCE = "min_balance"
    MAX_BALANCE = "max_balance"
    CRG = "CRG"
    CLIENT_NR = "client_nr"
    YEARMONTH = "yearmonth"
    _CREDIT_APPLICATION = "credit_application"  # naming differently cause it be the target
    NR_CREDIT_APPLICATIONS = "nr_credit_applications"


class AggregatedFeatures(StrEnum):
    MONTHLY_NET_VOLUME = "net_volume"
    MONTHLY_DEBIT_TOTAL_TRX_RATIO = "debit_total_trx_ratio"
    MONTHLY_DEBIT_TOTAL_VOL_RATIO = "debit_total_vol_ratio"
    MONTHLY_BALANCE_RANGE = "balance_range"

    # lag features
    MONTHLY_BALANCE_RANGE_LAG1 = "balance_range_lag1"
    MONTHLY_BALANCE_RANGE_LAG2 = "balance_range_lag2"
    MONTHLY_NET_VOLUME_LAG1 = "net_volume_lag1"
    MONTHLY_NET_VOLUME_LAG2 = "net_volume_lag2"
    MONTHLY_MIN_BALANCE_LAG1 = "min_balance_lag1"
    MONTHLY_MIN_BALANCE_LAG2 = "min_balance_lag2"
    MONTHLY_DEBIT_TOTAL_TRX_RATIO_LAG1 = "debit_total_trx_ratio_lag1"
    MONTHLY_DEBIT_TOTAL_TRX_RATIO_LAG2 = "debit_total_trx_ratio_lag2"
    MONTHLY_DEBIT_TOTAL_VOL_RATIO_LAG1 = "debit_total_vol_ratio_lag1"
    MONTHLY_DEBIT_TOTAL_VOL_RATIO_LAG2 = "debit_total_vol_ratio_lag2"

    # rolling average features
    MONTHLY_NET_VOLUME_ROLLING2 = "net_volume_rolling2"
    MONTHLY_NET_VOLUME_ROLLING4 = "net_volume_rolling4"
    MONTHLY_BALANCE_RANGE_ROLLING2 = "balance_range_rolling2"
    MONTHLY_BALANCE_RANGE_ROLLING4 = "balance_range_rolling4"
    MONTHLY_MIN_BALANCE_ROLLING2 = "min_balance_rolling2"
    MONTHLY_MIN_BALANCE_ROLLING4 = "min_balance_rolling4"
    MONTHLY_DEBIT_TOTAL_TRX_RATIO_ROLLING2 = "debit_total_trx_ratio_rolling2"
    MONTHLY_DEBIT_TOTAL_TRX_RATIO_ROLLING4 = "debit_total_trx_ratio_rolling4"
    MONTHLY_DEBIT_TOTAL_VOL_RATIO_ROLLING2 = "debit_total_vol_ratio_rolling2"
    MONTHLY_DEBIT_TOTAL_VOL_RATIO_ROLLING4 = "debit_total_vol_ratio_rolling4"

    # cumulative features
    CUMULATIVE_FREQUENCY_APPLICATIONS_TO_MONTH = "cumulative_applications"
    CUMULATIVE_NR_APPLICATIONS_TO_MONTH = "cumulative_nr_applications"


# this is the result of the model exploration done in the notebook (notebooks/model-exploration.ipynb)
SELECTED_BEST_MODEL_FEATURES = [
    AggregatedFeatures.MONTHLY_NET_VOLUME_LAG1,
    AggregatedFeatures.MONTHLY_DEBIT_TOTAL_TRX_RATIO_LAG1,
    AggregatedFeatures.MONTHLY_DEBIT_TOTAL_VOL_RATIO_LAG1,
    AggregatedFeatures.MONTHLY_MIN_BALANCE_LAG1,
    AggregatedFeatures.MONTHLY_BALANCE_RANGE_LAG1,
    AggregatedFeatures.CUMULATIVE_FREQUENCY_APPLICATIONS_TO_MONTH,
    AggregatedFeatures.CUMULATIVE_NR_APPLICATIONS_TO_MONTH,
    RawFeatures.CRG,
]

# this is the result of the model exploration done in the notebook (notebooks/model-exploration.ipynb)
CLASSIFICATION_THRESHOLD = 0.4

# Best parameters for Logistic Regression
LOGISTIC_REGRESSION_PARAMS = {"C": 0.01, "class_weight": "balanced", "max_iter": 10000}

# Best parameters for Random Forest
RANDOM_FOREST_PARAMS = {
    "n_estimators": 50,
    "max_depth": 5,
    "random_state": 42,
    "class_weight": "balanced",
}

# Best parameters for LightGBM
LIGHTGBM_PARAMS = {
    "learning_rate": 0.005,
    "n_estimators": 200,
    "random_state": 42,
    "scale_pos_weight": 11.3,
    "bagging_freq": 2,
    "neg_bagging_fraction": 0.1,
    "pos_bagging_fraction": 0.5,
    "verbosity": -1,
}

MODEL_PIPELINE_STEP_ARGUMENTS = {
    "imputer__strategy": "mean",
    "outlier_clip__factor": 7.0,
    "scaler__with_mean": True,
    "scaler__with_std": True,
    "var_thresh__threshold": 0.0,
}
