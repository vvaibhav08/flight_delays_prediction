from enum import StrEnum


class RawFeatures(StrEnum):
    ACTUAL_OFF_BLOCK_TIME = "actualOffBlockTime"
    AIRCRAFT_REGISTRATION = "aircraftRegistration"
    AIRCRAFT_TYPE_IATAMAIN = "aircraftType.iatamain"
    AIRCRAFT_TYPE_IATASUB = "aircraftType.iatasub"
    AIRLINE_CODE = "airlineCode"
    BAGGAGE_CLAIM = "baggageClaim"
    ESTIMATED_LANDING_TIME = "estimatedLandingTime"
    EXPECTED_TIME_BOARDING = "expectedTimeBoarding"
    EXPECTED_TIME_GATE_CLOSING = "expectedTimeGateClosing"
    EXPECTED_TIME_GATE_OPEN = "expectedTimeGateOpen"
    EXPECTED_TIME_ON_BELT = "expectedTimeOnBelt"
    FLIGHT_DIRECTION = "flightDirection"
    FLIGHT_NAME = "flightName"
    FLIGHT_NUMBER = "flightNumber"
    GATE = "gate"
    ID = "id"
    MAIN_FLIGHT = "mainFlight"
    PREFIX_IATA = "prefixIATA"
    PREFIX_ICAO = "prefixICAO"
    PUBLIC_ESTIMATED_OFF_BLOCK_TIME = "publicEstimatedOffBlockTime"
    PUBLIC_FLIGHT_STATE = "publicFlightState.flightStates"
    ROUTE_DESTINATIONS = "route.destinations"
    SCHEDULE_DATE = "scheduleDate"
    SCHEDULE_TIME = "scheduleTime"
    SERVICE_TYPE = "serviceType"
    TERMINAL = "terminal"
    TRANSFER_POSITIONS = "transferPositions"
    TRANSFER_POSITIONS_TRANSFER = "transferPositions.transferPositions"
    # Constructed / derived features:
    DEPARTURE_REGION = "region"
    DELAY_MINUTES = "delay_minutes"
    SCHEDULED_DEPARTURE = "scheduledDeparture"
    SCHEDULED_HOUR = "hour"
    SCHEDULED_DAY_OF_WEEK = "day_of_week"
    _DELAY_TARGET = "delay_target"


class AggregatedFeatures(StrEnum):
    AGG_DAILY_AVG_DELAY_OVERALL = "daily_2h_prior_avg_delay_overall"
    AGG_DAILY_COUNT_OVERALL = "daily_2h_prior_count_overall"
    AGG_DAILY_AVG_DELAY_TERMINAL = "daily_2h_prior_avg_delay_terminal"
    AGG_DAILY_COUNT_TERMINAL = "daily_2h_prior_count_terminal"
    AGG_DAILY_AVG_DELAY_REGION = "daily_2h_prior_avg_delay_region"
    AGG_DAILY_COUNT_REGION = "daily_2h_prior_count_region"
    AGG_LAG_DELAY_2H_OVERALL = "lag_avg_delay_2h_overall"
    AGG_LAG_DELAY_12H_OVERALL = "lag_avg_delay_12h_overall"
    AGG_LAG_DELAY_1D_OVERALL = "lag_avg_delay_1d_overall"


# These features were selected in the model exploration notebook.
SELECTED_BEST_MODEL_FEATURES = [
    RawFeatures.TERMINAL,
    RawFeatures.SERVICE_TYPE,
    RawFeatures.DEPARTURE_REGION,
    RawFeatures.SCHEDULED_HOUR,
    RawFeatures.SCHEDULED_DAY_OF_WEEK,
    AggregatedFeatures.AGG_LAG_DELAY_2H_OVERALL,
    AggregatedFeatures.AGG_LAG_DELAY_12H_OVERALL,
    AggregatedFeatures.AGG_LAG_DELAY_1D_OVERALL,
    AggregatedFeatures.AGG_DAILY_AVG_DELAY_OVERALL,
    AggregatedFeatures.AGG_DAILY_COUNT_OVERALL,
    AggregatedFeatures.AGG_DAILY_AVG_DELAY_TERMINAL,
    AggregatedFeatures.AGG_DAILY_COUNT_TERMINAL,
    AggregatedFeatures.AGG_DAILY_AVG_DELAY_REGION,
    AggregatedFeatures.AGG_DAILY_COUNT_REGION,
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
