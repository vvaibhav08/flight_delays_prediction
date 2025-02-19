from enum import StrEnum


class RawFeatures(StrEnum):
    AIRLINE_CODE = "airlineCode"
    ESTIMATED_LANDING_TIME = "estimatedLandingTime"
    EXPECTED_TIME_BOARDING = "expectedTimeBoarding"
    EXPECTED_TIME_GATE_CLOSING = "expectedTimeGateClosing"
    EXPECTED_TIME_GATE_OPEN = "expectedTimeGateOpen"
    EXPECTED_TIME_ON_BELT = "expectedTimeOnBelt"
    FLIGHT_DIRECTION = "flightDirection"
    FLIGHT_NAME = "flightName"
    FLIGHT_NUMBER = "flightNumber"
    GATE = "gate"
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
    SCHEDULED_DEPARTURE = "scheduledDeparture"
    DELAY_MINUTES = "delay_minutes"
    DELAY_MINUTES_LOG = "delay_minutes_log"
    NUM_DESTINATIONS = "num_destinations"
    FIRST_DESTINATION = "first_destination"
    AIRPORT = "Airport"
    NAME = "Name"
    CITY = "City"
    COUNTRY = "Country"
    LATITUDE = "Latitude"
    LONGITUDE = "Longitude"
    ALTITUDE = "Altitude"
    TIMEZONE = "Timezone"
    DST = "DST"
    TZ = "Tz"
    TYPE = "Type"
    SOURCE = "Source"
    REGION = "region"
    LOG_DELAY = "log_delay"
    HOUR = "hour"
    DAY_OF_WEEK = "day_of_week"
    DELAY_PREVIOUS_HOUR = "delay_previous_hour"
    DELAY_PREVIOUS_DAY = "delay_previous_day"
    _DELAY_TARGET = "delay_target"


class AggregatedFeatures(StrEnum):
    AGG_AVG_DELAY_OVERALL = "agg_avg_delay_overall"
    AGG_COUNT_OVERALL = "agg_count_overall"
    AGG_AVG_DELAY_TERMINAL = "agg_avg_delay_terminal"
    AGG_COUNT_TERMINAL = "agg_count_terminal"
    AGG_AVG_DELAY_REGION = "agg_avg_delay_region"
    AGG_COUNT_REGION = "agg_count_region"


# this is the result of the model exploration done in the notebook (notebooks/model-exploration.ipynb)
SELECTED_BEST_MODEL_FEATURES = [
    AggregatedFeatures.AGG_AVG_DELAY_OVERALL,
    AggregatedFeatures.AGG_COUNT_OVERALL,
    AggregatedFeatures.AGG_AVG_DELAY_TERMINAL,
    AggregatedFeatures.AGG_COUNT_TERMINAL,
    AggregatedFeatures.AGG_AVG_DELAY_REGION,
    AggregatedFeatures.AGG_COUNT_REGION,
    RawFeatures.DELAY_PREVIOUS_DAY,
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
