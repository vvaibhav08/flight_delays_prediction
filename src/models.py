from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import (
    LIGHTGBM_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
    MODEL_PIPELINE_STEP_ARGUMENTS,
    RANDOM_FOREST_PARAMS,
)
from src.processing import IQRClipper


class BaseModel:
    def __init__(
        self, model_arguments: dict | None = None, pipeline_step_arguments: dict = MODEL_PIPELINE_STEP_ARGUMENTS
    ):
        self.pipeline_step_arguments = pipeline_step_arguments
        self.model_arguments = model_arguments

    def create_pipeline(self, model_step):
        steps = [
            ("imputer", SimpleImputer(strategy=self.pipeline_step_arguments.get("imputer__strategy", "mean"))),
            ("outlier_clip", IQRClipper(factor=self.pipeline_step_arguments.get("outlier_clip__factor", 5.0))),
            (
                "scaler",
                StandardScaler(
                    with_mean=self.pipeline_step_arguments.get("scaler__with_mean", True),
                    with_std=self.pipeline_step_arguments.get("scaler__with_std", True),
                ),
            ),
            ("var_thresh", VarianceThreshold(threshold=self.pipeline_step_arguments.get("var_thresh__threshold", 0.0))),
        ]
        steps.append(model_step)  # we append the model step to the pipeline in child classes
        return Pipeline(steps)


class LogisticRegressionModel(BaseModel):
    def __init__(self, model_arguments: dict | None = None):
        model_arguments = model_arguments or LOGISTIC_REGRESSION_PARAMS
        super().__init__(model_arguments)

    def get_pipeline(self) -> Pipeline:
        model_step = ("lr", LogisticRegression(**self.model_arguments))
        return self.create_pipeline(model_step)


class RandomForestModel(BaseModel):
    def __init__(self, model_arguments: dict | None = None):
        model_arguments = model_arguments or RANDOM_FOREST_PARAMS
        super().__init__(model_arguments)

    def get_pipeline(self) -> Pipeline:
        model_step = ("rf", RandomForestClassifier(**self.model_arguments))
        return self.create_pipeline(model_step)


class LightGBMModel(BaseModel):
    def __init__(self, model_arguments: dict | None = None):
        model_arguments = model_arguments or LIGHTGBM_PARAMS
        super().__init__(model_arguments)

    def get_pipeline(self) -> Pipeline:
        model_step = ("lgbm", LGBMClassifier(**self.model_arguments))
        return self.create_pipeline(model_step)
