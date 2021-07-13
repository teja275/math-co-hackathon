from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_log_error


class Regressor(ABC):
    @abstractmethod
    def train(self, *params) -> None:
        pass

    @abstractmethod
    def evaluate(self, *params) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self, *params) -> np.ndarray:
        pass


class SklearnRegressor(Regressor):
    def __init__(
        self, estimator: BaseEstimator, features: List[str], target: str,
    ):
        self.clf = estimator
        self.features = features
        self.target = target

    def train(self, df_train: pd.DataFrame):
        self.clf.fit(df_train[self.features].values, df_train[self.target].values)

    def evaluate(self, df_test: pd.DataFrame):
        evaluation_metric = {"rmsle": np.sqrt(mean_squared_log_error(df_test["Price"], df_test["Price_predicted"]))}
        return evaluation_metric

    def predict(self, df: pd.DataFrame):
        return self.clf.predict(df[self.features].values)
