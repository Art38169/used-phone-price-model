from abc import ABC, abstractmethod

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


class BaseModel(ABC):

    @property
    @abstractmethod
    def estimator(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def train(self, X_train, y_train):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def save(self, path):
        raise NotImplementedError
    
class SklearnModel(BaseModel):

    @property
    def estimator(self):
        return self.model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

class LinearRegressionModel(SklearnModel):

    def __init__(self):
        self.model = LinearRegression()

    @property
    def name(self):
        return "linear_regression"   

class RandomForestModel(SklearnModel):

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )

    @property
    def name(self):
        return "random_forest"
    
class XGBoostModel(SklearnModel):

    def __init__(self):
        self.model = XGBRegressor(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
        )

    @property
    def name(self):
        return "xgboost"

class LightGBMModel(SklearnModel):

    def __init__(self):
        self.model = LGBMRegressor(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
        )

    @property
    def name(self):
        return "lightgbm"



def get_models():

    return [
        LinearRegressionModel(),
        RandomForestModel(),
        XGBoostModel(),
        LightGBMModel(),
    ]