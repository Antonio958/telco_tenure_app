
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge

class RegressorTrainer:
    def __init__(self, preprocessor, n_splits=5, random_state=42):
        self.preprocessor = preprocessor
        self.cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def _eval(self, model, X_raw, y, name):
        scoring = {
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2"
        }
        pipe = Pipeline(steps=[
            ("prep", self.preprocessor.preprocess),
            ("model", model)
        ])
        out = cross_validate(pipe, X_raw, y, cv=self.cv, scoring=scoring)
        return {
            "Modelo": name,
            "MAE": -np.mean(out["test_mae"]),
            "RMSE": -np.mean(out["test_rmse"]),
            "R2": np.mean(out["test_r2"])
        }

    def compare(self, X_raw, y):
        rows = []
        rows.append(self._eval(KNeighborsRegressor(n_neighbors=15, weights="distance"), X_raw, y, "KNN(base k=15)"))
        rows.append(self._eval(SVR(C=10.0, epsilon=0.1), X_raw, y, "SVR(C=10, eps=0.1)"))
        rows.append(self._eval(BayesianRidge(), X_raw, y, "BayesianRidge"))
        return pd.DataFrame(rows).sort_values("MAE")

    def knn_gridsearch(self, X_raw, y):
        pipe = Pipeline(steps=[
            ("prep", self.preprocessor.preprocess),
            ("model", KNeighborsRegressor())
        ])
        param_grid = {
            "model__n_neighbors": [3,5,7,9,11,15,21,31],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2]
        }
        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=self.cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )
        grid.fit(X_raw, y)
        return grid.best_params_, -grid.best_score_
