
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class Preprocessor:
    def __init__(self, drop_cols=None):
        self.drop_cols = drop_cols or []
        self.preprocess: ColumnTransformer | None = None

    def clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df2 = df.copy()
        df2["TotalCharges"] = pd.to_numeric(df2["TotalCharges"], errors="coerce")
        df2["TotalCharges"] = df2["TotalCharges"].fillna(df2["TotalCharges"].median())
        return df2

    def split_xy(self, df: pd.DataFrame):
        df2 = self.clean_df(df)
        y = df2["tenure"].astype(float)
        X = df2.drop(columns=["tenure","customerID"], errors="ignore").copy()
        X = X.drop(columns=self.drop_cols, errors="ignore")
        return X, y

    def fit(self, X: pd.DataFrame):
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]

        num_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocess = ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols)
            ]
        )
        self.preprocess.fit(X)
        return self

    def transform(self, X: pd.DataFrame):
        if self.preprocess is None:
            raise RuntimeError("Primero ejecuta fit(X).")
        return self.preprocess.transform(X)
