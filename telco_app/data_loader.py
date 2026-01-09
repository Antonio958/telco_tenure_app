
import pandas as pd

class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        self.validate(df)
        return df

    def validate(self, df: pd.DataFrame) -> None:
        required = {
            "customerID","gender","SeniorCitizen","Partner","Dependents","tenure",
            "PhoneService","MultipleLines","InternetService","OnlineSecurity",
            "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
            "StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
            "MonthlyCharges","TotalCharges","Churn"
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")
