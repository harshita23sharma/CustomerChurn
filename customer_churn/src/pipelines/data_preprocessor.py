# Pipeline Building

# 1st Component DataPreprocessor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from src.pipelines.configs import PreprocessConfig


class DataPreprocessor:
    def __init__(self, is_train: bool, data_path: str):
        self.is_train = is_train
        self.data_path = data_path

    def __call__(self):
        self.data = pd.read_csv(self.data_path)

        if self.is_train:
            self.data = self.data.dropna()
            data = DataPreprocessor._clean_df(self.data)
            data = DataPreprocessor._preprocess(data)
            data = DataPreprocessor._split_train_tes(data)
        else:
            data = DataPreprocessor._clean_df(self.data)
            data = DataPreprocessor._preprocess(data)
        return data

    @staticmethod
    def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.fit_transform(
            df[["tenure", "MonthlyCharges", "TotalCharges"]]
        )
        categorical = df.select_dtypes("object")
        number = df.select_dtypes("number").reset_index(drop=True)
        encoder = OrdinalEncoder().fit(categorical)
        encoded = encoder.transform(categorical)
        cate = pd.DataFrame(
            encoded.astype("int64"), columns=categorical.columns
        ).reset_index(drop=True)
        df_final = pd.concat([number, cate], axis=1)
        return df_final

    @staticmethod
    def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df.drop(labels=df[df["tenure"] == 0].index, axis=0, inplace=True)
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        # Replacing 'No internet service' with 'No'
        cols = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        for i in cols:
            df[i].replace("No internet service", "No", inplace=True)
        return df

    @staticmethod
    def _split_train_tes(df: pd.DataFrame) -> pd.DataFrame:
        X = df.drop(columns="Churn")
        y = df["Churn"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=0, test_size=0.25
        )
        train_indices = X_train.index
        test_indices = X_test.index
        df.loc[train_indices].to_csv(PreprocessConfig.train_path, index=False)
        df.loc[test_indices].to_csv(PreprocessConfig.test_path, index=False)
        return df
