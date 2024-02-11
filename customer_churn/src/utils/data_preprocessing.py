import logging
from typing import Tuple

import numpy as np
import pandas as pd
from src.clean_data import DataDivideStrategy, DataPreprocessing, LabelEncoding
from typing_extensions import Annotated
from zenml import step


# Define a ZenML step for cleaning and preprocessing data
@step(enable_cache=False)
def cleaning_data(
    df: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    try:
        # Instantiate the DataPreprocessing strategy
        data_preprocessing = DataPreprocessing()

        # Apply data preprocessing to the input DataFrame
        data = data_preprocessing.handle_data(df)

        # Instantiate the LabelEncoding strategy
        feature_encode = LabelEncoding()

        # Apply label encoding to the preprocessed data
        df_encoded = feature_encode.handle_data(data)

        # Log information about the DataFrame columns
        logging.info(df_encoded.columns)
        logging.info("Columns:", len(df_encoded))

        # Instantiate the DataDivideStrategy strategy
        split_data = DataDivideStrategy()

        # Split the encoded data into training and testing sets
        X_train, X_test, y_train, y_test = split_data.handle_data(df_encoded)

        # Return the split data as a tuple
        return X_train, X_test, y_train, y_test
    except Exception as e:
        # Handle and log any errors that occur during data cleaning
        logging.error("Error in step cleaning data", e)
        raise e
