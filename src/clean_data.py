import numpy as np
import pandas as pd


DROP_COLUMNS = ["country", "street", "city"]


def clean_raw_data(df: pd.DataFrame, target_quantile: float | None = None) -> pd.DataFrame:
    df_clean = df.copy()

    df_clean = df_clean[df_clean["price"] > 0].copy()
    df_clean["price"] = np.log(df_clean["price"])

    if target_quantile is not None:
        threshold = df_clean["price"].quantile(target_quantile)
        df_clean = df_clean[df_clean["price"] <= threshold].copy()

    df_clean["date"] = pd.to_datetime(df_clean["date"]).dt.year
    df_clean = df_clean.drop(columns=DROP_COLUMNS, errors="ignore")

    return df_clean.reset_index(drop=True)
