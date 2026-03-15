import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


NUMERIC_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "sold_build_diff",
    "ren_diff",
    "statezip_mean_price",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df_features = df.copy()

    df_features["sold_build_diff"] = df_features["date"] - df_features["yr_built"]
    df_features["ren_diff"] = df_features["date"] - df_features["yr_renovated"]
    df_features.loc[df_features["yr_renovated"] == 0, "ren_diff"] = 0

    df_features = df_features.drop(columns=["date", "yr_built", "yr_renovated"])

    df_features.loc[df_features["floors"] == 2.5, "floors"] = 3.0
    df_features.loc[df_features["floors"] == 3.5, "floors"] = 3.0

    return df_features


def add_features_and_split(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df_features = add_features(df)

    X = df_features.drop(columns=["price"])
    y = df_features["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    statezip_mean_price = y_train.groupby(X_train["statezip"]).mean()
    fallback_mean = y_train.mean()

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["statezip_mean_price"] = X_train["statezip"].map(statezip_mean_price).fillna(fallback_mean)
    X_test["statezip_mean_price"] = X_test["statezip"].map(statezip_mean_price).fillna(fallback_mean)

    X_train = X_train.drop(columns=["statezip"])
    X_test = X_test.drop(columns=["statezip"])

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train[NUMERIC_COLUMNS]),
        columns=NUMERIC_COLUMNS,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test[NUMERIC_COLUMNS]),
        columns=NUMERIC_COLUMNS,
        index=X_test.index,
    )
    X_train = X_train.drop(columns=NUMERIC_COLUMNS).join(X_train_scaled)
    X_test = X_test.drop(columns=NUMERIC_COLUMNS).join(X_test_scaled)

    return X_train, y_train, X_test, y_test
