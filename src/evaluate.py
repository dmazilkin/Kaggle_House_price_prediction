import numpy as np
import pandas as pd


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_pred_price = np.exp(y_pred)
    y_true_price = np.exp(y_true)

    rmse = float(np.sqrt(np.mean((y_true_price - y_pred_price) ** 2)))
    mae = float(np.mean(np.abs(y_pred_price - y_true_price)))

    return {"rmse": rmse, "mae": mae}


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    predictions = model.predict(X)
    return evaluate_predictions(y, predictions)
