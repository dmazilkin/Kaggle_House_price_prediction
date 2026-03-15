from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor


def get_model(model_name: str):
    model_key = model_name.lower()

    models = {
        "linear": LinearRegression(),
        "linearregression": LinearRegression(),
        "lasso": Lasso(alpha=0.05),
        "ridge": Ridge(),
        "decision_tree": DecisionTreeRegressor(criterion="squared_error", max_depth=5, random_state=42),
        "decisiontree": DecisionTreeRegressor(criterion="squared_error", max_depth=5, random_state=42),
        "tree": DecisionTreeRegressor(criterion="squared_error", max_depth=5, random_state=42),
    }

    if model_key not in models:
        supported = ", ".join(sorted(models))
        raise ValueError(f"Unsupported model '{model_name}'. Choose one of: {supported}.")

    return models[model_key]


def train_model(model_name: str, X_train, y_train):
    model = get_model(model_name)
    model.fit(X_train, y_train)
    return model
