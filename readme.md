# House Price Prediction

This project predicts house prices from tabular housing data using a preprocessing and feature engineering pipeline derived from [`notebooks/main.ipynb`](/Users/ibnsaud/Documents/code/ML/Kaggle/House_price_prediction/notebooks/main.ipynb).

The repository contains:

- exploratory work in the notebook
- reusable preprocessing, feature, training, and evaluation code in [`src/`](/Users/ibnsaud/Documents/code/ML/Kaggle/House_price_prediction/src)
- a small CLI entrypoint in [`main.py`](/Users/ibnsaud/Documents/code/ML/Kaggle/House_price_prediction/main.py)

## Project Structure

```text
.
├── data/
│   └── data.csv
├── notebooks/
│   └── main.ipynb
├── src/
│   ├── clean_data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── train.py
│   └── utils.py
├── main.py
├── readme.md
└── requirements.txt
```

## Pipeline

The current pipeline does the following:

1. Reads the raw CSV dataset.
2. Removes rows where `price <= 0`.
3. Applies `log(price)` as the training target.
4. Converts `date` to the sale year.
5. Drops `country`, `street`, and `city`.
6. Creates two engineered features:
   - `sold_build_diff`: `date - yr_built`
   - `ren_diff`: `date - yr_renovated`, with non-renovated homes reset to `0`
7. Normalizes unusual `floors` values by mapping `2.5` and `3.5` to `3.0`.
8. Splits the data into train and test sets.
9. Encodes `statezip` using the mean target value from the training split.
10. Scales numeric features with `StandardScaler`.
11. Trains a regression model and reports RMSE and MAE on the original price scale.

## Available Models

The CLI currently supports these model names:

- `linear`
- `linearregression`
- `lasso`
- `ridge`
- `decision_tree`
- `decisiontree`
- `tree`

Default model settings come from the notebook:

- `Lasso(alpha=0.05)`
- `DecisionTreeRegressor(max_depth=5, criterion="squared_error", random_state=42)`

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install scikit-learn
```

`scikit-learn` is required by the code in `src/` and `main.py`.

## Usage

Run training from the command line:

```bash
python main.py -d data/data.csv -m lasso
```

Example output:

```text
Model: lasso
RMSE: 308856.24
MAE: 117024.43
```

The cleaned dataset is also saved to:

```text
data/clean_data.csv
```

## Notebook

[`notebooks/main.ipynb`](/Users/ibnsaud/Documents/code/ML/Kaggle/House_price_prediction/notebooks/main.ipynb) contains the exploratory analysis and the experiments that led to the extracted pipeline in `src/`.

The notebook also includes an alternative run that trims the top `1%` of log-price values with `target_quantile=0.99`. That option exists in [`src/clean_data.py`](/Users/ibnsaud/Documents/code/ML/Kaggle/House_price_prediction/src/clean_data.py), but it is not currently exposed by the CLI in [`main.py`](/Users/ibnsaud/Documents/code/ML/Kaggle/House_price_prediction/main.py).

## Modules

- [`src/utils.py`](/Users/ibnsaud/Documents/code/ML/Kaggle/House_price_prediction/src/utils.py): CSV reading and saving helpers
- [`src/clean_data.py`](/Users/ibnsaud/Documents/code/ML/Kaggle/House_price_prediction/src/clean_data.py): raw dataset cleaning
- [`src/features.py`](/Users/ibnsaud/Documents/code/ML/Kaggle/House_price_prediction/src/features.py): feature engineering, split, encoding, and scaling
- [`src/train.py`](/Users/ibnsaud/Documents/code/ML/Kaggle/House_price_prediction/src/train.py): model selection and fitting
- [`src/evaluate.py`](/Users/ibnsaud/Documents/code/ML/Kaggle/House_price_prediction/src/evaluate.py): RMSE and MAE evaluation

## Notes

- Evaluation is done after converting predictions back from log-space with `exp`.
- `statezip` is target-encoded using training data only, with the training mean used as fallback for unseen values.
- The repository currently focuses on local experimentation rather than Kaggle submission packaging.
