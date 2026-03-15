from pathlib import Path

import pandas as pd


def read_data(file_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(file_path)


def save_data(df: pd.DataFrame, file_name: str, output_dir: str | Path = "data") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / f"{file_name}.csv"
    df.to_csv(file_path, index=False)

    return file_path
