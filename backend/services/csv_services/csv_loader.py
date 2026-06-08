from pathlib import Path

import pandas as pd


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = normalize_columns(df.columns)
    return df


def normalize_columns(columns) -> list[str]:
    return [column.replace("\ufeff", "").strip() for column in columns]
