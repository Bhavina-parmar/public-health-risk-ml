# model/validate.py

import pandas as pd
from .schema import SCHEMA

def validate_dataframe(df: pd.DataFrame):
    errors = []

    # 1️⃣ Check missing columns
    for col in SCHEMA:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")

    if errors:
        return False, errors

    # 2️⃣ Check types & ranges
    for col, rules in SCHEMA.items():
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"{col} is not numeric")
            continue

        if "min" in rules and (df[col] < rules["min"]).any():
            errors.append(f"{col} has values below {rules['min']}")

        if "max" in rules and (df[col] > rules["max"]).any():
            errors.append(f"{col} has values above {rules['max']}")

        if df[col].isnull().any():
            errors.append(f"{col} contains missing values")

    return len(errors) == 0, errors
