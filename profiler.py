import pandas as pd
import numpy as np
import json


def load_csv(csv_path: str) -> pd.DataFrame:
    """Loads the CSV, keeping everything as-is (no auto-conversion) so we can
    detect dtype problems ourselves rather than letting Pandas silently guess."""
    return pd.read_csv(csv_path)


def detect_schema(df: pd.DataFrame) -> dict:
    """Basic shape + column info."""
    return {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": {col: str(df[col].dtype) for col in df.columns}
    }


def detect_missing_values(df: pd.DataFrame) -> dict:
    """Which columns have missing values, and how many/what percent."""
    missing = {}
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            missing[col] = {
                "missing_count": int(null_count),
                "missing_percent": round(null_count / len(df) * 100, 2)
            }
    return missing


def detect_dtype_inconsistencies(df: pd.DataFrame) -> dict:
    """Finds columns that LOOK numeric/date but contain values that don't fit
    (e.g., a mostly-numeric column with a stray text string, or a date column
    with an unparseable entry)."""
    issues = {}

    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue

        # Try numeric conversion
        numeric_converted = pd.to_numeric(series, errors="coerce")
        numeric_failures = numeric_converted.isnull().sum()

        # Try date conversion
        date_converted = pd.to_datetime(series, errors="coerce")
        date_failures = date_converted.isnull().sum()

        # Only check text-like columns (skip columns already numeric/datetime).
        is_text_like = series.dtype == object or pd.api.types.is_string_dtype(series)
        if not is_text_like:
            continue

        # If MOST values convert successfully to numeric/date but a few fail,
        # that column likely SHOULD be that type but has bad entries.
        if 0 < numeric_failures < len(series) * 0.5:
            bad_values = series[numeric_converted.isnull()].tolist()
            issues[col] = {
                "likely_type": "numeric",
                "bad_value_count": int(numeric_failures),
                "bad_values": bad_values
            }
        elif 0 < date_failures < len(series) * 0.5:
            bad_values = series[date_converted.isnull()].tolist()
            issues[col] = {
                "likely_type": "date",
                "bad_value_count": int(date_failures),
                "bad_values": bad_values
            }

    return issues


def detect_outliers_iqr(df: pd.DataFrame) -> dict:
    """Uses the IQR (Interquartile Range) method — the standard statistical
    way to flag outliers without needing an AI model:
        lower bound = Q1 - 1.5*IQR
        upper bound = Q3 + 1.5*IQR
    Any value outside these bounds is flagged as an outlier."""
    outliers = {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_rows = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        if not outlier_rows.empty:
            all_values = outlier_rows[col].tolist()
            outliers[col] = {
                "outlier_count": len(outlier_rows),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                # Cap to a small sample - a large dataset can have thousands
                # of outliers, and dumping every single value into an LLM
                # prompt bloats tokens and risks the model misinterpreting
                # each value as a separate issue needing its own fix.
                "outlier_values_sample": all_values[:5],
                "outlier_row_indices_sample": outlier_rows.index.tolist()[:5]
            }

    return outliers


def profile_csv(csv_path: str) -> dict:
    """The main function — this is your Agent 1 (Data Profiler) core logic.
    Later, an LLM will read this structured output and turn it into
    human-readable proposed fixes."""
    df = load_csv(csv_path)

    report = {
        "schema": detect_schema(df),
        "missing_values": detect_missing_values(df),
        "dtype_inconsistencies": detect_dtype_inconsistencies(df),
        "outliers": detect_outliers_iqr(df)
    }
    return report


if __name__ == "__main__":
    result = profile_csv("data/sample.csv")
    print(json.dumps(result, indent=2, default=str))