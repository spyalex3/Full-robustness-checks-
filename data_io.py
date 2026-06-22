import pandas as pd


def _detect_date_column(df):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        if df[col].dtype == object:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.95:
                return col
    return None


def load_data(path, sheet_name=0, date_col=None):
    """Load an Excel workbook into a clean numeric time-series DataFrame.

    Returns (df, date_col_used) where df is indexed by the detected/declared
    date column (if any) and contains only numeric variable columns.
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    if date_col is None:
        date_col = _detect_date_column(df)

    if date_col is not None:
        if date_col not in df.columns:
            raise ValueError(f"date column '{date_col}' not found in sheet")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).set_index(date_col)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    dropped = [c for c in df.columns if c not in numeric_cols]
    df = df[numeric_cols].dropna()

    if df.shape[1] < 2:
        raise ValueError(
            "need at least 2 numeric variable columns to run regressions; "
            f"found {df.shape[1]} (dropped non-numeric columns: {dropped})"
        )

    return df, date_col, dropped
