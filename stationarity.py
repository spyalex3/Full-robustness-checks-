import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def run_stationarity_tests(df):
    """ADF (H0: unit root / non-stationary) and KPSS (H0: stationary) per variable."""
    rows = []
    for col in df.columns:
        series = df[col].dropna()

        try:
            adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
        except Exception:
            adf_stat, adf_p = np.nan, np.nan

        try:
            kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags="auto")
        except Exception:
            kpss_stat, kpss_p = np.nan, np.nan

        rows.append(
            {
                "variable": col,
                "adf_stat": adf_stat,
                "adf_pvalue": adf_p,
                "adf_says_stationary_5pct": (adf_p < 0.05) if pd.notna(adf_p) else None,
                "kpss_stat": kpss_stat,
                "kpss_pvalue": kpss_p,
                "kpss_says_stationary_5pct": (kpss_p > 0.05) if pd.notna(kpss_p) else None,
            }
        )
    return pd.DataFrame(rows)
