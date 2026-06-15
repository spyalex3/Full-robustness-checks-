"""
Time-series econometrics robustness toolkit.

Usage:
    python run.py path/to/data.csv
    python run.py path/to/data.csv -o results_dir --maxlags 6 --irf-periods 12

Point it at a CSV file where each column is a variable (one column may be a
date/time index - it will be auto-detected). It will:
  1. Run ADF + KPSS stationarity tests on every variable.
  2. Regress every variable on all the others (OLS) and run a robustness
     battery on each equation: multicollinearity (VIF), Ramsey RESET,
     Breusch-Godfrey (autocorrelation), Breusch-Pagan & White
     (heteroskedasticity), Jarque-Bera (normality).
  3. Fit a VAR across all variables, run pairwise Granger causality,
     Johansen cointegration, and impulse response functions (IRFs).
All results are written as CSV files (one per test, plus a readme.csv) into
an output directory, alongside an irf_plot.png.
"""

import argparse
import os
import sys
import tempfile
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

from data_io import load_data
from report import build_report
from single_equation import run_single_equation_battery
from stationarity import run_stationarity_tests
from var_analysis import (
    compute_irf,
    fit_var,
    granger_causality_matrix,
    irf_to_dataframe,
    johansen_cointegration,
    lag_selection_table,
    save_irf_plot,
)

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a battery of time-series econometrics robustness checks on a CSV dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Path to the input CSV file (.csv)")
    parser.add_argument("-o", "--output", default=None, help="Output report directory (default: <input>_robustness_report/)")
    parser.add_argument("--date-col", default=None, help="Name of the date/time column, if not auto-detected")
    parser.add_argument("--maxlags", type=int, default=8, help="Max lags considered for VAR lag-order selection (default: 8)")
    parser.add_argument("--ic", default="aic", choices=["aic", "bic", "hqic", "fpe"], help="Information criterion for VAR lag selection (default: aic)")
    parser.add_argument("--irf-periods", type=int, default=10, help="Number of horizons for impulse response functions (default: 10)")
    parser.add_argument("--bg-lags", type=int, default=4, help="Number of lags for the Breusch-Godfrey autocorrelation test (default: 4)")
    return parser.parse_args()


def main():
    args = parse_args()
    warnings = []

    df, date_col, dropped_cols = load_data(args.input, date_col=args.date_col)
    print(f"Loaded {df.shape[0]} observations x {df.shape[1]} variables: {list(df.columns)}")
    if date_col:
        print(f"Detected date/time index column: {date_col}")
    if dropped_cols:
        print(f"Dropped non-numeric columns: {dropped_cols}")
        warnings.append(f"Dropped non-numeric columns: {dropped_cols}")

    print("Running stationarity tests (ADF, KPSS)...")
    stationarity_df = run_stationarity_tests(df)

    print("Running single-equation OLS robustness battery...")
    models, coef_df, diag_df, vif_frames, sb_warnings = run_single_equation_battery(df, bg_lags=args.bg_lags)
    warnings.extend(sb_warnings)

    print("Fitting VAR system...")
    irf_df = pd.DataFrame()
    lag_table_df = pd.DataFrame()
    granger_df = pd.DataFrame()
    johansen_df = pd.DataFrame()
    irf_png_path = None
    tmp_png_handle = None

    try:
        fitted, lag_order_results, selected_lag = fit_var(df, maxlags=args.maxlags, ic=args.ic)
        lag_table_df = lag_selection_table(lag_order_results)
        lag_table_df.loc[len(lag_table_df)] = ["used", selected_lag]
        print(f"  Selected VAR lag order ({args.ic.upper()}): {selected_lag}")

        print("Running pairwise Granger causality tests...")
        granger_df = granger_causality_matrix(fitted, list(df.columns))

        print("Running Johansen cointegration test...")
        johansen_df = johansen_cointegration(df, det_order=0, k_ar_diff=selected_lag)

        print("Computing impulse response functions...")
        irf = compute_irf(fitted, periods=args.irf_periods)
        irf_df = irf_to_dataframe(irf, list(df.columns))

        tmp_png_handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        irf_png_path = tmp_png_handle.name
        save_irf_plot(irf, irf_png_path)
    except Exception as exc:
        warnings.append(f"VAR analysis failed: {exc}")
        print(f"  WARNING: VAR analysis failed: {exc}")

    output_dir = args.output or os.path.splitext(args.input)[0] + "_robustness_report"

    run_info_df = pd.DataFrame(
        [
            {"key": "input_file", "value": os.path.abspath(args.input)},
            {"key": "run_timestamp", "value": datetime.now().isoformat(timespec="seconds")},
            {"key": "n_observations", "value": df.shape[0]},
            {"key": "variables", "value": ", ".join(df.columns)},
            {"key": "date_column", "value": date_col or "(none detected)"},
            {"key": "var_maxlags_searched", "value": args.maxlags},
            {"key": "var_lag_selection_criterion", "value": args.ic},
            {"key": "irf_periods", "value": args.irf_periods},
            {"key": "breusch_godfrey_lags", "value": args.bg_lags},
        ]
    )

    print(f"Writing report to {output_dir} ...")
    build_report(
        output_dir,
        run_info_df,
        stationarity_df,
        diag_df,
        coef_df,
        vif_frames,
        lag_table_df,
        granger_df,
        johansen_df,
        irf_df,
        irf_png_path,
        warnings,
    )

    if tmp_png_handle is not None:
        tmp_png_handle.close()
        try:
            os.remove(irf_png_path)
        except OSError:
            pass

    print(f"Done. Report saved to: {os.path.abspath(output_dir)}")
    if warnings:
        print(f"({len(warnings)} warning(s) - see warnings.csv in the report)")


if __name__ == "__main__":
    main()
