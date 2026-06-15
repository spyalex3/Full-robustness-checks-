import os
import shutil

import pandas as pd

README_TEXT = [
    ("Table", "What it contains / how to read it"),
    ("run_info", "Input file, detected date column, variables used, and parameters for this run."),
    (
        "stationarity",
        "ADF (H0: unit root, non-stationary) and KPSS (H0: stationary) per variable. "
        "If ADF says non-stationary and KPSS agrees, consider differencing before "
        "trusting level regressions, or rely on the johansen_cointegration result instead.",
    ),
    (
        "ols_diagnostics",
        "One row per dependent variable (regressed on all other variables, levels, OLS). "
        "Flags: multicollinearity (max VIF > 10), Ramsey RESET (functional form misspecification, "
        "p<0.05 = reject correct specification), Breusch-Godfrey (residual autocorrelation, "
        "p<0.05 = present), Breusch-Pagan/White (heteroskedasticity, p<0.05 = present), "
        "Jarque-Bera (non-normal residuals, p<0.05 = reject normality).",
    ),
    ("ols_coefficients", "Coefficient, std error, t-stat, p-value and 95% CI for every regressor in every single-equation model."),
    ("vif", "Variance Inflation Factor per regressor per equation. VIF > 10 is the conventional multicollinearity warning threshold."),
    ("var_lag_selection", "VAR lag order selected by each information criterion (AIC/BIC/HQIC/FPE) and the lag actually used."),
    (
        "granger_causality",
        "Pairwise Granger causality from the fitted VAR. H0: 'causing' does NOT Granger-cause 'caused'. "
        "p<0.05 = reject H0 (evidence of predictive causality, not structural causality).",
    ),
    (
        "johansen_cointegration",
        "Johansen trace and max-eigenvalue tests for cointegrating relationships among the (assumed I(1)) variables. "
        "'rejects_5pct'=True for rank r means the null of at most r cointegrating vectors is rejected at 5%.",
    ),
    ("irf_data", "Numeric orthogonalized (Cholesky) impulse response values: response of each variable to a one-unit shock in each variable, by horizon."),
    ("irf_plot.png", "Plotted orthogonalized impulse response functions with 95% confidence bands. Note: Cholesky ordering follows the column order in your input file."),
    ("warnings", "Any test that failed to compute (e.g. due to small sample size) and why."),
]


def build_report(
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
):
    """Write one CSV per result table into output_dir, plus the IRF plot PNG."""
    vif_long = []
    for dep, vdf in vif_frames.items():
        for _, row in vdf.iterrows():
            vif_long.append({"dependent": dep, **row.to_dict()})
    vif_df = pd.DataFrame(vif_long)

    warnings_df = pd.DataFrame({"warning": warnings}) if warnings else pd.DataFrame({"warning": []})
    readme_df = pd.DataFrame(README_TEXT, columns=["Table", "Description"])

    tables = {
        "readme": readme_df,
        "run_info": run_info_df,
        "stationarity": stationarity_df,
        "ols_diagnostics": diag_df,
        "ols_coefficients": coef_df,
        "vif": vif_df,
        "var_lag_selection": lag_table_df,
        "granger_causality": granger_df,
        "johansen_cointegration": johansen_df,
        "irf_data": irf_df,
        "warnings": warnings_df,
    }

    os.makedirs(output_dir, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)

    if irf_png_path and os.path.exists(irf_png_path):
        shutil.copyfile(irf_png_path, os.path.join(output_dir, "irf_plot.png"))
