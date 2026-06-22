import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    het_breuschpagan,
    het_white,
    linear_reset,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera


def compute_vif(X):
    """Variance inflation factor for every regressor in X (X must include 'const')."""
    rows = []
    for idx, col in enumerate(X.columns):
        if col == "const":
            continue
        try:
            vif = variance_inflation_factor(X.values, idx)
        except Exception:
            vif = np.nan
        rows.append({"regressor": col, "VIF": vif})
    return pd.DataFrame(rows)


def run_single_equation_battery(df, bg_lags=4):
    """For every column, regress it on all other columns (OLS w/ constant) and
    run the standard robustness battery on the fitted model.

    Returns: (models dict, coef_df, diagnostics_df, vif_frames dict, warnings list)
    """
    models = {}
    coef_rows = []
    diag_rows = []
    vif_frames = {}
    warnings = []

    for dep in df.columns:
        regressors = [c for c in df.columns if c != dep]
        y = df[dep]
        X = sm.add_constant(df[regressors])

        try:
            model = sm.OLS(y, X).fit()
        except Exception as exc:
            warnings.append(f"OLS failed for dependent='{dep}': {exc}")
            continue

        models[dep] = model

        ci = model.conf_int()
        for name in model.params.index:
            coef_rows.append(
                {
                    "dependent": dep,
                    "regressor": name,
                    "coef": model.params[name],
                    "std_err": model.bse[name],
                    "t_stat": model.tvalues[name],
                    "p_value": model.pvalues[name],
                    "ci_low_95": ci.loc[name, 0],
                    "ci_high_95": ci.loc[name, 1],
                }
            )

        vif_df = compute_vif(X)
        vif_frames[dep] = vif_df
        max_vif = vif_df["VIF"].max() if not vif_df.empty else np.nan

        # Ramsey RESET: functional form misspecification (H0: model is correctly specified)
        try:
            reset_res = linear_reset(model, power=3, use_f=True)
            reset_stat, reset_p = reset_res.fvalue, reset_res.pvalue
        except Exception as exc:
            reset_stat, reset_p = np.nan, np.nan
            warnings.append(f"RESET failed for dependent='{dep}': {exc}")

        # Breusch-Godfrey: autocorrelation in residuals (H0: no serial correlation)
        try:
            bg_lm, bg_p, _, _ = acorr_breusch_godfrey(model, nlags=bg_lags)
        except Exception as exc:
            bg_lm, bg_p = np.nan, np.nan
            warnings.append(f"Breusch-Godfrey failed for dependent='{dep}': {exc}")

        # Breusch-Pagan & White: heteroskedasticity (H0: homoskedastic)
        try:
            bp_lm, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
        except Exception as exc:
            bp_lm, bp_p = np.nan, np.nan
            warnings.append(f"Breusch-Pagan failed for dependent='{dep}': {exc}")
        try:
            white_lm, white_p, _, _ = het_white(model.resid, model.model.exog)
        except Exception as exc:
            white_lm, white_p = np.nan, np.nan
            warnings.append(f"White test failed for dependent='{dep}': {exc}")

        # Jarque-Bera: normality of residuals (H0: normal)
        try:
            jb_stat, jb_p, skew, kurt = jarque_bera(model.resid)
        except Exception as exc:
            jb_stat, jb_p, skew, kurt = np.nan, np.nan, np.nan, np.nan
            warnings.append(f"Jarque-Bera failed for dependent='{dep}': {exc}")

        diag_rows.append(
            {
                "dependent": dep,
                "n_obs": int(model.nobs),
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "f_stat": model.fvalue,
                "f_pvalue": model.f_pvalue,
                "durbin_watson": durbin_watson(model.resid),
                "max_vif": max_vif,
                "multicollinearity_flag_vif_gt_10": (max_vif > 10) if pd.notna(max_vif) else None,
                "ramsey_reset_stat": reset_stat,
                "ramsey_reset_pvalue": reset_p,
                "reset_misspecified_5pct": (reset_p < 0.05) if pd.notna(reset_p) else None,
                "breusch_godfrey_lm_stat": bg_lm,
                "breusch_godfrey_pvalue": bg_p,
                "autocorrelation_flag_5pct": (bg_p < 0.05) if pd.notna(bg_p) else None,
                "breusch_pagan_lm_stat": bp_lm,
                "breusch_pagan_pvalue": bp_p,
                "white_lm_stat": white_lm,
                "white_pvalue": white_p,
                "heteroskedasticity_flag_5pct": (
                    (bp_p < 0.05) or (white_p < 0.05)
                    if pd.notna(bp_p) and pd.notna(white_p)
                    else None
                ),
                "jarque_bera_stat": jb_stat,
                "jarque_bera_pvalue": jb_p,
                "non_normal_residuals_flag_5pct": (jb_p < 0.05) if pd.notna(jb_p) else None,
            }
        )

    coef_df = pd.DataFrame(coef_rows)
    diag_df = pd.DataFrame(diag_rows)
    return models, coef_df, diag_df, vif_frames, warnings
