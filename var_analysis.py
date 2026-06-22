import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def fit_var(df, maxlags=8, ic="aic"):
    """Fit a VAR with lag order chosen by information criterion `ic`."""
    maxlags = max(1, min(maxlags, (len(df) - 1) // (df.shape[1] + 1)))
    model = VAR(df)
    lag_order_results = model.select_order(maxlags=maxlags)
    selected_lag = lag_order_results.selected_orders.get(ic, 1)
    selected_lag = max(1, int(selected_lag))
    fitted = model.fit(selected_lag)
    return fitted, lag_order_results, selected_lag


def lag_selection_table(lag_order_results):
    return pd.DataFrame(lag_order_results.selected_orders.items(), columns=["criterion", "selected_lag"])


def granger_causality_matrix(fitted, variables, signif=0.05):
    """Pairwise Granger causality within the fitted VAR system.

    H0: `causing` does NOT Granger-cause `caused`.
    """
    rows = []
    for caused in variables:
        for causing in variables:
            if causing == caused:
                continue
            try:
                test = fitted.test_causality(caused, [causing], kind="f", signif=signif)
                rows.append(
                    {
                        "causing": causing,
                        "caused": caused,
                        "test_stat": test.test_statistic,
                        "p_value": test.pvalue,
                        "causing_granger_causes_caused_5pct": test.pvalue < signif,
                    }
                )
            except Exception:
                rows.append(
                    {
                        "causing": causing,
                        "caused": caused,
                        "test_stat": np.nan,
                        "p_value": np.nan,
                        "causing_granger_causes_caused_5pct": None,
                    }
                )
    return pd.DataFrame(rows)


def johansen_cointegration(df, det_order=0, k_ar_diff=1):
    """Johansen trace/max-eigenvalue cointegration test.

    det_order: -1 no deterministic terms, 0 constant, 1 constant+trend.
    """
    result = coint_johansen(df, det_order, k_ar_diff)
    rows = []
    n = df.shape[1]
    for i in range(n):
        rows.append(
            {
                "null_rank_le": i,
                "trace_stat": result.lr1[i],
                "trace_crit_90": result.cvt[i, 0],
                "trace_crit_95": result.cvt[i, 1],
                "trace_crit_99": result.cvt[i, 2],
                "trace_rejects_5pct": result.lr1[i] > result.cvt[i, 1],
                "max_eig_stat": result.lr2[i],
                "max_eig_crit_90": result.cvm[i, 0],
                "max_eig_crit_95": result.cvm[i, 1],
                "max_eig_crit_99": result.cvm[i, 2],
                "max_eig_rejects_5pct": result.lr2[i] > result.cvm[i, 1],
            }
        )
    return pd.DataFrame(rows)


def compute_irf(fitted, periods=10):
    return fitted.irf(periods)


def irf_to_dataframe(irf, variables):
    """Long-format table of orthogonalized (Cholesky) impulse responses."""
    arr = irf.orth_irfs  # shape (periods+1, k, k): [horizon, response_i, impulse_j]
    rows = []
    for t in range(arr.shape[0]):
        for i, resp in enumerate(variables):
            for j, imp in enumerate(variables):
                rows.append({"period": t, "impulse": imp, "response": resp, "value": arr[t, i, j]})
    return pd.DataFrame(rows)


def save_irf_plot(irf, png_path):
    fig = irf.plot(orth=True)
    fig.set_size_inches(11, 8.5)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
