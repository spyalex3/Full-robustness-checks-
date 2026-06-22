# Econ Robustness Toolkit

Point it at an Excel file of time-series data, run one command, get back an Excel
report with a full battery of econometric robustness checks.

## What it does

Given a spreadsheet where each column is a variable (one column may be a date,
auto-detected):

1. **Stationarity** — ADF and KPSS tests on every variable.
2. **Single-equation OLS** — regresses every variable on all the others, then
   runs on each equation:
   - Variance Inflation Factor (multicollinearity)
   - Ramsey RESET (functional form misspecification)
   - Breusch-Godfrey (residual autocorrelation)
   - Breusch-Pagan & White (heteroskedasticity)
   - Jarque-Bera (residual normality)
   - Durbin-Watson
3. **VAR system** — fits a VAR (lag order chosen by AIC/BIC/HQIC/FPE), then runs:
   - Pairwise Granger causality
   - Johansen cointegration (trace & max-eigenvalue)
   - Orthogonalized impulse response functions (data + plot)

Everything is written to a single Excel workbook, one sheet per test, plus a
`ReadMe` sheet explaining how to interpret each one.

## Usage

```bash
pip install -r requirements.txt
python run.py path/to/data.xlsx
```

Output defaults to `<input>_robustness_report.xlsx` next to your input file.

Useful flags:

```bash
python run.py data.xlsx -o results.xlsx --maxlags 6 --ic bic --irf-periods 12 --bg-lags 4 --date-col Date
```

| Flag | Meaning | Default |
|---|---|---|
| `-o/--output` | Output report path | `<input>_robustness_report.xlsx` |
| `--sheet` | Sheet name/index to read from input | first sheet |
| `--date-col` | Name of the date column, if not auto-detected | auto-detect |
| `--maxlags` | Max lags searched for VAR lag-order selection | 8 |
| `--ic` | Information criterion for VAR lag selection (`aic`/`bic`/`hqic`/`fpe`) | `aic` |
| `--irf-periods` | Horizons for impulse response functions | 10 |
| `--bg-lags` | Lags for the Breusch-Godfrey test | 4 |

## Notes

- The single-equation OLS regressions use levels, not differences. Check the
  `Stationarity` sheet first — if a variable is flagged non-stationary, lean on
  the `Johansen_Cointegration` sheet rather than the raw OLS coefficients for
  that relationship.
- Granger causality here means predictive precedence within the fitted VAR, not
  structural causality.
- The IRF Cholesky ordering follows the column order of your input file.
