"""Generates examples/co2_weekly.csv used in the example report.

statsmodels.datasets.co2 is a single weekly series (Mauna Loa atmospheric
CO2, ~2,284 observations, 1958-2001) with a few NaN gaps. This toolkit needs
>= 2 numeric variable columns to run VAR/Granger/Johansen, so a synthetic
second series ("proxy") is added: a noisy, lagged function of CO2.

Run from the repo root:
    python examples/generate_co2_example.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

co2 = sm.datasets.co2.load_pandas().data["co2"].interpolate()

rng = np.random.default_rng(0)
proxy = 0.6 * co2.shift(1).bfill() + rng.normal(0, 1.5, size=len(co2)).cumsum() * 0.05

df = pd.DataFrame({"date": co2.index, "co2": co2.values, "proxy": proxy.values})
df.to_csv("examples/co2_weekly.csv", index=False)
print(f"Wrote examples/co2_weekly.csv ({df.shape[0]} rows x {df.shape[1]} cols)")
