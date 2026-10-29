"""Audit the controlled timing ledger with full Eq. 20 and log-log exponents."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "results" / "exp"
raw = pd.read_csv(EXP / "complexity.csv")
T, S, R_REF = 140, 10, 18

# Full Eq. 20 operation-count design, with realized k in all k-dependent terms.
A = np.column_stack([
    T * raw.n * raw.mean_k * raw.d,
    T * raw.mean_k_squared * raw.d,
    raw.realized_k ** 2 * raw.n * raw.d,
    raw.refinement_active * S * R_REF * raw.n * raw.K * raw.d,
]).astype(float)
y = raw.time_s.to_numpy(float)
ols, *_ = np.linalg.lstsq(A, y, rcond=None)
nnls_coef, _ = nnls(A, y)
pred_ols = A @ ols
pred_nnls = A @ nnls_coef
r2_ols = 1 - ((y - pred_ols) ** 2).sum() / ((y - y.mean()) ** 2).sum()
r2_nnls = 1 - ((y - pred_nnls) ** 2).sum() / ((y - y.mean()) ** 2).sum()

rows = []
for axis in ("n", "K", "d"):
    grouped = raw.groupby(["axis", "value"], as_index=False).agg(
        time_s=("time_s", "mean"), time_sd=("time_s", "std"), realized_k=("realized_k", "mean"))
    grouped = grouped[grouped.axis == axis]
    x = grouped.value.to_numpy(float)
    z = np.log(grouped.time_s.to_numpy(float))
    lx = np.log(x)
    slope, intercept = np.polyfit(lx, z, 1)
    fitted = intercept + slope * lx
    residual_sd = np.sqrt(((z - fitted) ** 2).sum() / max(1, len(x) - 2))
    slope_se = residual_sd / np.sqrt(((lx - lx.mean()) ** 2).sum())
    rows.append({"axis": axis, "slope": slope, "ci95_low": slope - 1.96 * slope_se,
                 "ci95_high": slope + 1.96 * slope_se,
                 "r_squared": np.corrcoef(lx, z)[0, 1] ** 2,
                 "points": len(x)})

summary = pd.DataFrame(rows)
summary.to_csv(EXP / "complexity_loglog_summary.csv", index=False)
report = {
    "generator": {
        "n_axis": "K=8, d=30; fixed circular centre geometry, Gaussian SD 0.3, jitter SD 0.05 only for replication",
        "K_axis": "n=2000, d=2; centre count varied on the same circular geometry family, Gaussian SD 0.3",
        "d_axis": "n=2000, K=8; centres in the first two coordinates, zero offsets in added dimensions, Gaussian SD 0.3",
        "seeds": [3, 7, 11],
    },
    "eq20_ols_coefficients": ols.tolist(),
    "eq20_ols_r_squared": float(r2_ols),
    "eq20_nonnegative_coefficients": nnls_coef.tolist(),
    "eq20_nonnegative_r_squared": float(r2_nnls),
    "realized_k32_mean_sd": [float(raw[(raw.axis == "K") & (raw.value == 32)].realized_k.mean()), float(raw[(raw.axis == "K") & (raw.value == 32)].realized_k.std())],
    "realized_k64_mean_sd": [float(raw[(raw.axis == "K") & (raw.value == 64)].realized_k.mean()), float(raw[(raw.axis == "K") & (raw.value == 64)].realized_k.std())],
    "loglog": summary.to_dict("records"),
}
(EXP / "complexity_loglog_fit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
lines = ["Controlled log-log timing audit", "", "Full Eq. 20 OLS coefficients c1..c4: " + repr(ols.tolist()),
         f"OLS R^2 = {r2_ols:.4f} (descriptive; coefficients are not causal operation costs)",
         "Full Eq. 20 non-negative least-squares coefficients c1..c4: " + repr(nnls_coef.tolist()),
         f"NNLS R^2 = {r2_nnls:.4f} (descriptive only)", "", summary.to_string(index=False)]
(EXP / "complexity_loglog_fit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
