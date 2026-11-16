"""Part 1 — Estimation, inference and a regression baseline."""
import math
import os
import numpy as np
from helpers import load_records


def closed_form_simple_reg(x, y):
    n = len(x)
    mean_x = float(np.sum(x)) / n
    mean_y = float(np.sum(y)) / n
    dx = x - mean_x
    dy = y - mean_y
    cov_xy = float(np.sum(dx * dy)) / (n - 1)
    var_x = float(np.sum(dx * dx)) / (n - 1)
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    fitted = intercept + slope * x
    tss = float(np.sum((y - mean_y) ** 2))
    ess = float(np.sum((fitted - mean_y) ** 2))
    rss = float(np.sum((y - fitted) ** 2))
    r2 = 1.0 - rss / tss
    rmse = math.sqrt(rss / n)
    mae = float(np.mean(np.abs(y - fitted)))
    return {
        "slope": slope,
        "intercept": intercept,
        "fitted": fitted,
        "tss": tss,
        "ess": ess,
        "rss": rss,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
    }


def sample_sd(arr):
    m = float(np.mean(arr))
    return math.sqrt(float(np.sum((arr - m) ** 2)) / (len(arr) - 1))


def main():
    _, X, _, value, _ = load_records()
    n = len(value)

    # Task 1.1
    mean_value = float(np.mean(value))
    sd_value = sample_sd(value)
    se_value = sd_value / math.sqrt(n)
    ci95_low = mean_value - 1.96 * se_value
    ci95_high = mean_value + 1.96 * se_value

    # Task 1.2
    z_stat = (mean_value - 8.0) / se_value
    reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

    # Task 1.3
    n_required = math.ceil((1.96 * sd_value / 0.5) ** 2)

    # Task 1.4: Simple regression of value on X4 (index 3 in X)
    x4 = X[:, 3]
    reg = closed_form_simple_reg(x4, value)
    tss_equals = "yes" if abs(reg["tss"] - reg["ess"] - reg["rss"]) < 1e-6 * reg["tss"] else "no"

    # Task 1.5: Gradient descent on standardised X4
    mean_x4 = float(np.mean(x4))
    sd_x4 = sample_sd(x4)
    z = (x4 - mean_x4) / sd_x4
    m = 0.0
    c = 0.0
    eta = 0.1
    j_hist = {}
    j_hist[0] = float(np.mean((value - (m * z + c)) ** 2))
    for step in range(1, 501):
        e = value - (m * z + c)
        dm = -(2.0 / n) * float(np.sum(z * e))
        dc = -(2.0 / n) * float(np.sum(e))
        m = m - eta * dm
        c = c - eta * dc
        if step in (1, 10, 100, 500):
            j_hist[step] = float(np.mean((value - (m * z + c)) ** 2))
    gd_slope = m / sd_x4
    gd_intercept = c - m * mean_x4 / sd_x4
    gd_slope_gap = abs(gd_slope - reg["slope"])

    # Task 1.6: Funnel check and log remedy
    order_raw = np.argsort(reg["fitted"], kind="stable")
    res_raw = value - reg["fitted"]
    resid_sd_lower_raw = sample_sd(res_raw[order_raw[:400]])
    resid_sd_upper_raw = sample_sd(res_raw[order_raw[400:]])
    resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

    ln_x4 = np.log(x4)
    ln_val = np.log(value)
    reg_log = closed_form_simple_reg(ln_x4, ln_val)
    order_log = np.argsort(reg_log["fitted"], kind="stable")
    res_log = ln_val - reg_log["fitted"]
    resid_sd_lower_log = sample_sd(res_log[order_log[:400]])
    resid_sd_upper_log = sample_sd(res_log[order_log[400:]])
    resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log

    # Task 1.7: Multiple regression by normal equation
    # Columns: ones, ln(X4), X1, X2, X3, X5, X6, X7
    X_mr = np.column_stack([
        np.ones(n, dtype=np.float64),
        ln_x4,
        X[:, 0],  # X1
        X[:, 1],  # X2
        X[:, 2],  # X3
        X[:, 4],  # X5
        X[:, 5],  # X6
        X[:, 6],  # X7
    ])
    beta = np.linalg.solve(X_mr.T @ X_mr, X_mr.T @ ln_val)
    fitted_mr = X_mr @ beta
    tss_mr = float(np.sum((ln_val - np.mean(ln_val)) ** 2))
    rss_mr = float(np.sum((ln_val - fitted_mr) ** 2))
    mr_r2 = 1.0 - rss_mr / tss_mr
    p_mr = 7
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - p_mr - 1)
    loglog_adj_r2 = 1.0 - (1.0 - reg_log["r2"]) * (n - 1) / (n - 1 - 1)

    lines = [
        f"n = {n}",
        f"mean_value = {mean_value:.4f}",
        f"sd_value = {sd_value:.4f}",
        f"se_value = {se_value:.4f}",
        f"ci95_low = {ci95_low:.4f}",
        f"ci95_high = {ci95_high:.4f}",
        f"z_stat = {z_stat:.4f}",
        f"reject_h0 = {reject_h0}",
        f"n_required = {n_required}",
        f"reg_slope = {reg['slope']:.4f}",
        f"reg_intercept = {reg['intercept']:.4f}",
        f"tss = {reg['tss']:.4f}",
        f"ess = {reg['ess']:.4f}",
        f"rss = {reg['rss']:.4f}",
        f"tss_equals_ess_plus_rss = {tss_equals}",
        f"r2 = {reg['r2']:.4f}",
        f"rmse = {reg['rmse']:.4f}",
        f"mae = {reg['mae']:.4f}",
        f"gd_J0 = {j_hist[0]:.4f}",
        f"gd_J1 = {j_hist[1]:.4f}",
        f"gd_J10 = {j_hist[10]:.4f}",
        f"gd_J100 = {j_hist[100]:.4f}",
        f"gd_J500 = {j_hist[500]:.4f}",
        f"gd_slope = {gd_slope:.4f}",
        f"gd_intercept = {gd_intercept:.4f}",
        f"gd_slope_gap = {gd_slope_gap:.4f}",
        f"resid_sd_lower_raw = {resid_sd_lower_raw:.4f}",
        f"resid_sd_upper_raw = {resid_sd_upper_raw:.4f}",
        f"resid_sd_ratio_raw = {resid_sd_ratio_raw:.4f}",
        f"loglog_intercept = {reg_log['intercept']:.4f}",
        f"loglog_slope = {reg_log['slope']:.4f}",
        f"loglog_r2 = {reg_log['r2']:.4f}",
        f"loglog_rmse = {reg_log['rmse']:.4f}",
        f"resid_sd_lower_log = {resid_sd_lower_log:.4f}",
        f"resid_sd_upper_log = {resid_sd_upper_log:.4f}",
        f"resid_sd_ratio_log = {resid_sd_ratio_log:.4f}",
    ]
    for i, b_val in enumerate(beta):
        lines.append(f"mr_beta_{i} = {b_val:.4f}")
    lines.append(f"mr_r2 = {mr_r2:.4f}")
    lines.append(f"mr_adj_r2 = {mr_adj_r2:.4f}")
    lines.append(f"loglog_adj_r2 = {loglog_adj_r2:.4f}")

    os.makedirs("outputs", exist_ok=True)
    with open(os.path.join("outputs", "part1_results.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Part 1 complete. Key checkpoints: n =", n, "mean_value =", f"{mean_value:.4f}")


if __name__ == "__main__":
    main()
