#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QUBO vs DistMatch ΔMacro-F1 Heatmap

We vary only λ_conf and λ_red for plotting.
Internally we filter to λ_div=0 and γ=1.0, but we do NOT mention them in titles.
Outputs exactly one SVG:
  latex/figures/qubo_vs_distmatch_delta_f1_heatmap_ldiv0_gamma1.0.svg
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# -----------------------------
# Matplotlib (publication safe)
# -----------------------------
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams["font.size"] = 11
mpl.rcParams["axes.unicode_minus"] = False


# -----------------------------
# Helpers
# -----------------------------
def parse_run(r: str):
    out = {
        "method": None,
        "lam_conf": np.nan,
        "lam_red": np.nan,
        "lam_div": np.nan,
        "gamma": np.nan,
        "seed": np.nan,
    }
    r_str = str(r)
    r_lower = r_str.lower()

    if r_lower.startswith("qubo"):
        out["method"] = "QUBO"
    elif r_lower.startswith("distmatch"):
        out["method"] = "DistMatch"

    m = re.search(r"lconf([0-9.]+)", r_str)
    if m:
        out["lam_conf"] = float(m.group(1))

    m = re.search(r"lred([0-9.]+)", r_str)
    if m:
        out["lam_red"] = float(m.group(1))

    m = re.search(r"ldiv([0-9.]+)", r_str)
    if m:
        out["lam_div"] = float(m.group(1))

    m = re.search(r"gamma([0-9.]+)", r_str)
    if m:
        out["gamma"] = float(m.group(1))

    m = re.search(r"_s([0-9]+)", r_str)
    if m:
        out["seed"] = int(m.group(1))

    return out


def plot_heatmap(agg_df: pd.DataFrame, out_path: str):
    if agg_df.empty:
        raise RuntimeError("Nothing to plot: aggregated dataframe is empty.")

    lam_conf_vals = sorted(agg_df["lam_conf"].unique())
    lam_red_vals = sorted(agg_df["lam_red"].unique())

    conf_to_j = {v: j for j, v in enumerate(lam_conf_vals)}
    red_to_i = {v: i for i, v in enumerate(lam_red_vals)}

    grid = np.full((len(lam_red_vals), len(lam_conf_vals)), np.nan)
    for _, r in agg_df.iterrows():
        i = red_to_i[r["lam_red"]]
        j = conf_to_j[r["lam_conf"]]
        grid[i, j] = r["delta_f1"]

    vmax = float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 0.01
    vmax = max(vmax, 0.01)

    plt.figure(figsize=(8, 5.5))
    im = plt.imshow(
        grid,
        origin="lower",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )

    # annotate cells
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            if not np.isnan(val):
                plt.text(
                    j, i,
                    f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=9,
                    color="black" if abs(val) < vmax * 0.7 else "white",
                )

    plt.xticks(range(len(lam_conf_vals)), [str(v) for v in lam_conf_vals])
    plt.yticks(range(len(lam_red_vals)), [str(v) for v in lam_red_vals])

    plt.xlabel(r"$\lambda_{\text{conf}}$")
    plt.ylabel(r"$\lambda_{\text{red}}$")
    plt.title(r"QUBO–DistMatch Macro-F1 Advantage ($\Delta \mathrm{F1}$)")

    cbar = plt.colorbar(im)
    cbar.set_label(r"$\text{Macro-F1}_{\text{QUBO}} - \text{Macro-F1}_{\text{DistMatch}}$")

    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, len(lam_conf_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(lam_red_vals), 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.3)

    plt.tight_layout()
    plt.savefig(out_path, format="svg", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved:", out_path)


# -----------------------------
# Load data
# -----------------------------
RESULTS_DIR = "./steps_datasets/results"
ID_CSV = os.path.join(RESULTS_DIR, "id_eval_summary.csv")

id_df = pd.read_csv(ID_CSV)

# Parse run -> prefix to avoid duplicate columns
parsed = id_df["run"].map(parse_run).apply(pd.Series).add_prefix("pr_")

# Safe concat (no duplicate column names now)
df = pd.concat([id_df, parsed], axis=1)

# Keep only recognized methods from parsed run string
df = df[df["pr_method"].isin(["QUBO", "DistMatch"])].copy()

# -----------------------------
# Internal fixed filters
# -----------------------------
df["pr_lam_div"] = pd.to_numeric(df["pr_lam_div"], errors="coerce").fillna(np.nan)
df["pr_gamma"] = pd.to_numeric(df["pr_gamma"], errors="coerce").fillna(np.nan)

df = df[np.isclose(df["pr_lam_div"].values.astype(float), 0.0, atol=1e-12)]
df = df[np.isclose(df["pr_gamma"].values.astype(float), 1.0, atol=1e-12)]

if df.empty:
    raise RuntimeError("After filtering to λ_div=0 and γ=1.0, no rows remain. Check run names / values.")

# -----------------------------
# Aggregate macro_f1 over seed/priors (everything else)
# We only keep (method, λ_conf, λ_red)
# -----------------------------
if "macro_f1" not in df.columns:
    raise ValueError("macro_f1 column not found in id_eval_summary.csv")

cfg = (
    df.groupby(["pr_method", "pr_lam_conf", "pr_lam_red"], dropna=False)
      .agg(macro_f1=("macro_f1", "mean"))
      .reset_index()
)

qubo = cfg[cfg["pr_method"] == "QUBO"].copy()
dist = cfg[cfg["pr_method"] == "DistMatch"].copy()

paired = qubo.merge(
    dist,
    on=["pr_lam_conf", "pr_lam_red"],
    suffixes=("_qubo", "_dist"),
    how="inner",
)

if paired.empty:
    raise RuntimeError("No overlapping (λ_conf, λ_red) configs between QUBO and DistMatch at γ=1.0.")

paired["delta_f1"] = paired["macro_f1_qubo"] - paired["macro_f1_dist"]

# Rename to clean plotting columns
plot_df = paired.rename(columns={"pr_lam_conf": "lam_conf", "pr_lam_red": "lam_red"})[
    ["lam_conf", "lam_red", "delta_f1"]
]

# -----------------------------
# Plot (single output)
# -----------------------------
OUT_DIR = "latex/figures"
os.makedirs(OUT_DIR, exist_ok=True)

out_path = os.path.join(
    OUT_DIR,
    "qubo_vs_distmatch_delta_f1_heatmap_ldiv0_gamma1.0.svg"
)

plot_heatmap(plot_df, out_path)
print("Done.")
