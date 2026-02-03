#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Matplotlib config: keep text as real text in SVG / PDF (selectable)
# ---------------------------------------------------------------------
mpl.rcParams["svg.fonttype"] = "none"   # VERY IMPORTANT for selectable text
mpl.rcParams["pdf.fonttype"] = 42       # TrueType in PDF
mpl.rcParams["ps.fonttype"] = 42        # TrueType in PS

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
id_path = "./steps_datasets/results/id_eval_summary.csv"
diag_path = "./steps_datasets/results/selection_diagnostics.csv"

id_df = pd.read_csv(id_path)
diag_df = pd.read_csv(diag_path)

print(len(id_df), "rows in id_eval_summary.csv")
print(len(diag_df), "rows in selection_diagnostics.csv")
print("id_eval_summary columns:", id_df.columns.tolist())
print("selection_diagnostics columns:", diag_df.columns.tolist())

# ---------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------
df = id_df.merge(
    diag_df[["run", "redundancy_mean_sim", "reliability_mean"]],
    on="run",
    how="left",
)
print("Merged columns:", df.columns.tolist())

# ---------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------
out_dir = "latex/figures"
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------------
# Filter to QUBO runs
# ---------------------------------------------------------------------
qubo = df[df["method"] == "qubo"].copy()
print("QUBO rows:", len(qubo))

# Ensure gamma exists
if "gamma" not in qubo.columns:
    qubo["gamma"] = np.nan

gamma_vals = sorted(qubo["gamma"].dropna().unique())
lam_conf_vals_global = sorted(qubo["lam_conf"].dropna().unique())
lam_red_vals_global = sorted(qubo["lam_red"].dropna().unique())

print("λ_conf values (global):", lam_conf_vals_global)
print("λ_red values (global):", lam_red_vals_global)
print("gamma values:", gamma_vals)

# ---------------------------------------------------------------------
# Figure 1: Trade-offs λ_conf, λ_red vs Macro-F1 and redundancy
#           – now ONLY combined over all gamma
# ---------------------------------------------------------------------

# IGNORE gamma as a plotting dimension: use only "all gamma combined"
gamma_vals_for_plot = [None]

for g in gamma_vals_for_plot:
    if g is None:
        qubo_sub = qubo.copy()
        suffix = "gamma_all"
        title_suffix = " (all γ combined)"
        print("\n[Trade-offs] Using all gamma values combined.")
    else:
        # This branch will never run with gamma_vals_for_plot = [None],
        # but kept for minimal code changes.
        qubo_sub = qubo[qubo["gamma"] == g].copy()
        if qubo_sub.empty:
            print(f"[Trade-offs] No rows for gamma={g}, skipping.")
            continue
        suffix = f"gamma{g}"
        title_suffix = f" (γ={g})"
        print(f"\n[Trade-offs] Using gamma={g} subset, rows:", len(qubo_sub))

    lam_conf_vals = sorted(qubo_sub["lam_conf"].dropna().unique())
    lam_red_vals = sorted(qubo_sub["lam_red"].dropna().unique())
    print("  λ_conf values (subset):", lam_conf_vals)
    print("  λ_red values (subset):", lam_red_vals)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    ax11, ax12 = axes[0]
    ax21, ax22 = axes[1]

    # Top-left: λ_conf vs Macro-F1 (lines per λ_red)
    for lr in lam_red_vals:
        sub = qubo_sub[qubo_sub["lam_red"] == lr]
        if sub.empty:
            continue
        sub_g = (
            sub.groupby("lam_conf")[["macro_f1"]]
            .mean()
            .reset_index()
            .sort_values("lam_conf")
        )
        ax11.plot(sub_g["lam_conf"], sub_g["macro_f1"], marker="o", label=f"λ_red={lr}")
    ax11.set_title(f"λ_conf vs Macro-F1{title_suffix}")
    ax11.set_xlabel("λ_conf")
    ax11.set_ylabel("Macro-F1 (mean over seeds)")
    ax11.legend(fontsize=8)

    # Top-right: λ_conf vs redundancy (lines per λ_red)
    for lr in lam_red_vals:
        sub = qubo_sub[qubo_sub["lam_red"] == lr]
        if sub.empty:
            continue
        sub_g = (
            sub.groupby("lam_conf")[["redundancy_mean_sim"]]
            .mean()
            .reset_index()
            .sort_values("lam_conf")
        )
        ax12.plot(
            sub_g["lam_conf"],
            sub_g["redundancy_mean_sim"],
            marker="o",
            label=f"λ_red={lr}",
        )
    ax12.set_title(f"λ_conf vs Redundancy{title_suffix}")
    ax12.set_xlabel("λ_conf")
    ax12.set_ylabel("Redundancy (mean cosine sim)")
    ax12.legend(fontsize=8)

    # Bottom-left: λ_red vs Macro-F1 (lines per λ_conf)
    for lc in lam_conf_vals:
        sub = qubo_sub[qubo_sub["lam_conf"] == lc]
        if sub.empty:
            continue
        sub_g = (
            sub.groupby("lam_red")[["macro_f1"]]
            .mean()
            .reset_index()
            .sort_values("lam_red")
        )
        ax21.plot(sub_g["lam_red"], sub_g["macro_f1"], marker="o", label=f"λ_conf={lc}")
    ax21.set_title(f"λ_red vs Macro-F1{title_suffix}")
    ax21.set_xlabel("λ_red")
    ax21.set_ylabel("Macro-F1 (mean over seeds)")
    ax21.legend(fontsize=8)

    # Bottom-right: λ_red vs redundancy (lines per λ_conf)
    for lc in lam_conf_vals:
        sub = qubo_sub[qubo_sub["lam_conf"] == lc]
        if sub.empty:
            continue
        sub_g = (
            sub.groupby("lam_red")[["redundancy_mean_sim"]]
            .mean()
            .reset_index()
            .sort_values("lam_red")
        )
        ax22.plot(
            sub_g["lam_red"],
            sub_g["redundancy_mean_sim"],
            marker="o",
            label=f"λ_conf={lc}",
        )
    ax22.set_title(f"λ_red vs Redundancy{title_suffix}")
    ax22.set_xlabel("λ_red")
    ax22.set_ylabel("Redundancy (mean cosine sim)")
    ax22.legend(fontsize=8)

    fig.tight_layout()
    out_path_svg = os.path.join(out_dir, f"qubo_tradeoffs_lamconf_lamred_{suffix}.svg")
    fig.savefig(out_path_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path_svg)

# ---------------------------------------------------------------------
# Figure 2: Pareto front – Macro-F1 vs redundancy
#           now ONLY combined over all gamma
# ---------------------------------------------------------------------

gamma_vals_for_pareto = [None]

for g in gamma_vals_for_pareto:
    if g is None:
        pareto_df = qubo.dropna(subset=["macro_f1", "redundancy_mean_sim"]).copy()
        suffix = "gamma_all"
        title_suffix = " (all γ combined)"
        print("\n[Pareto] Using all gamma values combined.")
    else:
        # This branch will never run with gamma_vals_for_pareto = [None],
        # but kept for minimal code changes.
        pareto_df = qubo[qubo["gamma"] == g].dropna(
            subset=["macro_f1", "redundancy_mean_sim"]
        ).copy()
        if pareto_df.empty:
            print(f"[Pareto] No rows for gamma={g}, skipping.")
            continue
        suffix = f"gamma{g}"
        title_suffix = f" (γ={g})"
        print(f"\n[Pareto] Using gamma={g} subset, rows:", len(pareto_df))

    fig, ax = plt.subplots(figsize=(7, 5))

    # Normalise
    f1_min, f1_max = pareto_df["macro_f1"].min(), pareto_df["macro_f1"].max()
    red_min, red_max = pareto_df["redundancy_mean_sim"].min(), pareto_df["redundancy_mean_sim"].max()

    pareto_df["f1_norm"] = (pareto_df["macro_f1"] - f1_min) / (f1_max - f1_min + 1e-8)
    pareto_df["red_norm"] = (pareto_df["redundancy_mean_sim"] - red_min) / (red_max - red_min + 1e-8)

    indices = pareto_df.index.tolist()
    pareto_indices = []
    for i in indices:
        r = pareto_df.loc[i]
        dominated = False
        for j in indices:
            if i == j:
                continue
            s = pareto_df.loc[j]
            if (
                s["macro_f1"] >= r["macro_f1"]
                and s["redundancy_mean_sim"] <= r["redundancy_mean_sim"]
                and (s["macro_f1"] > r["macro_f1"] or s["redundancy_mean_sim"] < r["redundancy_mean_sim"])
            ):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)

    pareto_front = pareto_df.loc[pareto_indices].copy()
    print(
        "Pareto-optimal runs for",
        ("all gamma" if g is None else f"gamma={g}"),
        ":",
    )
    cols_show = [
        "run",
        "macro_f1",
        "redundancy_mean_sim",
        "lam_conf",
        "lam_red",
        "lam_div",
        "gamma",
        "seed",
    ]
    print(pareto_front[cols_show].sort_values("macro_f1", ascending=False).head(10))

    # Scatter all QUBO runs in this subset
    ax.scatter(
        pareto_df["red_norm"],
        pareto_df["f1_norm"],
        alpha=0.2,
        label="All QUBO runs (subset)",
    )

    # Scatter Pareto front
    ax.scatter(
        pareto_front["red_norm"],
        pareto_front["f1_norm"],
        alpha=0.9,
        marker="o",
        label="Pareto front",
    )

    # Best by Macro-F1 on Pareto front
    best = pareto_front.sort_values("macro_f1", ascending=False).iloc[0]
    best_settings = {
        "run": best["run"],
        "lam_conf": float(best["lam_conf"]),
        "lam_red": float(best["lam_red"]),
        "lam_div": float(best["lam_div"]),
        "gamma": float(best["gamma"]) if not pd.isna(best["gamma"]) else np.nan,
        "seed": int(best["seed"]),
        "macro_f1": float(best["macro_f1"]),
        "redundancy_mean_sim": float(best["redundancy_mean_sim"]),
    }
    print("Chosen best_settings:", best_settings)

    bx = best["red_norm"]
    by = best["f1_norm"]
    ax.scatter(bx, by, marker="*", s=120, label="Chosen config")

    ax.set_xlabel("Redundancy (normalised, lower is better)")
    ax.set_ylabel("Macro-F1 (normalised, higher is better)")
    ax.set_title(f"QUBO Pareto F1 vs Redundancy{title_suffix}")
    ax.legend()

    fig.tight_layout()
    out_path_svg = os.path.join(out_dir, f"qubo_pareto_f1_vs_redundancy_{suffix}.svg")
    fig.savefig(out_path_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path_svg)
