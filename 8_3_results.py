import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------
# SVG + text selectability settings
# ---------------------------------
mpl.rcParams["svg.fonttype"] = "none"              # keep text as text (selectable)
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]  # clean font
mpl.rcParams["font.size"] = 11                     # base font size
mpl.rcParams["axes.unicode_minus"] = False         # avoid minus-sign issues

# ============================================================
# 1. Config
# ============================================================

# <<< CHANGE THIS BACK IN YOUR ENV >>>
RESULTS_DIR = "./steps_datasets/results"
#RESULTS_DIR = "/mnt/data"  # for testing

ID_CSV = os.path.join(RESULTS_DIR, "id_eval_summary.csv")
DIAG_CSV = os.path.join(RESULTS_DIR, "selection_diagnostics.csv")

OUT_DIR = "latex/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 2. Load & merge CSVs (new format already has lam_* + method)
# ============================================================

id_df = pd.read_csv(ID_CSV)
diag_df = pd.read_csv(DIAG_CSV)

# Only take extra diagnostics from selection_diagnostics to avoid
# duplicate lam_* columns after merge.
diag_extra = diag_df[["run", "redundancy_mean_sim", "reliability_mean"]]

df = id_df.merge(diag_extra, on="run", how="left")

# Normalise method names to canonical strings
df["method"] = (
    df["method"]
    .astype(str)
    .str.lower()
    .map({"qubo": "QUBO", "distmatch": "DistMatch"})
)

print("Methods in df:\n", df["method"].value_counts(dropna=False), "\n")
print("Columns in df:", df.columns.tolist(), "\n")

# Keep only recognized methods
df = df[df["method"].isin(["QUBO", "DistMatch"])].copy()

# ============================================================
# 3. ΔF1 heatmap: QUBO vs DistMatch (shared λ_conf, λ_red, λ_div)
# ============================================================

group_cols = ["method", "lam_conf", "lam_red", "lam_div"]

cfg_summary = (
    df.groupby(group_cols, dropna=False)
      .agg(macro_f1_mean=("macro_f1", "mean"))
      .reset_index()
)

print("Config summary (head):")
print(cfg_summary.head(), "\n")

qubo_cfg = cfg_summary[cfg_summary["method"] == "QUBO"].copy()
dm_cfg   = cfg_summary[cfg_summary["method"] == "DistMatch"].copy()

paired = qubo_cfg.merge(
    dm_cfg,
    on=["lam_conf", "lam_red", "lam_div"],
    suffixes=("_qubo", "_dm"),
    how="inner",
)

print("Number of paired configs:", len(paired))
print("Paired configs (head):")
print(paired.head(), "\n")

if paired.empty:
    raise RuntimeError("No overlapping QUBO/DistMatch configs to compare.")

paired["delta_f1"] = paired["macro_f1_mean_qubo"] - paired["macro_f1_mean_dm"]

# Collapse over lam_div if you have multiple diversity settings
agg = (
    paired.groupby(["lam_conf", "lam_red"])
          .agg(delta_f1=("delta_f1", "mean"))
          .reset_index()
)

print("Collapsed over lam_div (head):")
print(agg.head(), "\n")

lam_conf_vals = sorted(agg["lam_conf"].unique())
lam_red_vals  = sorted(agg["lam_red"].unique())

conf_to_idx = {v: i for i, v in enumerate(lam_conf_vals)}
red_to_idx  = {v: i for i, v in enumerate(lam_red_vals)}

grid = np.full((len(lam_red_vals), len(lam_conf_vals)), np.nan)

for _, row in agg.iterrows():
    i = red_to_idx[row["lam_red"]]
    j = conf_to_idx[row["lam_conf"]]
    grid[i, j] = row["delta_f1"]

# --------------- Plot ΔF1 heatmap ---------------

plt.figure(figsize=(7, 5))

# Symmetric scale around zero so neutral is "no advantage"
vmax = np.nanmax(np.abs(grid))
if np.isnan(vmax) or vmax == 0:
    vmax = 0.01  # fallback

im = plt.imshow(
    grid,
    origin="lower",
    cmap="coolwarm",
    vmin=-vmax,
    vmax=vmax,
    aspect="auto",
)

# Annotate each cell with its ΔF1 value
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        val = grid[i, j]
        if not np.isnan(val):
            plt.text(
                j, i,
                f"{val:.3f}",
                ha="center", va="center",
                fontsize=8,
                color="black" if abs(val) < vmax * 0.7 else "white",
            )

plt.xticks(
    ticks=np.arange(len(lam_conf_vals)),
    labels=[str(v) for v in lam_conf_vals],
)
plt.yticks(
    ticks=np.arange(len(lam_red_vals)),
    labels=[str(v) for v in lam_red_vals],
)

plt.xlabel(r"$\lambda_{conf}$")
plt.ylabel(r"$\lambda_{red}$")
plt.title(r"QUBO–DistMatch Macro-F1 Advantage ($\Delta \mathrm{F1}$)")

cbar = plt.colorbar(im)
cbar.set_label(r"$\mathrm{Macro\text{-}F1}_{QUBO} - \mathrm{Macro\text{-}F1}_{DistMatch}$")

plt.gca().set_xticks(np.arange(-0.5, len(lam_conf_vals), 1), minor=True)
plt.gca().set_yticks(np.arange(-0.5, len(lam_red_vals), 1), minor=True)
plt.grid(which="minor", color="gray", linestyle='-', linewidth=0.3)

plt.tight_layout()
out_path_delta = os.path.join(OUT_DIR, "qubo_vs_distmatch_delta_f1_heatmap.svg")
plt.savefig(out_path_delta, format="svg", bbox_inches="tight")
plt.close()

print("Saved ΔF1 heatmap to:", out_path_delta)

# ============================================================
# 4. QUBO advantage vs DistMatch baseline (scatter + heatmap)
#    using diagnostics CSV only (no jsonl / TF-IDF needed)
# ============================================================

dist_df = df[df["method"] == "DistMatch"].copy()
if dist_df.empty:
    raise RuntimeError("No DistMatch rows found – cannot define baseline.")

# Baseline Macro-F1 and redundancy from all DistMatch runs
f1_dm = dist_df["macro_f1"].mean()
red_dm = dist_df["redundancy_mean_sim"].mean()

print(f"DistMatch baseline Macro-F1 (mean over runs): {f1_dm:.6f}")
print(f"DistMatch baseline redundancy (mean cosine sim): {red_dm:.6f}\n")

qubo_df = df[df["method"] == "QUBO"].copy()

group_cols_adv = ["lam_conf", "lam_red"]

qubo_cfg_adv = (
    qubo_df
    .groupby(group_cols_adv, as_index=False)
    .agg({
        "macro_f1": "mean",
        "redundancy_mean_sim": "mean",
    })
    .rename(columns={
        "macro_f1": "macro_f1_qubo",
        "redundancy_mean_sim": "red_qubo",
    })
)

# Advantage vs DistMatch baseline
qubo_cfg_adv["delta_f1"] = qubo_cfg_adv["macro_f1_qubo"] - f1_dm
qubo_cfg_adv["delta_red"] = qubo_cfg_adv["red_qubo"] - red_dm

print("QUBO config summary with advantages (head):")
print(qubo_cfg_adv.head(), "\n")

# ---------- Scatter: ΔRed vs ΔF1 ----------

scatter_path = os.path.join(OUT_DIR, "qubo_advantage_scatter.svg")

plt.figure(figsize=(7, 5))

sc = plt.scatter(
    qubo_cfg_adv["delta_red"],
    qubo_cfg_adv["delta_f1"],
    c=qubo_cfg_adv["delta_f1"],
    cmap="coolwarm",
    edgecolors="black",
    linewidths=0.6,
)

plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
plt.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)

plt.xlabel(r"$\Delta$ Redundancy (QUBO - DistMatch)")
plt.ylabel(r"$\Delta$ Macro-F1 (QUBO - DistMatch)")
plt.title("QUBO Advantage: Accuracy–Redundancy Space")

cbar = plt.colorbar(sc)
cbar.set_label(r"$\Delta$ Macro-F1")

# Annotate each point with (lam_conf, lam_red)
for _, row in qubo_cfg_adv.iterrows():
    lc = row["lam_conf"]
    lr = row["lam_red"]
    plt.annotate(
        f"{lc},{lr}",
        (row["delta_red"], row["delta_f1"]),
        textcoords="offset points",
        xytext=(2, 2),
        fontsize=6,
    )

plt.tight_layout()
plt.savefig(scatter_path, format="svg", bbox_inches="tight")
plt.close()
print("Saved scatter advantage plot to:", scatter_path)

# ---------- Heatmap: advantage over (λ_conf, λ_red) ----------

heatmap_adv_path = os.path.join(OUT_DIR, "qubo_advantage_map_delta_f1.svg")

lam_conf_vals2 = sorted(qubo_cfg_adv["lam_conf"].unique())
lam_red_vals2  = sorted(qubo_cfg_adv["lam_red"].unique())

conf_to_idx2 = {v: i for i, v in enumerate(lam_conf_vals2)}
red_to_idx2  = {v: i for i, v in enumerate(lam_red_vals2)}

adv_grid = np.full((len(lam_red_vals2), len(lam_conf_vals2)), np.nan)

for _, row in qubo_cfg_adv.iterrows():
    i = red_to_idx2[row["lam_red"]]
    j = conf_to_idx2[row["lam_conf"]]
    adv_grid[i, j] = row["delta_f1"]

plt.figure(figsize=(6, 4.5))

vmax2 = np.nanmax(np.abs(adv_grid))
if np.isnan(vmax2) or vmax2 == 0:
    vmax2 = 1e-4

im2 = plt.imshow(
    adv_grid,
    origin="lower",
    aspect="auto",
    cmap="coolwarm",
    vmin=-vmax2,
    vmax=vmax2,
)

plt.xticks(range(len(lam_conf_vals2)), lam_conf_vals2)
plt.yticks(range(len(lam_red_vals2)), lam_red_vals2)

plt.xlabel(r"$\lambda_{conf}$")
plt.ylabel(r"$\lambda_{red}$")
plt.title(r"QUBO Advantage Map: $\Delta$ Macro-F1 vs DistMatch")

cbar2 = plt.colorbar(im2)
cbar2.set_label(r"$\Delta$ Macro-F1 (QUBO - DistMatch)")

plt.tight_layout()
plt.savefig(heatmap_adv_path, format="svg", bbox_inches="tight")
plt.close()
print("Saved QUBO Advantage Map heatmap to:", heatmap_adv_path)
