#!/usr/bin/env python
"""
Critic score distributions for high vs low reliability examples.

- De-duplicate by `text2015`
- Split into low vs high reliability using reliability_prob
- Plot violin + boxplot of critic_score_sum
- Save as SVG and high-res PNG
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------
# SVG + text selectability settings
# ---------------------------------
mpl.rcParams["svg.fonttype"] = "none"         # keep text as text (selectable)
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]  # or another clean font
mpl.rcParams["font.size"] = 11               # base font size
mpl.rcParams["axes.unicode_minus"] = False   # avoid minus-sign issues

# -----------------------------
# 0. Paths & basic config
# -----------------------------

WEAK_PATH = "weak_labels_with_temp_data_discriminator_soft.jsonl"
OUT_DIR = "latex/figures"
os.makedirs(OUT_DIR, exist_ok=True)

RELI_THRESHOLD = 0.5   # r >= 0.5 -> "high", else "low"
TEXT_COL = "text2015"  # column to use for de-duplication


# -----------------------------
# 1. Load + de-duplicate
# -----------------------------

def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return pd.DataFrame(rows)


print(f"Loading weak labels from: {WEAK_PATH}")
df = load_jsonl(WEAK_PATH)
print(f"Loaded {len(df):,} rows before de-duplication")

if TEXT_COL in df.columns:
    df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)
    print(f"After de-duplication on '{TEXT_COL}': {len(df):,} rows\n")
else:
    print(f"WARNING: column '{TEXT_COL}' not found, skipping de-duplication.\n")

# -----------------------------
# 2. Define reliability groups
# -----------------------------

if "reliability_prob" not in df.columns:
    raise ValueError("Column 'reliability_prob' not found in DataFrame.")

df["reli_group"] = np.where(
    df["reliability_prob"] >= RELI_THRESHOLD,
    "high",
    "low",
)

group_counts = df["reli_group"].value_counts()
print("Reliability groups (counts):")
print(group_counts, "\n")

# -----------------------------
# 3. Prepare data for plotting
# -----------------------------

if "critic_score_sum" not in df.columns:
    raise ValueError("Column 'critic_score_sum' not found in DataFrame.")

order = ["low", "high"]
labels = ["Low reliability", "High reliability"]

data = []
for g in order:
    scores = df.loc[df["reli_group"] == g, "critic_score_sum"].dropna()
    data.append(scores.values)
    print(f"{g}: n={len(scores)}, mean={scores.mean():.2f}, "
          f"min={scores.min() if len(scores) else 'NA'}, "
          f"max={scores.max() if len(scores) else 'NA'}")

print()

# -----------------------------
# 4. Violin plot
# -----------------------------

fig, ax = plt.subplots(figsize=(4.5, 3.5))

vp = ax.violinplot(
    data,
    showmeans=True,
    showextrema=True,
    showmedians=False,
)

ax.set_xticks([1, 2])
ax.set_xticklabels(labels, rotation=0)
ax.set_ylabel("Critic rubric score (0–8)")
ax.set_title("Critic score distribution by reliability group")

plt.tight_layout()

svg_path = os.path.join(OUT_DIR, "multiagent_critic_score_violin.svg")
png_path = os.path.join(OUT_DIR, "multiagent_critic_score_violin.png")

fig.savefig(svg_path, format="svg", bbox_inches="tight")
fig.savefig(png_path, format="png", dpi=400, bbox_inches="tight")
plt.close(fig)

print(f"Saved violin plot to:\n  {svg_path}\n  {png_path}\n")

# -----------------------------
# 5. Boxplot
# -----------------------------

fig, ax = plt.subplots(figsize=(4.5, 3.5))

bp = ax.boxplot(
    data,
    labels=labels,
    showmeans=True,
)

ax.set_ylabel("Critic rubric score (0–8)")
ax.set_title("Critic score (boxplot) by reliability group")

plt.tight_layout()

svg_path = os.path.join(OUT_DIR, "multiagent_critic_score_boxplot.svg")
png_path = os.path.join(OUT_DIR, "multiagent_critic_score_boxplot.png")

fig.savefig(svg_path, format="svg", bbox_inches="tight")
fig.savefig(png_path, format="png", dpi=400, bbox_inches="tight")
plt.close(fig)

print(f"Saved boxplot to:\n  {svg_path}\n  {png_path}")
