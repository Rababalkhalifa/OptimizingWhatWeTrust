#!/usr/bin/env python
"""
Analyse multi-agent weak labels + reliability discriminator.

- Sanity checks on ranges (confidence, critic scores, reliability).
- Overall label distribution.
- Agreement vs disagreement behaviour.
- Reliability buckets (high / medium / low).
- Per-frame reliability and confidence.
- Simple correlations between:
    * final_confidence and reliability_prob
    * critic_score_sum and reliability_prob

OPTION: change WEAK_PATH to match your local file path.
"""

import json
import math
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 0. Paths & basic config
# -----------------------------

# <<< CHANGE THIS IF NEEDED >>>
WEAK_PATH = "steps_datasets/weak_labels_with_temp_data_discriminator_soft.jsonl"

OUT_DIR = "latex/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Reliability buckets (you can tweak these if you like)
RELI_BINS = [0.0, 0.33, 0.66, 1.01]
RELI_LABELS = ["low", "medium", "high"]


# -----------------------------
# 1. Load JSONL into DataFrame
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
print(f"Loaded {len(df):,} rows\n")

# -----------------------------
# 2. Basic sanity checks
# -----------------------------

# Helper: extract a clean frame string from final_label like '["Rights/Justice"]'
def normalise_final_label(s):
    if not isinstance(s, str):
        return s
    # Very simple clean-up: strip brackets/quotes if present
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    return s

df["frame"] = df["final_label"].apply(normalise_final_label)

# Sanity ranges
def check_range(column, lo, hi):
    vals = df[column].dropna()
    n = len(vals)
    bad_lo = (vals < lo).sum()
    bad_hi = (vals > hi).sum()
    print(f"[CHECK] {column}: n={n}, min={vals.min():.3f}, max={vals.max():.3f}")
    if bad_lo or bad_hi:
        print(f"       WARNING: {bad_lo} values < {lo}, {bad_hi} values > {hi}")
    else:
        print("       OK: all values in expected range.\n")

check_range("final_confidence", 0.0, 1.0)
check_range("reliability_prob", 0.0, 1.0)
check_range("critic_score_sum", 0.0, 8.0)

# Missing flags
for col in ["labeler_agree", "reliable_flag", "low_confidence"]:
    missing = df[col].isna().sum()
    print(f"[CHECK] {col}: {missing} missing values")

print()

# -----------------------------
# 3. Overall label distribution
# -----------------------------

frame_counts = df["frame"].value_counts().sort_values(ascending=False)
print("Frame distribution (top 20):")
print(frame_counts.head(20))
print()

# Save as CSV for later use in tables
frame_counts.to_csv(os.path.join(OUT_DIR, "multiagent_frame_counts.csv"),
                    header=["count"])

# -----------------------------
# 4. Agreement vs disagreement
# -----------------------------

agree_mask = df["labeler_agree"] == True
disagree_mask = df["labeler_agree"] == False

n_agree = agree_mask.sum()
n_disagree = disagree_mask.sum()

print(f"Labeler agreement rate: {n_agree}/{len(df)} = {n_agree/len(df):.3f}")
print(f"Labeler disagreement rate: {n_disagree}/{len(df)} = {n_disagree/len(df):.3f}")

def summary_for_mask(name, mask):
    sub = df[mask]
    if len(sub) == 0:
        print(f"\n[{name}] no rows")
        return
    print(f"\n[{name}] n={len(sub)}")
    print(f"  mean final_confidence = {sub['final_confidence'].mean():.3f}")
    print(f"  mean critic_score_sum = {sub['critic_score_sum'].mean():.3f}")
    print(f"  mean reliability_prob  = {sub['reliability_prob'].mean():.33f}")
    print(f"  high reliability (r>=0.8): {(sub['reliability_prob']>=0.8).mean():.3f}")
    print(f"  low reliability (r<=0.2): {(sub['reliability_prob']<=0.2).mean():.3f}")

summary_for_mask("AGREE", agree_mask)
summary_for_mask("DISAGREE", disagree_mask)

# -----------------------------
# 5. Reliability buckets
# -----------------------------

df["reli_bucket"] = pd.cut(
    df["reliability_prob"],
    bins=RELI_BINS,
    labels=RELI_LABELS,
    include_lowest=True
)

print("\nReliability buckets:")
print(df["reli_bucket"].value_counts().sort_index())
print()

bucket_stats = (
    df.groupby("reli_bucket")
      .agg(
          n=("reliability_prob", "size"),
          mean_r=("reliability_prob", "mean"),
          mean_conf=("final_confidence", "mean"),
          mean_critic=("critic_score_sum", "mean"),
          agree_rate=("labeler_agree", "mean"),
      )
      .reset_index()
)

print("Bucket-level stats:")
print(bucket_stats.to_string(index=False))
print()

bucket_stats.to_csv(os.path.join(OUT_DIR, "multiagent_bucket_stats.csv"),
                    index=False)

# -----------------------------
# 6. Per-frame reliability
# -----------------------------

frame_stats = (
    df.groupby("frame")
      .agg(
          n=("frame", "size"),
          mean_r=("reliability_prob", "mean"),
          median_r=("reliability_prob", "median"),
          mean_conf=("final_confidence", "mean"),
          mean_critic=("critic_score_sum", "mean"),
          agree_rate=("labeler_agree", "mean"),
      )
      .sort_values("n", ascending=False)
)

print("Per-frame stats (head):")
print(frame_stats.head(10))
print()

frame_stats.to_csv(os.path.join(OUT_DIR, "multiagent_frame_stats.csv"))

# -----------------------------
# 7. Simple correlations
# -----------------------------

def corr(a, b):
    a = df[a].astype(float)
    b = df[b].astype(float)
    mask = a.notna() & b.notna()
    if mask.sum() < 2:
        return np.nan
    return np.corrcoef(a[mask], b[mask])[0, 1]

rho_conf_r = corr("final_confidence", "reliability_prob")
rho_critic_r = corr("critic_score_sum", "reliability_prob")
rho_conf_critic = corr("final_confidence", "critic_score_sum")

print(f"corr(final_confidence, reliability_prob) = {rho_conf_r:.3f}")
print(f"corr(critic_score_sum, reliability_prob)  = {rho_critic_r:.3f}")
print(f"corr(final_confidence, critic_score_sum)  = {rho_conf_critic:.3f}\n")

# -----------------------------
# 8. Optional: quick plots
# -----------------------------

# Reliability histogram
plt.figure(figsize=(4, 3))
plt.hist(df["reliability_prob"].dropna(), bins=20)
plt.xlabel("reliability_prob")
plt.ylabel("Count")
plt.title("Distribution of reliability_prob")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "multiagent_reliability_hist.png"), dpi=300)

# Conf vs reliability scatter (subsample for readability)
sample = df.sample(min(len(df), 2000), random_state=0)

plt.figure(figsize=(4, 3))
plt.scatter(sample["final_confidence"], sample["reliability_prob"], alpha=0.4, s=10)
plt.xlabel("final_confidence")
plt.ylabel("reliability_prob")
plt.title("final_confidence vs reliability_prob")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "multiagent_conf_vs_reliability.png"), dpi=300)

print(f"Saved plots and CSVs to: {OUT_DIR}")
