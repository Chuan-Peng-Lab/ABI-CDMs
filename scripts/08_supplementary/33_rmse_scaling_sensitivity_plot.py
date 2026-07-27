#!/usr/bin/env python
# coding: utf-8
"""
Plot-only script for RMSE Scaling Sensitivity.
Loads precomputed summary CSV; no heavy computation.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Arial", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("white")

MODELS_MAIN = ["DSTP", "DMC", "SSP", "DDM"]
MODEL_COLORS = {
    "DSTP": "#81cef0", "DMC": "#95d8c3",
    "SSP": "#92d28e", "DDM": "#b0d97c",
}
MODEL_EDGE = {
    "DSTP": "#4dc3eb", "DMC": "#75cfb6",
    "SSP": "#6fc96f", "DDM": "#9dd25b",
}

SUMMARY_PATH = "33_rmse_scaling_summary.csv"
SAVE_SVG = "../figs/S_rmse_scaling_sensitivity.svg"
SAVE_PNG = "../figs/S_rmse_scaling_sensitivity.png"


def strip_trailing_whitespace(path):
    p = Path(path)
    t = p.read_text(encoding="utf-8")
    p.write_text("\n".join(l.rstrip() for l in t.splitlines()) + "\n", encoding="utf-8")


def plot_sensitivity(summary_df):
    fig = plt.figure(figsize=(11, 9))

    # Panel A: Overall winner proportions across scaling factors
    ax_a = fig.add_subplot(2, 2, 1)
    overall = summary_df[summary_df["group_type"] == "overall"]
    x = overall["caf_scale_base"].unique()
    x_sorted = sorted(x)
    bottom = np.zeros(len(x_sorted))
    for m in MODELS_MAIN:
        vals = []
        for s in x_sorted:
            row = overall[(overall["caf_scale_base"] == s) & (overall["model"] == m)]
            vals.append(row["winner_prop"].values[0] * 100 if len(row) > 0 else 0)
        ax_a.bar(range(len(x_sorted)), vals, bottom=bottom, label=m,
                 color=MODEL_COLORS[m], edgecolor=MODEL_EDGE[m], linewidth=1.0, width=0.7)
        bottom += np.array(vals)
    ax_a.set_xticks(range(len(x_sorted)))
    ax_a.set_xticklabels([str(s) for s in x_sorted])
    ax_a.set_ylim(0, 100)
    ax_a.set_ylabel("Winner proportion (%)")
    ax_a.set_xlabel("CAF scale base")
    ax_a.set_title("A: Overall winner proportions", fontweight="bold", loc="left")
    ax_a.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.12))
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # Panel B: Agreement with default (500)
    ax_b = fig.add_subplot(2, 2, 2)
    agree = summary_df[summary_df["group_type"] == "overall"].drop_duplicates(
        ["caf_scale_base", "agreement_with_default"]
    )
    ax_b.plot(agree["caf_scale_base"], agree["agreement_with_default"] * 100, "o-", color="#2c7bb6", lw=2)
    ax_b.axhline(100, color="gray", ls="--", alpha=0.5)
    ax_b.set_ylabel("Agreement with default (%)")
    ax_b.set_xlabel("CAF scale base")
    ax_b.set_ylim(80, 105)
    ax_b.set_title("B: Agreement with default (s=500)", fontweight="bold", loc="left")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # Panel C: Task-level winners at selected factors
    ax_c = fig.add_subplot(2, 2, 3)
    selected_scales = [0, 500, 1000]
    tasks = sorted(summary_df[summary_df["group_type"] == "task"]["group"].unique())
    bar_width = 0.25
    x_pos = np.arange(len(tasks))
    for i, s in enumerate(selected_scales):
        task_data = summary_df[(summary_df["group_type"] == "task") & (summary_df["caf_scale_base"] == s)]
        dom = task_data.loc[task_data.groupby("group")["winner_prop"].idxmax()]
        props = []
        for t in tasks:
            r = dom[dom["group"] == t]
            props.append(r["winner_prop"].values[0] * 100 if len(r) > 0 else 0)
        ax_c.bar(x_pos + i * bar_width - bar_width, props, bar_width,
                 label=f"s={s}", color=[plt.cm.Blues(0.3 + i * 0.3)], edgecolor="black", linewidth=0.5)
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(tasks, fontsize=9)
    ax_c.set_ylabel("Dominant model win %")
    ax_c.set_title("C: Dominant model win% by task", fontweight="bold", loc="left")
    ax_c.legend(fontsize=8)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # Panel D: Dataset heatmap — agreement with default
    ax_d = fig.add_subplot(2, 2, 4)
    ds_data = summary_df[summary_df["group_type"] == "dataset"].drop_duplicates(
        ["caf_scale_base", "group", "agreement_with_default"]
    )
    ds_pivot = ds_data.pivot(index="group", columns="caf_scale_base", values="agreement_with_default")
    sns.heatmap(ds_pivot * 100, annot=True, fmt=".0f", cmap="RdYlGn", vmin=70, vmax=100,
                ax=ax_d, cbar_kws={"label": "Agreement (%)"}, linewidths=0.5)
    ax_d.set_title("D: Dataset agreement with default", fontweight="bold", loc="left")

    plt.tight_layout()
    plt.savefig(SAVE_SVG, bbox_inches="tight")
    strip_trailing_whitespace(SAVE_SVG)
    plt.savefig(SAVE_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {SAVE_SVG}")
    print(f"Saved: {SAVE_PNG}")


def main():
    summary_df = pd.read_csv(SUMMARY_PATH)
    print(f"Loaded summary: {summary_df.shape[0]} rows × {summary_df.shape[1]} cols")
    plot_sensitivity(summary_df)


if __name__ == "__main__":
    main()
