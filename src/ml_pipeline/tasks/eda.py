"""Exploratory Data Analysis and Visualization Generator."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_INPUT_CSV = Path("data/processed/heart_disease_cleaned.csv")
DEFAULT_REPORTS_DIR = Path("reports")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "figure.titlesize": 13})

NUMERICAL_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def generate_summary(df: pd.DataFrame, output_path: Path):
    summary = {
        "dataset_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "target_balance": df["target"].value_counts(normalize=True).round(4).to_dict(),
        "numerical_stats": df[NUMERICAL_COLS].describe().round(2).to_dict(),
        "correlation_with_target": df.corr(numeric_only=True)["target"].round(4).to_dict()
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved statistical summary: {output_path}")


def plot_class_balance(df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["target"].value_counts().sort_index()
    labels = ["No Disease (0)", "Disease Present (1)"]
    palette = ["#2b5c8f", "#d95f02"]

    bars = ax.bar(labels, counts, color=palette, width=0.5, edgecolor="black", alpha=0.85)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height} ({height/len(df):.1%})",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Patient Count")
    ax.set_title("Heart Disease Target Class Balance")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_numerical_distributions(df: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, col in enumerate(NUMERICAL_COLS):
        ax = axes[idx]
        sns.histplot(data=df, x=col, hue="target", kde=True, ax=ax,
                     palette=["#2b5c8f", "#d95f02"], alpha=0.6, element="step")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    fig.delaxes(axes[5])
    plt.suptitle("Numerical Features Distribution by Target", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_categorical_relationships(df: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, col in enumerate(CATEGORICAL_COLS):
        ax = axes[idx]
        cross_tab = pd.crosstab(df[col], df["target"], normalize="index") * 100
        cross_tab.plot(kind="bar", stacked=True, ax=ax, color=["#2b5c8f", "#d95f02"], alpha=0.85, edgecolor="black")
        ax.set_title(f"Prevalence by {col}")
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel(col)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(["No Disease", "Disease"], loc="upper right", fontsize=8)

    plt.suptitle("Heart Disease Proportion Across Categorical Features", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Feature Correlation Matrix (Pearson)", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_eda(input_csv: Path = DEFAULT_INPUT_CSV, reports_dir: Path = DEFAULT_REPORTS_DIR):
    df = pd.read_csv(input_csv)
    rep_dir = Path(reports_dir)
    fig_dir = rep_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    generate_summary(df, rep_dir / "eda_summary.json")
    plot_class_balance(df, fig_dir / "01_class_balance.png")
    plot_numerical_distributions(df, fig_dir / "02_numerical_distributions.png")
    plot_categorical_relationships(df, fig_dir / "03_categorical_relationships.png")
    plot_correlation_matrix(df, fig_dir / "04_correlation_matrix.png")
    print(f"Generated all EDA figures in: {fig_dir}")


if __name__ == "__main__":
    run_eda()
