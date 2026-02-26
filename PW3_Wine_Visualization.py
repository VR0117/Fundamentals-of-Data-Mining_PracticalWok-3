import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# SETTINGS
# ============================================================

np.random.seed(42)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (11, 7)
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 13

# ============================================================
# LOAD DATASET
# ============================================================

file_name = "winequality-red.csv"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

if not os.path.exists(file_name):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, file_name)

df = pd.read_csv(file_name, sep=";")

print("\nDataset Loaded Successfully\n")
print(df.head())
print(df.info())

# ============================================================
# QUALITY MEANINGFUL LABELS
# ============================================================

def quality_label(q):
    if q <= 3:
        return "Very Poor"
    elif q == 4:
        return "Poor"
    elif q == 5:
        return "Average"
    elif q == 6:
        return "Good"
    elif q == 7:
        return "Very Good"
    else:
        return "Excellent"

df["quality_label"] = df["quality"].apply(quality_label)

quality_order = ["Very Poor", "Poor", "Average",
                 "Good", "Very Good", "Excellent"]

# ============================================================
# CREATE SAVE FOLDER
# ============================================================

save_folder = "saved_graphs"
os.makedirs(save_folder, exist_ok=True)

figures = []

def store_fig(name):
    fig = plt.gcf()
    figures.append((fig, name))
    plt.show()

# ============================================================
# UNIVARIATE ANALYSIS
# ============================================================

plt.figure()
sns.histplot(df["alcohol"], bins=40, kde=True, color="royalblue")
plt.title("Distribution of Alcohol")
plt.xlabel("Alcohol")
plt.ylabel("Frequency")
store_fig("01_alcohol_distribution.png")

plt.figure()
sns.countplot(
    x="quality_label",
    data=df,
    order=quality_order,
    hue="quality_label",
    legend=False,
    palette="viridis"
)

plt.xticks(rotation=30)
plt.title("Wine Quality Distribution (Meaningful Categories)")
plt.xlabel("Quality Category")
plt.ylabel("Count")
store_fig("02_quality_distribution.png")

# ============================================================
# BIVARIATE ANALYSIS
# ============================================================

plt.figure()
sns.scatterplot(
    x="alcohol",
    y="density",
    data=df,
    alpha=0.6
)

plt.title("Alcohol vs Density")
plt.xlabel("Alcohol")
plt.ylabel("Density")
store_fig("03_alcohol_vs_density.png")

plt.figure()
sns.boxplot(
    x="quality_label",
    y="alcohol",
    data=df,
    order=quality_order,
    hue="quality_label",
    legend=False,
    palette="Set2"
)

plt.xticks(rotation=30)
plt.title("Alcohol vs Quality (Boxplot)")
plt.xlabel("Quality Category")
plt.ylabel("Alcohol")
store_fig("04_alcohol_vs_quality_boxplot.png")

# ============================================================
# CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(14, 10))

corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")
store_fig("05_correlation_heatmap.png")

# ============================================================
# PARALLEL COORDINATES
# ============================================================

features = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "alcohol",
    "quality"
]

df_parallel = df[features].copy()
df_parallel["quality"] = df_parallel["quality"].astype(str)

plt.figure(figsize=(14, 8))
sample_df = df_parallel.sample(400, random_state=42)

parallel_coordinates(sample_df, "quality")

plt.title("Parallel Coordinates Plot")
plt.xticks(rotation=45)
store_fig("06_parallel_coordinates.png")

# ================
# 3D VISUALIZATION
# ================

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

unique_labels = df["quality_label"].unique()
colors = sns.color_palette("tab10", len(unique_labels))

for label, color in zip(unique_labels, colors):
    subset = df[df["quality_label"] == label]

    ax.scatter(
        subset["alcohol"],
        subset["density"],
        subset["pH"],
        label=label,
        color=color,
        s=70,
        alpha=0.8
    )

ax.set_xlabel("Alcohol")
ax.set_ylabel("Density")
ax.set_zlabel("pH")
ax.set_title("3D Scatter Plot (Quality Categories)")

ax.legend(title="Quality Category")

plt.tight_layout()
plt.show()

figures.append((fig, "07_3d_scatter.png"))

# ============================================================
# ALCOHOL DISTRIBUTION ACROSS QUALITY
# ============================================================

plt.figure()

sns.histplot(
    data=df,
    x="alcohol",
    hue="quality_label",
    multiple="stack",
    bins=40,
    palette="tab10"
)

plt.title("Alcohol Distribution Across Quality Categories")
plt.xlabel("Alcohol")
plt.ylabel("Count")

store_fig("08_alcohol_distribution_quality.png")

# ============================================================
# SAVE ALL GRAPHS SILENTLY
# ============================================================

for fig, name in figures:
    path = os.path.join(save_folder, name)
    fig.savefig(path, dpi=300)
    plt.close(fig)

# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL REPORT")
print("=" * 70)

print("""
✔ All plots generated successfully.
✔ Quality converted into meaningful categories.
✔ 3D plot improved with category-based coloring.
✔ Graphs saved inside 'saved_graphs' folder.

Quality Meaning:
- Very Poor  -> Score ≤ 3
- Poor       -> Score 4
- Average    -> Score 5
- Good       -> Score 6
- Very Good  -> Score 7
- Excellent  -> Score 8

Key Insights:
1. Alcohol negatively correlates with density.
2. Higher alcohol wines mostly fall into Good / Very Good.
3. Most wines belong to Average and Good categories.
""")

print("=" * 70)