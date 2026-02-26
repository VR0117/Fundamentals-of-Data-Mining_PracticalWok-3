import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D

# =============================
# SETTINGS
# =============================

np.random.seed(42)
sns.set_theme(style="whitegrid")

plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 13

# =============================
# LOAD DATASET
# =============================

file_name = "winequality-red.csv"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

if not os.path.exists(file_name):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, file_name)

df = pd.read_csv(file_name, sep=";")

print("\nDataset Loaded Successfully\n")
print(df.head())
print(df.info())

# =============================
# SAVE FOLDER
# =============================

save_folder = "saved_graphs"
os.makedirs(save_folder, exist_ok=True)

# =============================
# QUALITY LABELS
# =============================

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

# =============================
# FUNCTION TO DISPLAY + SAVE
# =============================

def display_and_save(fig, filename):
    fig.tight_layout()
    path = os.path.join(save_folder, filename)
    fig.savefig(path, dpi=300)
    plt.show(block=True)
    plt.close(fig)

# =====================================================
# 1. UNIVARIATE ANALYSIS
# =====================================================

# Alcohol Distribution
fig1 = plt.figure()
sns.histplot(df["alcohol"], bins=40, kde=True)
plt.title("Univariate Analysis - Alcohol Distribution")
plt.xlabel("Alcohol")
plt.ylabel("Frequency")
display_and_save(fig1, "01_alcohol_distribution.png")

# Quality Distribution
fig2 = plt.figure()
sns.countplot(
    x="quality_label",
    data=df,
    order=quality_order,
    hue="quality_label",
    legend=False
)
plt.xticks(rotation=30)
plt.title("Univariate Analysis - Wine Quality Distribution")
plt.xlabel("Quality Category")
plt.ylabel("Count")
display_and_save(fig2, "02_quality_distribution.png")

# =====================================================
# 2. BIVARIATE ANALYSIS
# =====================================================

# Alcohol vs Density
fig3 = plt.figure()
sns.scatterplot(x="alcohol", y="density", data=df)
plt.title("Bivariate Analysis - Alcohol vs Density")
plt.xlabel("Alcohol")
plt.ylabel("Density")
display_and_save(fig3, "03_alcohol_vs_density.png")

# Boxplot
fig4 = plt.figure()
sns.boxplot(
    x="quality_label",
    y="alcohol",
    data=df,
    order=quality_order,
    hue="quality_label",
    legend=False
)
plt.xticks(rotation=30)
plt.title("Alcohol Distribution Across Quality (Boxplot)")
plt.xlabel("Quality")
plt.ylabel("Alcohol")
display_and_save(fig4, "04_alcohol_boxplot.png")

# Violin Plot
fig5 = plt.figure()
sns.violinplot(
    x="quality_label",
    y="alcohol",
    data=df,
    order=quality_order,
    hue="quality_label",
    legend=False
)
plt.xticks(rotation=30)
plt.title("Alcohol Distribution Across Quality (Violin Plot)")
plt.xlabel("Quality")
plt.ylabel("Alcohol")
display_and_save(fig5, "05_alcohol_violin.png")

# Joint Plot
joint = sns.jointplot(
    x="alcohol",
    y="quality",
    data=df,
    kind="scatter",
    height=8
)

joint.fig.suptitle("Joint Plot - Alcohol vs Quality", y=1.02)
joint.savefig(os.path.join(save_folder, "06_joint_plot_alcohol_quality.png"))
plt.show(block=True)
plt.close(joint.fig)

print("Joint plot saved.\n")

# =====================================================
# 3. MULTIVARIATE ANALYSIS
# =====================================================

# Parallel Coordinates
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

scaler = MinMaxScaler()

numeric_cols = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "alcohol"
]

df_parallel[numeric_cols] = scaler.fit_transform(df_parallel[numeric_cols])

sample_df = df_parallel.sample(500, random_state=42)

fig6 = plt.figure(figsize=(16, 9))
parallel_coordinates(sample_df, "quality", colormap=plt.get_cmap("tab10"))
plt.title("Multivariate Analysis - Parallel Coordinates")
plt.xticks(rotation=45)
plt.legend(title="Quality")
display_and_save(fig6, "07_parallel_coordinates.png")

# 3D Scatter Plot
fig7 = plt.figure(figsize=(14, 10))
ax = fig7.add_subplot(111, projection="3d")

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
ax.set_title("Multivariate Analysis - 3D Scatter (Alcohol, Density, pH)")
ax.legend(title="Quality Category")

display_and_save(fig7, "08_3d_scatter.png")

# =====================================================
# 🔥 NEW: HEATMAP (Correlation Matrix)
# =====================================================

fig8 = plt.figure(figsize=(14, 10))

corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Wine Features")

display_and_save(fig8, "09_correlation_heatmap.png")

# =====================================================
# Alcohol Distribution Across Quality
# =====================================================

fig9 = plt.figure()

sns.histplot(
    data=df,
    x="alcohol",
    hue="quality_label",
    multiple="stack",
    bins=40
)

plt.title("Alcohol Distribution Across Quality Levels")
plt.xlabel("Alcohol")
plt.ylabel("Count")

display_and_save(fig9, "10_alcohol_distribution_quality.png")

# =====================================================
# FINAL OUTPUT
# =====================================================

print("\n====================================")
print("FINAL REPORT")
print("====================================")

print("""
✔ All required graphs are included:
  - Univariate (Continuous + Categorical)
  - Bivariate (Scatter + Boxplot + Violin + Joint Plot)
  - Multivariate (Parallel Coordinates + 3D Scatter)
  - Heatmap (Correlation Matrix)
  - Alcohol Distribution Across Quality

✔ Graphs saved inside 'saved_graphs' folder.

Project Status: COMPLETE ✅
""")

print("====================================")