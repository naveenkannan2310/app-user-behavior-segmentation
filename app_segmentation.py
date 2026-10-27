import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------
# STEP 1: LOAD THE DATASET
# -------------------------------------------------------------
# Note: Make sure your downloaded dataset CSV is in the same folder 
# and update 'user_behavior.csv' if your file has a different name.
df = pd.read_csv("user_behavior.csv")

print("--- Dataset Info ---")
print(df.info())
print(df.head())

# -------------------------------------------------------------
# STEP 2: DATA CLEANING & FEATURE SELECTION
# -------------------------------------------------------------
# Drop missing values if any
df = df.dropna()

# Select relevant numerical features for behavioral segmentation
# (Modify these column names if your dataset uses different names)
feature_cols = [
    "session_frequency",
    "session_duration",
    "engagement_score",
    "activity_metric",
]

X = df[feature_cols]

# -------------------------------------------------------------
# STEP 3: FEATURE SCALING (StandardScaler)
# -------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------------------
# STEP 4: FIND OPTIMAL CLUSTERS USING THE ELBOW METHOD
# -------------------------------------------------------------
inertia = []
K_range = range(1, 11)

for k in K_range:
  kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
  kmeans.fit(X_scaled)
  inertia.append(kmeans.inertia_)

# Plotting and saving the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker="o", linestyle="-")
plt.title("The Elbow Method to Find Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.savefig("elbow_curve.png")
plt.show()

# -------------------------------------------------------------
# STEP 5: TRAIN K-MEANS MODEL (K = 4 behavioral groups)
# -------------------------------------------------------------
optimal_k = 4  
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# -------------------------------------------------------------
# STEP 6: CLUSTER PROFILING & INTERPRETATION
# -------------------------------------------------------------
print("\n--- Average Feature Values per Cluster ---")
cluster_profile = df.groupby("Cluster")[feature_cols].mean()
print(cluster_profile)

# Map clusters to business behavior segments
cluster_mapping = {
    0: "High Engagement Users",
    1: "Moderate Engagement Users",
    2: "Low Engagement / At-Risk Users",
    3: "Occasional Users",
}
df["Segment_Name"] = df["Cluster"].map(cluster_mapping)

# Save segmented output to a new CSV file
df.to_csv("segmented_user_behavior.csv", index=False)
print("\nSegmented data successfully saved to 'segmented_user_behavior.csv'!")

# -------------------------------------------------------------
# STEP 7: DIMENSIONALITY REDUCTION (PCA) & VISUALIZATION
# -------------------------------------------------------------
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

df["PCA1"] = principal_components[:, 0]
df["PCA2"] = principal_components[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="PCA1",
    y="PCA2",
    hue="Segment_Name",
    data=df,
    palette="viridis",
    s=50,
    alpha=0.8,
)
plt.title("App User Behavior Segmentation (PCA Visualization)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="User Segments")
plt.grid(True)
plt.savefig("pca_clusters.png")
plt.show()