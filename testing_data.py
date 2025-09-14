import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv("manila_testing_landsat_data.csv")

# Keep only numeric columns
X = df.select_dtypes(include=[np.number])

# Replace inf values with NaN, then fill with column mean
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute mean vector
mean_vec = X_scaled.mean(axis=0)

# Euclidean distance from mean vector
distances = np.linalg.norm(X_scaled - mean_vec, axis=1)

# Add new column to original dataframe
df["distance_from_boundary"] = distances

# Save new dataset
df.to_csv("manila_testing_with_boundary.csv", index=False)

print("✅ New dataset saved as manila_testing_with_boundary.csv")
