# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 1. Load the Iris Dataset
# The Iris dataset contains 150 samples, with 4 features and 3 classes of flowers
iris = load_iris()
X = iris.data  # Features: Sepal length, Sepal width, Petal length, Petal width
y = iris.target  # Target: Species (Setosa, Versicolor, Virginica)

# Convert to DataFrame for better visualization and handling
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['species'] = iris.target_names[y]

print("First few rows of the dataset:")
print(df.head())

# 2. Standardize the Data
# PCA is sensitive to the scale of the features, so we need to standardize the data.
# StandardScaler will normalize the features to have mean 0 and standard deviation 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA for Dimensionality Reduction
# Reduce the dimensionality to 2 components for easy visualization.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a new DataFrame containing the two principal components and the species label
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['species'] = iris.target_names[y]

print("\nFirst few rows after PCA transformation:")
print(df_pca.head())

# 4. Explained Variance Ratio
# Check how much variance each principal component captures from the original data.

# This tells us how much variance each principal component captures from the original data.
explained_variance_ratio = pca.explained_variance_ratio_
print("\nExplained variance by each principal component:", explained_variance_ratio)
print("Total variance explained by the two components:", sum(explained_variance_ratio))

# 5. Visualize the PCA Result
# We will plot the two principal components (PC1 and PC2) to see how well PCA separated the classes.
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
species = ['setosa', 'versicolor', 'virginica']

# Plot each class with different colors
for i, species_name in enumerate(species):
    subset = df_pca[df_pca['species'] == species_name]
    plt.scatter(subset['PC1'], subset['PC2'], label=species_name, c=colors[i], s=50)

# Add labels and legend
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of the Iris Dataset')
plt.legend()
plt.grid(True)
plt.show()

# 6. Cumulative Explained Variance
# Cumulative explained variance tells us how much of the original variance we retain as we keep adding more components.
pca_full = PCA(n_components=X.shape[1])  # Keep all components to see the cumulative effect
X_pca_full = pca_full.fit(X_scaled)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
print("\nCumulative explained variance for all principal components:")
print(cumulative_variance)

# Plot the cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, X.shape[1] + 1), cumulative_variance, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance as a Function of the Number of Components')
plt.grid(True)
plt.show()