# Import necessary libraries
import sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from scipy import stats

#load the dataset
california_housing = fetch_california_housing(as_frame=True)

df = california_housing.frame

# Step 1: Data standardization
scaler = StandardScaler()
df_standard = pd.DataFrame(scaler.fit_transform(df.drop('MedHouseVal', axis=1)), columns=df.columns[:-1])
df_standard['MedHouseVal'] = df['MedHouseVal'] # Add the target column back without scaling
df_standard

# Step 2: Look for outliers using 2D visualization (for a couple of features)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_standard)
plt.show()

# Step 3: Look for outliers using 3D visualization (for three features)

fig = px.scatter_3d(
    df_standard,
    x='MedInc',
    y='AveRooms',
    z='Population',
    color='MedHouseVal'
)

fig.show()

# Step 4: Remove outliers
# We will use Z-score to identify and remove outliers
df_clean = df_standard[(np.abs(stats.zscore(df_standard)) < 3).all(axis=1)]

df_clean

# Step 5: Check for missing values
missing_values = df_clean.isnull().sum()
# Print the missing values (if any)
print(missing_values)

# Step 6: Handle missing values (if any)
# For this dataset, there are no missing values usually, but if there were, you could handle them like this:
df_clean = df_clean.fillna(df_clean.mean()) # to replace with mean
df_clean.dropna(inplace=True) # to remove rows with missing values

# Step 7: Check for any categorical values
# This dataset does not contain categorical features

# Step 8: Perform dimensionality reduction
# Keep 95% of variance
pca = PCA(n_components=0.95)
df_reduced = pca.fit_transform(df_clean)
print("Reduced dataset shape:", df_reduced.shape)
pca.explained_variance_ratio_

df_clean['dimensionality_reduced'] = list(df_reduced)

df_clean
