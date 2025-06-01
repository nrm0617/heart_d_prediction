# model_training.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('dataset.csv')

# Display basic info
print("First 5 rows of dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# Correlation heatmap
corrmat = df.corr(numeric_only=True)
plt.figure(figsize=(16,16))
sns.heatmap(corrmat, annot=True, cmap="RdYlGn")
plt.title('Correlation Heatmap')
plt.show()

# Count plot of target variable
sns.set_style('whitegrid')
sns.countplot(x='target', data=df, palette='RdBu_r')
plt.title('Target Variable Distribution')
plt.show()

# One-hot encoding of categorical features
dataset = pd.get_dummies(df, columns=[
    'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'
])

# Separate features and target
X = dataset.drop('target', axis=1)
y = dataset['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

# KNN: Cross-validation for different K values
knn_scores = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=10)
    knn_scores.append(score.mean())

# Plot KNN accuracy scores
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), knn_scores, color='red', marker='o')
for i in range(1, 21):
    plt.text(i, knn_scores[i - 1], f"{knn_scores[i - 1]:.2f}", ha='center', va='bottom')
plt.xticks(range(1, 21))
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Cross-Validation Score')
plt.title('K-Nearest Neighbors Accuracy for Different K Values')
plt.grid(True)
plt.show()

# Best KNN model evaluation (example: K=12)
best_k = 12
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_score = cross_val_score(knn_best, X, y, cv=10).mean()
print(f"\nKNN Classifier (k={best_k}) Mean CV Accuracy: {knn_score:.4f}")

# Random Forest Classifier evaluation
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_score = cross_val_score(rf_classifier, X, y, cv=10).mean()
print(f"Random Forest Classifier Mean CV Accuracy: {rf_score:.4f}")

