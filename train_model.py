import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
df = pd.read_csv('dataset.csv')

# One-hot encode categorical variables
df = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Scale continuous features
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

