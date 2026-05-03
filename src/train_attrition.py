import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("data/employee.csv")

# Drop useless columns
df.drop(['EmployeeCount','Over18','StandardHours'], axis=1, inplace=True)

# Split features/target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Column types
num_cols = X.select_dtypes(include=['int64','float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Pipeline
model = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier())
])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model.fit(X_train, y_train)

# Save
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/attrition_model.pkl")

print("✅ Pipeline model saved!")