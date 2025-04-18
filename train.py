import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import joblib

# Load dataset
df = pd.read_csv("student_depression_dataset.csv")

# Drop irrelevant columns
df.drop(columns=["id", "City", "Profession"], inplace=True)

# Clean 'Sleep Duration'
df["Sleep Duration"] = df["Sleep Duration"].str.replace("'", "").str.extract(r'(Less than \d+|\d+-\d+)')
df["Sleep Duration"] = df["Sleep Duration"].replace({
    'Less than 5': 4,
    '5-6': 5.5,
    '7-8': 7.5,
    '9-10': 9.5
})

# Encode categorical variables
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop("Depression", axis=1)
y = df["Depression"]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "depression_model.pkl")

# Evaluate
y_pred = model.predict(X_test)
print("✅ Model saved as 'depression_model.pkl'")
print("✅ Scaler saved as 'scaler.pkl'")
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
