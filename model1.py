import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Used LLM for syntax and use of libraries

# Load data sets
# df = pd.read_csv("original.csv")
df = pd.read_csv("cleaned.csv")

# Choose features if eda bad
# features = [
#     "Communication_Skills",
#     "CGPA",
#     "Prev_Sem_Result",
#     "IQ",
#     "Projects_Completed",
#     "Extra_Curricular_Score",
#     "Internship_Experience",
#     "Academic_Performance"
# ]

# Placement is the label
X = df.drop(columns=["Placement"])
y = df["Placement"]

# Split data into test and train, 20% train, 80% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=64)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train on a Random Forest classifier
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Test the model
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Some analysis suggested by LLM
importances = model.feature_importances_
features = X.columns

feat_importances = pd.Series(importances, index=features).sort_values(ascending=False)
print(feat_importances)

feat_importances.plot(kind="bar")
plt.show()