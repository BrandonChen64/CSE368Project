import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Used LLM for syntax and use of sklearn libraries

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

# Train on an A-NN
model = MLPClassifier(hidden_layer_sizes=(6, 6),
                      activation='relu',
                      solver='adam',
                      max_iter=1000,
                      random_state=64)
model.fit(X_train_scaled, y_train)

# Test the model
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)