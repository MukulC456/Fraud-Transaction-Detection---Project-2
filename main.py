import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Sample Data 
data = {
    "TRANSACTION_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "TX_DATETIME": [
        "2025-02-01 12:34:00", "2025-02-02 15:22:00", "2025-02-03 09:10:00",
        "2025-02-04 18:45:00", "2025-02-05 22:30:00", "2025-02-06 08:05:00",
        "2025-02-07 14:10:00", "2025-02-08 19:15:00", "2025-02-09 11:50:00", "2025-02-10 07:30:00"
    ],
    "CUSTOMER_ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "TERMINAL_ID": [201, 202, 203, 204, 205, 206, 207, 208, 209, 210],
    "TX_AMOUNT": [50, 500, 200, 30, 700, 100, 220, 300, 20, 450],
    "TX_FRAUD": [0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
}

# Converting to DataFrame
df = pd.DataFrame(data)

# Converting TX_DATETIME to datetime format
df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])

# Feature Engineering
df['hour'] = df['TX_DATETIME'].dt.hour
df['day'] = df['TX_DATETIME'].dt.day
df['month'] = df['TX_DATETIME'].dt.month

# Tracking Fraud Patterns (Fraud per terminal and customer)
fraud_per_terminal = df.groupby("TERMINAL_ID")["TX_FRAUD"].sum().reset_index()
fraud_per_terminal.rename(columns={"TX_FRAUD": "fraud_count_terminal"}, inplace=True)
df = df.merge(fraud_per_terminal, on="TERMINAL_ID", how="left")

fraud_per_customer = df.groupby("CUSTOMER_ID")["TX_FRAUD"].sum().reset_index()
fraud_per_customer.rename(columns={"TX_FRAUD": "fraud_count_customer"}, inplace=True)
df = df.merge(fraud_per_customer, on="CUSTOMER_ID", how="left")

# Replace NaN with 0
df.fillna(0, inplace=True)

# Creating Fraud Indicator for Amount > 220
df['suspicious_amount'] = (df['TX_AMOUNT'] > 220).astype(int)

# Select Features & Target Variable
features = ['TX_AMOUNT', 'hour', 'fraud_count_terminal', 'fraud_count_customer', 'suspicious_amount']
X = df[features]
y = df['TX_FRAUD']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle Class Imbalance using SMOTE (Fixed k_neighbors issue)
smote = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Training ML Model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train_resampled)

# Making Predictions
y_pred = model.predict(X_test_scaled)

# Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC-AUC Score
roc_score = roc_auc_score(y_test, y_pred)
print("ROC AUC Score:", roc_score)
