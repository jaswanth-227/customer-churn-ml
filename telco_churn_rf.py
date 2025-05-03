# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                           accuracy_score, roc_auc_score, roc_curve, precision_recall_curve)

#  Load Dataset
df = pd.read_csv('/content/WA_Fn-UseC_-Telco-Customer-Churn.csv')  # Replace with your path

# Data Preprocessing
# Drop irrelevant columns
df.drop('customerID', axis=1, inplace=True)

# Handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Encode target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical features
cat_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Scale numerical columns
scaler = StandardScaler()
scaled_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

# Split data
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    # Store results
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    # Print basic metrics
    print(f"\n{name} Performance:")
    print(f"Accuracy: {results[name]['accuracy']:.4f}")
    print(f"ROC-AUC: {results[name]['roc_auc']:.4f}")
    print("Classification Report:")
    print(results[name]['report'])

# Comparative Analysis
# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results],
    'ROC-AUC': [results[m]['roc_auc'] for m in results]
}).sort_values('Accuracy', ascending=False)

print("\nModel Comparison:")
print(comparison_df)

# Visualization
plt.figure(figsize=(12, 6))

# Accuracy comparison
plt.subplot(1, 2, 1)
sns.barplot(x='Accuracy', y='Model', data=comparison_df, palette='viridis')
plt.title('Model Accuracy Comparison')
plt.xlim(0.7, 0.85)

# ROC-AUC comparison
plt.subplot(1, 2, 2)
sns.barplot(x='ROC-AUC', y='Model', data=comparison_df, palette='viridis')
plt.title('Model ROC-AUC Comparison')
plt.xlim(0.7, 0.9)

plt.tight_layout()
plt.show()

# ROC Curves for all models
plt.figure(figsize=(8, 6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Feature Importance (for tree-based models)
tree_models = {
    "Decision Tree": models["Decision Tree"],
    "Random Forest": models["Random Forest"]
}

for name, model in tree_models.items():
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feat_df)
        plt.title(f'Top 15 Important Features - {name}')
        plt.tight_layout()
        plt.show()
