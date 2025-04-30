# 📊 Telco Customer Churn Prediction
## 🔍 Problem Statement
Telecom companies lose millions due to customer churn. This project aims to predict whether a customer is likely to churn (i.e., stop using the service), based on usage patterns, demographics, service subscriptions, and support history. Accurate churn prediction helps businesses implement retention strategies and reduce revenue loss.
## 🎯 Objective
- Build a predictive model to classify customers as likely to churn or not.
- Improve model accuracy and fairness using data augmentation techniques.
- Analyze key features driving churn and visualize model performance.
## 📁 Dataset Description
- **Source**: IBM Telco Customer Churn Dataset (Kaggle)
- **Target column**: `Churn` (Yes/No)
- **Records**: ~7,000
- **Features**: Customer demographics, services (Internet, Phone), contract type, billing, and tenure.
## 📌 Data Augmentation
The original dataset suffers from class imbalance (many more non-churners than churners), leading to biased predictions.
### ✅ Techniques Used:
- **2x Augmentation**: Duplicated churned records twice.
- **3x Augmentation**: Duplicated churned records thrice.
These approaches help the model learn churn patterns better and balance class distribution.
## 🛠️ Tech Stack
- Python (pandas, numpy, seaborn, matplotlib)
- scikit-learn (Random Forest, train_test_split, metrics)
- Jupyter/VS Code
- Git & GitHub
## ⚙️ Implementation Steps
1. Load and preprocess dataset (handle missing values, encode categorical data)
2. Apply augmentation (Original, 2x, or 3x versions)
3. Train/test split
4. Fit a **Random Forest Classifier**
5. Evaluate using:
   - Accuracy
   - Precision, Recall, F1-Score
   - Confusion Matrix
   - ROC-AUC Curve
   - Feature Importance
## 📈 Results Comparison

| **Metric**        | Original Data | 2x Augmentation | 3x Augmentation |
|------------------|---------------|------------------|------------------|
| Accuracy          | 0.79          | 0.83             | 0.86             |
| Precision         | 0.71          | 0.78             | 0.83             |
| Recall            | 0.62          | 0.81             | 0.87             |
| F1 Score          | 0.66          | 0.79             | 0.85             |
| ROC-AUC           | 0.84          | 0.89             | 0.91             |

> 📌 **Conclusion**: Augmenting the minority class greatly improved the model’s ability to detect churned customers, especially recall and F1-score.
## 📊 Key Visualizations
- Confusion Matrix
- ROC Curve
- Top 15 Feature Importances
