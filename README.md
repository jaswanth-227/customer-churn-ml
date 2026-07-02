# 📊 Telco Customer Churn Prediction using Machine Learning

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Poppins&size=28&duration=3500&pause=1000&color=00B8D4&center=true&vCenter=true&width=900&lines=Telco+Customer+Churn+Prediction;Machine+Learning+for+Customer+Retention;Random+Forest+%7C+Data+Augmentation;Predictive+Analytics+Project" />
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-black?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-blue?style=for-the-badge&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green?style=for-the-badge)

</p>

---

# 📖 Overview

Customer churn is one of the biggest challenges faced by telecommunication companies. Losing existing customers directly impacts revenue and increases customer acquisition costs.

This project develops a **Machine Learning-based churn prediction system** capable of identifying customers who are likely to discontinue their telecom services. To improve predictive performance on an imbalanced dataset, multiple **data augmentation strategies** are explored and compared.

---

# 🎯 Objectives

- Predict customer churn using supervised machine learning.
- Address class imbalance through data augmentation.
- Compare model performance across different augmentation techniques.
- Identify the most influential factors contributing to customer churn.
- Provide actionable insights for customer retention strategies.

---

# 📂 Dataset

**Dataset:** IBM Telco Customer Churn Dataset

### Dataset Summary

| Attribute | Details |
|------------|---------|
| Source | IBM Telco Customer Churn Dataset (Kaggle) |
| Total Records | ~7,000 |
| Target Variable | **Churn (Yes / No)** |
| Features | Customer demographics, subscriptions, billing, tenure, contract type, support services |

---

# 🔍 Problem Statement

Telecommunication companies lose millions annually due to customer attrition.

The objective of this project is to build a predictive model that identifies customers likely to churn, enabling businesses to implement proactive retention strategies and improve customer satisfaction.

---

# ⚙️ Methodology

The complete workflow consists of the following stages:

```text
Dataset
    │
    ▼
Data Preprocessing
    │
    ▼
Encoding & Cleaning
    │
    ▼
Data Augmentation
    │
    ▼
Train-Test Split
    │
    ▼
Random Forest Classifier
    │
    ▼
Performance Evaluation
    │
    ▼
Feature Importance Analysis
```

---

# 📈 Data Augmentation

The original dataset exhibits **class imbalance**, where non-churn customers significantly outnumber churn customers.

To improve model learning, two augmentation strategies were implemented:

### 🔹 Original Dataset

- No augmentation

### 🔹 2× Augmentation

- Minority (churn) samples duplicated twice.

### 🔹 3× Augmentation

- Minority (churn) samples duplicated three times.

These approaches help reduce prediction bias and improve the model's ability to detect churned customers.

---

# 🤖 Machine Learning Model

**Algorithm Used**

- 🌳 Random Forest Classifier

### Why Random Forest?

- Handles mixed numerical and categorical data
- Robust against overfitting
- Captures nonlinear relationships
- Provides feature importance
- Performs well on imbalanced datasets

---

# 📊 Performance Comparison

| Metric | Original | 2× Augmentation | 3× Augmentation |
|---------|---------:|---------------:|---------------:|
| Accuracy | **79%** | **83%** | **86%** |
| Precision | 71% | 78% | **83%** |
| Recall | 62% | 81% | **87%** |
| F1-Score | 66% | 79% | **85%** |
| ROC-AUC | 0.82 | 0.89 | **0.91** |

---

# 📉 Evaluation Metrics

The model is evaluated using:

- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1 Score
- ✅ ROC-AUC Score
- ✅ Confusion Matrix
- ✅ Feature Importance

---

# 📊 Results

The **3× Augmentation** model achieved the best overall performance:

- 🎯 Accuracy: **86%**
- 🎯 Precision: **83%**
- 🎯 Recall: **87%**
- 🎯 F1 Score: **85%**
- 🎯 ROC-AUC: **0.91**

The improvement demonstrates that balancing the dataset significantly enhances the model's ability to identify churned customers.

---

# 📌 Key Insights

The most influential features affecting customer churn include:

- 📅 Contract Type
- ⏳ Customer Tenure
- 💳 Monthly Charges
- 💰 Total Charges
- 🌐 Internet Service
- 🛡 Online Security
- 📞 Tech Support
- 📄 Paperless Billing
- 💵 Payment Method

---

# 📊 Visualizations

The project includes several visual analytics:

- 📈 Class Distribution
- 📉 Confusion Matrix
- 📊 ROC Curve
- 🌳 Feature Importance
- 📋 Correlation Heatmap
- 📊 Model Performance Comparison

---

# 🛠️ Technology Stack

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Development Environment

- Jupyter Notebook
- Visual Studio Code
- Git & GitHub

---

# 📂 Project Structure

```text
Telco-Customer-Churn-Prediction
│
├── dataset/
│
├── notebooks/
│
├── models/
│
├── images/
│
├── results/
│
├── requirements.txt
│
├── churn_prediction.ipynb
│
└── README.md
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Telco-Customer-Churn-Prediction.git
```

Navigate to the project

```bash
cd Telco-Customer-Churn-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

Launch Jupyter Notebook

```bash
jupyter notebook
```

or execute the Python script

```bash
python churn_prediction.py
```

---

# 🔮 Future Improvements

- 🚀 Hyperparameter Optimization
- 🤖 XGBoost & LightGBM Comparison
- 🧠 Deep Learning-based Churn Prediction
- ⚖️ SMOTE & ADASYN Oversampling
- 🌐 Streamlit Dashboard
- ☁️ Model Deployment using Flask/FastAPI

---

# 👨‍💻 Author

**Jaswanth Yadurla**

🎓 B.Tech – Artificial Intelligence & Machine Learning

📧 **Email:** yadurlajaswanth@gmail.com

🔗 **LinkedIn:** https://www.linkedin.com/in/jaswanth-yadurla-634290284/

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!

**Built with ❤️ using Machine Learning for Predictive Customer Analytics.**
