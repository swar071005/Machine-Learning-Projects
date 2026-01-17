# 🚢 Experiment 1: Titanic – Machine Learning from Disaster

## 🎯 Project Objective
The objective of this experiment is to apply **Machine Learning techniques** to analyze real-world structured data and understand how different models behave in terms of **bias and variance**.  
This experiment focuses on data preprocessing, feature scaling, training multiple regression models, and evaluating their performance.

---

## ❓ Problem Statement
Predictive modeling on real-world datasets (such as Titanic-like disaster data) involves multiple interacting features.  
Simple models may underfit the data, while complex models may overfit.

**Problem:**  
Build and evaluate multiple Machine Learning models to:
- Learn relationships between features and target values  
- Analyze bias–variance trade-off  
- Improve prediction accuracy using ensemble learning  

---

## 📊 Dataset Description
The dataset is loaded from an Excel file using Pandas.

### 🎯 Target Variable
- `median_house_value`

### 📌 Input Features
- longitude, latitude  
- housing_median_age  
- total_rooms, total_bedrooms  
- population, households  
- median_income  
- One-hot encoded categorical features  

---

## 📁 Folder Contents
Titanic-Machine-Learning-from-Disaster/

├── Titanic - Machine Learning from Disaster.ipynb

├── housing.xlsx

├── README.md

---

## 🛠 Tools & Technologies
- 🧑‍💻 Platform: Google Colab  
- 🐍 Language: Python  
- 📦 Libraries:
  - pandas
  - numpy
  - scikit-learn
  - openpyxl
- 🤖 Models Used:
  - Linear Regression
  - Ridge Regression
  - Decision Tree Regressor
  - Random Forest Regressor

---

## 🔍 Methodology

### 1️⃣ Data Loading
- Excel dataset loaded using `pandas.read_excel()`
- Data preview and column verification
- Missing values checked using `isnull().sum()`

### 2️⃣ Feature Selection
- Target variable:
  - `median_house_value`
- Input features:
  - All remaining columns

### 3️⃣ Train–Test Split
- 80% training data  
- 20% testing data  
- Ensures unbiased model evaluation

### 4️⃣ Feature Scaling
- `StandardScaler` applied to numerical features
- Used for Linear & Ridge Regression
- Tree-based models trained without scaling

### 5️⃣ Model Training
The following models were trained:
- Linear Regression
- Ridge Regression (regularization)
- Decision Tree Regression
- Random Forest Regression (ensemble learning)

### 6️⃣ Model Evaluation
Models evaluated using:
- 📉 Root Mean Squared Error (RMSE)
- 📐 Mean Absolute Error (MAE)

---

## 📈 Results & Insights

### 🔎 Model Performance Summary

|       Model       | Train RMSE | Test RMSE | Test MAE |
|-------------------|------------|-----------|----------|
| Linear Regression |   High     |   High    |   High   |
| Ridge Regression  |   High     |   High    |   High   |
| Decision Tree     |    0.0     |   High    |   High   |
| Random Forest     |   18118    |   49038   |  31639   |

### 🧠 Bias–Variance Analysis
- **Linear & Ridge Regression**  
  🔹 High bias → underfitting  
  🔹 Fail to capture non-linear relationships  

- **Decision Tree**  
  🔸 High variance → overfitting  
  🔸 Perfect training accuracy but poor test results  

- **Random Forest**  
  ✅ Reduces overfitting using multiple trees  
  ✅ Better generalization and stability  

---

## 🧪 Step-by-Step Execution
1. Install dependency:
   ```bash
   pip install openpyxl
2. Load dataset using Pandas

3. Check missing values and columns

4. Split dataset into training & testing sets

5. Apply feature scaling

6. Train models

7. Evaluate using RMSE and MAE

8. Compare results and analyze performance

---

## 📝 Notes

1. Feature scaling is essential for linear models

2. Decision Trees do not require scaling

3. Ensemble models improve robustness

4. Model selection depends on bias–variance trade-off

---

## ✅ Conclusion

This experiment demonstrates a complete Machine Learning workflow including data preprocessing, model training, evaluation, and bias–variance analysis.
The use of ensemble learning improves model generalization and highlights best practices for real-world predictive modeling.

---

## 📚 References
Google Colab Documentation: [https://colab.research.google.com/]

Kaggle Titanic Dataset: [https://www.kaggle.com/datasets/c/titanic]
