# 🏠 Experiment 1: House Price Prediction using Machine Learning

## 🎯 Project Objective
The objective of this experiment is to build and evaluate multiple **Machine Learning regression models** to predict house prices based on given housing features.  
The project focuses on understanding:
- Data preprocessing and feature scaling
- Training different regression models
- Model evaluation using error metrics
- Bias–Variance trade-off in Machine Learning

---

## ❓ Problem Statement
House prices depend on multiple factors such as location, number of rooms, population, and income levels.  
Predicting house prices accurately is challenging due to **non-linear relationships, noise, and feature interactions**.

**Problem:**  
Given a structured housing dataset, train different regression models and compare their performance to identify the most suitable model for accurate price prediction.

---

## 📊 Dataset Description
The dataset is loaded from an Excel file (`housing.xlsx`) using Pandas.

---

## 🎯 Target Variable
- `median_house_value`

---

## 📌 Input Features
- longitude  
- latitude  
- housing_median_age  
- total_rooms  
- total_bedrooms  
- population  
- households  
- median_income  
- one-hot encoded ocean proximity features  

---

## 📁 Folder Contents
House-Price-Prediction/
├── House_Price_Prediction.ipynb
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
- Dataset loaded using `pandas.read_excel()`
- Previewed using `head()`
- Missing values checked using `isnull().sum()`

### 2️⃣ Feature Selection
- Target variable separated:
  - `y = median_house_value`
- Input features:
  - `X = all remaining columns`

### 3️⃣ Train–Test Split
- Data split into:
  - 80% Training data
  - 20% Testing data
- Ensures unbiased evaluation

### 4️⃣ Feature Scaling
- `StandardScaler` applied to input features
- Scaling used for:
  - Linear Regression
  - Ridge Regression
- Tree-based models trained without scaling

### 5️⃣ Model Training
The following models were trained:
- Linear Regression
- Ridge Regression (L2 regularization)
- Decision Tree Regression
- Random Forest Regression (ensemble learning)

### 6️⃣ Model Evaluation
Models evaluated using:
- 📉 Root Mean Squared Error (RMSE)
- 📐 Mean Absolute Error (MAE)

---

## 📈 Results & Insights

### 🔎 Model Performance Comparison

|       Model       | Train RMSE | Test RMSE | Test MAE |
|-------------------|------------|-----------|----------|
| Linear Regression |    High    |    High   |    High  |
| Ridge Regression  |    High    |    High   |    High  |
| Decision Tree     |    0.0     |    High   |    High  |
| Random Forest     |   18118    |   49038   |   31639  |

---

## 🧠 Bias & Variance Analysis

- **Linear Regression & Ridge Regression**  
  🔹 Show **underfitting (high bias)**  
  🔹 Training and testing errors are similar and relatively high  
  🔹 Unable to capture complex non-linear patterns  

- **Decision Tree Regression**  
  🔸 Shows **overfitting (high variance)**  
  🔸 Zero training error but high testing error  
  🔸 Memorizes training data  

- **Random Forest Regression**  
  ✅ Reduces overfitting by combining multiple trees  
  ✅ Provides better generalization  
  ✅ Lower test RMSE and MAE compared to a single Decision Tree  

---

## 🧪 Step-by-Step Execution
1. Install required dependency:
   ```bash
   pip install openpyxl
   
2. Load the dataset

3. Check for missing values

4. Split the data into training and testing sets

5. Apply feature scaling

6. Train regression models

7. Evaluate and compare model performance
   
---

## 📝 Notes

1. Feature scaling is essential for linear models

2. Tree-based models do not require scaling

3. Ensemble methods improve stability and accuracy

4. Model selection depends on bias–variance trade-off

---

## 🎓 Viva-Voce Key Points

1. Supervised learning and regression

2. Importance of feature scaling

3. Bias vs Variance

4. Overfitting and underfitting

5. Why Random Forest performs better

6. RMSE vs MAE

---

## ✅ Conclusion

This experiment demonstrates a complete Machine Learning regression pipeline, including data preprocessing, model training, evaluation, and bias–variance analysis.
Among all models, Random Forest Regression performs best due to its ability to handle non-linear relationships and reduce overfitting.

---

## 🔗 Project & Dataset Links

📘 Google Colab Notebook: [https://colab.research.google.com/drive/1lIqtIIUedyLFuLV7IFcKBgurQxJy81wl]

📊 Kaggle Dataset: [https://www.kaggle.com/datasets/camnugent/california-housing-prices]



