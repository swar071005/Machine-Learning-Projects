# 🏦 Experiment 2: Predict Term Deposit Subscription using Logistic Regression

## 🎯 Project Objective
The objective of this experiment is to design, train, and evaluate a **binary classification system** using **Logistic Regression** to predict whether a bank customer will subscribe to a **term deposit**.

This experiment emphasizes:
- Proper handling of categorical and numerical data  
- Probability-based classification using Logistic Regression  
- Comprehensive model evaluation beyond accuracy  
- Threshold optimization for business-oriented decision making  

---

## ❓ Problem Statement
Banks conduct marketing campaigns to encourage customers to invest in term deposits.  
However, incorrect predictions can result in wasted resources or missed revenue opportunities.

**Problem:**  
Given customer demographic information and marketing campaign details, develop a machine learning model that predicts term deposit subscription (`yes` / `no`) while considering the unequal costs of false positives and false negatives.

---

## 📊 Dataset Description
- **Dataset Name:** Bank Marketing Dataset  
- **Source:** UCI Machine Learning Repository  
- **Data Type:** Structured tabular data  
- **Target Variable:** Term deposit subscription (`y`)  

🔗 Dataset Link:  
https://archive.ics.uci.edu/dataset/222/bank+marketing  

---

## 🎯 Target Variable
- **y**
  - `yes` → 1 (Subscribed)  
  - `no` → 0 (Not Subscribed)  

---

## 📌 Input Features
- Client demographic attributes (age, job, marital status, education)
- Financial attributes (balance, housing loan, personal loan)
- Campaign-related attributes (contact type, duration, number of contacts)
- Previous campaign outcomes
- All categorical features encoded using **One-Hot Encoding**

---

## 📁 Folder Contents
Bank-Term-Deposit-Prediction/

├── Bank-Marketing-Campaign.ipynb

├── probabilities.csv

├── README.md

---

## 🛠 Tools & Technologies
🧑‍💻 Platform: Google Colab  
🐍 Language: Python  

📦 Libraries:
- pandas  
- numpy  
- scikit-learn  
- matplotlib  

🤖 Model Used:
- Logistic Regression  

---

## 🔍 Methodology

### 1️⃣ Data Loading
- Dataset loaded from CSV / Excel files extracted from a ZIP archive
- Dataset structure and size verified
- Target variable distribution analyzed for class imbalance

---

### 2️⃣ Data Preprocessing
- Conversion of target variable to binary numeric format
- One-hot encoding applied to categorical variables
- Dataset transformed into a machine-learning-ready format

---

### 3️⃣ Feature–Target Separation
- **X:** All predictive features  
- **y:** Binary target variable  

---

### 4️⃣ Train–Test Split
- **Training set:** 75%  
- **Testing set:** 25%  
- Fixed random seed used to ensure reproducibility

---

### 5️⃣ Model Training
- Logistic Regression model trained using training data
- Probability-based predictions generated

---

### 6️⃣ Model Evaluation (Threshold = 0.5)
Model performance evaluated using:
- Confusion Matrix  
- Precision  
- Recall (Sensitivity)  
- Specificity  
- F1-score  

---

### 7️⃣ ROC Curve and ROC-AUC Analysis
- ROC curve plotted to visualize performance across thresholds
- ROC-AUC score computed as a threshold-independent metric

---

### 8️⃣ Threshold Optimization
- An alternative threshold selected to improve recall
- Demonstrates the precision–recall trade-off
- Enables cost-sensitive decision making

---

### 9️⃣ Output Generation
- Predictions saved in `probabilities.csv`
- Output file includes:
  - RecordId  
  - Probability of subscription  
  - Predicted class label  

---

## 📈 Results & Insights
- Logistic Regression provides stable and interpretable predictions
- ROC-AUC indicates strong discriminatory capability
- Threshold tuning significantly impacts campaign effectiveness
- Evaluation beyond accuracy improves real-world reliability

---

## 🧠 Bias–Variance & Business Perspective
- Logistic Regression exhibits low variance and high interpretability
- Suitable for structured business datasets
- Threshold customization aligns predictions with business priorities

---

## 🧪 Step-by-Step Execution
1. Upload ZIP dataset to Google Colab  
2. Extract and load dataset files  
3. Preprocess and encode data  
4. Split data into training and testing sets  
5. Train Logistic Regression model  
6. Evaluate performance using classification metrics  
7. Analyze ROC curve and ROC-AUC  
8. Optimize decision threshold  
9. Export prediction results to CSV  

---

## 📝 Notes
- Accuracy alone does not capture true model performance  
- ROC curve provides holistic evaluation  
- Lower thresholds increase recall at the cost of precision  
- Threshold selection should be business-driven  

---

## 🎓 Viva-Voce Key Points
- Logistic Regression for binary classification  
- Confusion Matrix interpretation  
- Precision vs Recall trade-off  
- Sensitivity and Specificity  
- ROC curve and ROC-AUC  
- Business-driven threshold selection  

---

## ✅ Conclusion
This experiment presents a complete and systematic approach to binary classification using Logistic Regression. By incorporating ROC analysis and threshold optimization, the model aligns technical performance with practical business requirements.

---

## 🔗 Project & Dataset Links
📘 **Google Colab Notebook** 👉 [https://colab.research.google.com/drive/1LeoTkdZObKmTU0Ll8T77f0_3oQsCg2-z?usp=sharing]

📊 **UCI Bank Marketing Dataset** 👉 [https://archive.ics.uci.edu/dataset/222/bank+marketing]

---

