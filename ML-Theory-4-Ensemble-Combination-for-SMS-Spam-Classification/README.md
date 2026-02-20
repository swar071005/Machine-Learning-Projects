# 🏠Experiment 4: Ensemble Learning for SMS Spam Classification Using Voting, Stacking, and AdaBoost

---

# 🎯 Project Objective

The objective of this project is to implement and compare multiple classifier combination techniques for SMS spam detection. The project evaluates individual base learners and ensemble strategies including Hard Voting, Soft Voting, Stacking, and AdaBoost with decision stumps using Stratified K-Fold Cross Validation.

---

# 📄 Problem Statement  

A messaging platform wants to automatically classify SMS messages as **Spam** or **Ham (Not Spam)**.

The goal is to:

- Train individual machine learning classifiers  
- Combine classifiers using ensemble techniques  
- Implement AdaBoost using decision stumps (max_depth = 1)  
- Compare performance using Precision, Recall, F1-score, ROC-AUC  
- Recommend the best combining strategy  

---

# 📊 Dataset Description  

Dataset used: SMS Spam Collection (UCI Dataset ID: 228)

This dataset contains 5,574 SMS messages labeled as spam or ham.

Each record consists of:
- Label (ham or spam)  
- Message text  

---

# 🎯 Target Variable  

**label**

- 0 → Ham  
- 1 → Spam  

---

# 📁 Folder Contents  

```
Spam-Ensemble-Project
 ┣ task4_spam_ensemble_combination.py
 ┣ sms.csv
 ┣ ensemble_comparison.csv
 ┣ final_model_predictions.csv
 ┣ README.md
```

---

# 📥 Input Features  

| Feature  |   Description    |
|----------|------------------|
| message  | SMS text content |

Text is converted into numerical format using **TF-IDF vectorization**.

---

# 🛠 Tools & Technologies  

- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- TF-IDF Vectorizer  
- Google Colab  

---

# ⚙ Methodology  

## 1️⃣ Data Preprocessing  
- Convert dataset to CSV format  
- Label Encoding (ham=0, spam=1)  
- TF-IDF Vectorization  

## 2️⃣ Base Learners  
- Multinomial Naive Bayes  
- Logistic Regression  
- Linear SVM  

## 3️⃣ Ensemble Methods  
- Hard Voting  
- Soft Voting  
- Stacking (Meta-Learner: Logistic Regression)  
- AdaBoost with Decision Stumps (max_depth=1)  

## 4️⃣ Evaluation Strategy  
- Stratified 5-Fold Cross Validation  
- Metrics:
  - Precision  
  - Recall  
  - F1-score  
  - ROC-AUC  
- Confusion Matrix  

---

# 📈 Results & Insights  

Key Observations:

- Logistic Regression and Linear SVM performed strongly as individual models.
- Hard Voting improved stability but ignored probability information.
- Soft Voting performed better than Hard Voting.
- Stacking provided better generalization by learning optimal combination weights.
- AdaBoost with stumps improved weak learners but was slightly less effective on high-dimensional TF-IDF features.

---

### 🔥 Best Performing Model:
**Stacking Classifier**

It achieved the highest F1-score and ROC-AUC with stable cross-validation performance.

---

# ⚖ Bias & Variance Analysis  

|       Model         |       Bias       |            Variance              |
|---------------------|------------------|----------------------------------|
| Naive Bayes         | High Bias        |           Low Variance           |
| Logistic Regression | Moderate         |            Moderate              |
| Linear SVM          | Low Bias         |      Slightly Higher Variance    |
| Hard Voting         | Reduced Variance |          Moderate Bias           |
| Stacking            | Balanced         |         Reduced Variance         |
| AdaBoost            | Low Bias         | Can increase variance if overfit |

Stacking achieves the best bias-variance tradeoff.

---

# ▶ Step-by-Step Execution  

## Step 1  
Install required libraries (if using Colab)

## Step 2  
Run the script

## Step 3  
Check generated files

---

# 📝 Notes  

- Stratified K-Fold ensures balanced spam/ham distribution.  
- AdaBoost uses decision stumps (max_depth=1) as required.  
- TF-IDF improves text feature representation.  
- Stacking uses Logistic Regression as meta-learner.  

---

# 🎓 Viva-Voce Key Points  

1. Why use TF-IDF instead of CountVectorizer?  
   → TF-IDF reduces importance of common words.

2. Difference between Hard and Soft Voting?  
   → Hard uses majority voting; Soft uses probability averaging.

3. What is a Decision Stump?  
   → A decision tree with depth = 1.

4. Why Stratified K-Fold?  
   → Maintains class balance in each fold.

5. Why Stacking performed best?  
   → Meta-learner learns optimal combination of base models.

6. How does AdaBoost work?  
   → Sequentially focuses on misclassified samples.

---

# 🏁 Conclusion  

This project demonstrates that ensemble methods significantly improve spam classification performance.

Among all combining strategies, **Stacking Classifier** achieved the best balance between bias and variance and delivered the highest F1-score and ROC-AUC.

Therefore, stacking is recommended for production-level SMS spam filtering systems.

---

# 🔗 Project & Dataset Links  

**Google Colab Notebook**: 👉[https://colab.research.google.com/drive/14Bf7eai2Bk16McOK-TKvkXsUv23d9PU5#scrollTo=OOGw3mUb1oHr] 

**Dataset Link**:👉[https://archive.ics.uci.edu/dataset/228/sms+spam+collection] 

---

# 🙏 Acknowledgement  

We thank the UCI Machine Learning Repository for providing the SMS Spam Collection dataset for academic use.
We also acknowledge the developers of Scikit-learn for providing powerful machine learning tools.

---
