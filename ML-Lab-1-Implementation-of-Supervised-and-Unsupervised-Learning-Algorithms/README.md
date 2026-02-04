# 🏠 Experiment 1: Implementation of Supervised and Unsupervised Learning Algorithms

---

## 🎯 PROJECT OBJECTIVE

The objective of this project is to understand and implement **Supervised and Unsupervised Machine Learning algorithms** using a real-world dataset.  
The experiment demonstrates prediction using labeled data through **Linear Regression** and pattern discovery in unlabeled data using **K-Means Clustering**.

---

## 🧩 PROBLEM STATEMENT

Machine learning problems often require choosing the correct learning paradigm based on data availability and business needs.  
This experiment addresses:
- Predicting house prices using supervised learning
- Discovering natural groupings in housing data using unsupervised learning

The goal is to analyze, compare, and understand both approaches in a practical setting.

---

## 📊 DATASET DESCRIPTION

- **Dataset Name:** USA Housing Dataset  
- **Source:** Kaggle  
- **Number of Records:** 5000  
- **Number of Features:** 7  

The dataset contains numerical attributes related to housing characteristics and a target variable representing house prices.

---

## 🎯 TARGET VARIABLE

- **Price** – Represents the house price and is used as the output variable in supervised learning (regression).

---

## 📥 INPUT FEATURES

- Avg. Area Income  
- Avg. Area House Age  
- Avg. Area Number of Rooms  
- Avg. Area Number of Bedrooms  
- Area Population  

(The `Address` column is excluded from modeling.)

---

## 📂 FOLDER CONTENTS

| File Name                                   | Description |
|--------------------------------------------|-------------|
| `Experiment1_Supervised_Unsupervised.ipynb` | Colab/Jupyter notebook with code, results, and explanations |
| `USA_Housing.csv`                           | Dataset used for the experiment |
| `README.md`                                | Project documentation |

---

## 🛠️ TOOLS & TECHNOLOGIES

- 🐍 Python 3  
- ☁️ Google Colab  
- 📊 Pandas  
- 🔢 NumPy  
- 📈 Matplotlib  
- 🎨 Seaborn  
- 🤖 Scikit-learn  

---

## 🔍 METHODOLOGY

### Supervised Learning
- Selected features and target variable
- Performed train–test split
- Trained a **Linear Regression** model
- Evaluated performance using MAE, MSE, and RMSE
- Visualized actual vs predicted prices

### Unsupervised Learning
- Removed target variable and categorical data
- Scaled features using `StandardScaler`
- Applied **K-Means Clustering**
- Visualized clusters based on income and population

---

## 📈 RESULTS & INSIGHTS

- Linear Regression predicted house prices with reasonable accuracy.
- The Actual vs Predicted plot showed a strong linear correlation.
- K-Means clustering revealed meaningful groupings in housing data.
- Supervised learning provided measurable accuracy.
- Unsupervised learning helped explore hidden patterns without labels.

---

## ⚖️ BIAS–VARIANCE & BUSINESS PERSPECTIVE

- Linear Regression has **low variance** and **moderate bias**, making it stable and interpretable.
- K-Means clustering is sensitive to feature scaling and the number of clusters.
- From a business perspective:
  - Supervised learning is suitable for **price prediction**
  - Unsupervised learning is useful for **market segmentation and analysis**

---

## ▶️ STEP-BY-STEP EXECUTION

1. Open the Colab notebook
2. Upload the dataset (`USA_Housing.csv`)
3. Run all cells sequentially
4. Observe printed metrics and visualizations
5. Analyze results and conclusions

---

## 📝 NOTES

- Feature scaling is essential for clustering algorithms.
- Linear Regression assumes a linear relationship between features and target.
- Evaluation metrics are clearer in supervised learning than unsupervised learning.

---

## 🎓 VIVA-VOCE KEY POINTS

- Supervised learning uses labeled data; unsupervised learning does not.
- Linear Regression is used for continuous output prediction.
- K-Means clustering groups similar data points.
- Scaling is important for distance-based algorithms.
- Choice of algorithm depends on data and problem requirements.

---

## 🏁 CONCLUSION

This experiment successfully demonstrated both **Supervised and Unsupervised learning techniques** using Python. Linear Regression proved effective for predicting house prices when labeled data was available, while K-Means clustering uncovered hidden structures within the dataset. The experiment highlights the importance of selecting appropriate learning approaches based on data availability, problem objectives, and business requirements.

---

## 🔗 PROJECT & DATASET LINKS

- **Google Colab Notebook:**  
  https://colab.research.google.com/drive/1PQRBhoPnNgC-NJRkUefkyjA3-xRaAluA

- **Dataset Reference:**  
  https://www.kaggle.com/code/fatmakursun/supervised-unsupervised-learning-examples/notebook

---

## 🙌 ACKNOWLEDGEMENT

This project was carried out as part of the **Machine Learning Laboratory** to gain hands-on experience with supervised and unsupervised learning algorithms using real-world data.

---
