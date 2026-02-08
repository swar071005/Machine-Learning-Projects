# 🏥 Experiment 2: Implementation of Linear and Nonlinear Regression Models  

---

## 🎯 PROJECT OBJECTIVE  
The objective of this project is to implement and compare **Linear Regression (Supervised Learning)** and **Nonlinear Regression (Polynomial Regression)** using a real-world medical insurance dataset.  

The experiment aims to analyze how different regression techniques perform in predicting insurance charges and to understand the importance of modeling nonlinear relationships in real-world data.

---

## 🧩 PROBLEM STATEMENT  
Medical insurance charges depend on several demographic and health-related factors such as age, BMI, smoking habits, and region.  

This experiment addresses:  

- Predicting medical insurance charges using Linear Regression  
- Improving prediction accuracy using Nonlinear (Polynomial) Regression  
- Comparing model performance using evaluation metrics  

The goal is to determine whether a simple linear model is sufficient or a nonlinear model better captures data patterns.

---

## 📊 DATASET DESCRIPTION  
**Dataset Name:** Medical Insurance Cost Dataset  
**Source:** Kaggle  
**Number of Records:** 1338  
**Number of Features:** 7 (6 input features + 1 target variable)  

The dataset contains both numerical and categorical attributes describing individuals and their corresponding insurance charges.

---

## 🎯 TARGET VARIABLE  
**charges** – Represents the medical insurance cost (continuous numerical value).

---

## 📥 INPUT FEATURES  

- age – Age of the individual  
- sex – Gender  
- bmi – Body Mass Index  
- children – Number of children  
- smoker – Smoking status  
- region – Residential region  

(Categorical variables were converted into numerical form using encoding techniques.)

---

## 📂 FOLDER CONTENTS  

|        File Name               |                     Description                      |
|--------------------------------|------------------------------------------------------|
| Insurance_Regression.ipynb     | Colab notebook containing implementation and outputs |
| insurance.csv                  | Dataset used for training and testing                |
| README.md                      | Project documentation                                |

---

## 🛠️ TOOLS & TECHNOLOGIES  

- 🐍 Python 3  
- ☁️ Google Colab  
- 📊 Pandas  
- 🔢 NumPy  
- 📈 Matplotlib  
- 🤖 Scikit-learn  
- 📐 PolynomialFeatures  

---

## 🔍 METHODOLOGY  

### Linear Regression  
- Selected input features and target variable  
- Encoded categorical variables  
- Performed train–test split (80:20)  
- Trained Linear Regression model  
- Evaluated using MAE, MSE, RMSE, and R² Score  

### Nonlinear Regression (Polynomial Regression)  
- Applied Polynomial Feature transformation (degree 2)  
- Trained regression model on transformed features  
- Compared performance with Linear Regression  
- Visualized prediction performance  

---

## 📈 RESULTS & INSIGHTS  

- Linear Regression provided a strong baseline prediction model.  
- Polynomial Regression captured nonlinear relationships more effectively.  
- Nonlinear model showed improved R² score and lower RMSE.  
- Smoking status and BMI significantly influenced insurance charges.  
- Real-world datasets often contain nonlinear patterns that simple linear models may not fully capture.  

---

## ⚖️ BIAS–VARIANCE & BUSINESS PERSPECTIVE  

**Linear Regression:**  
- Higher bias (may underfit complex relationships)  
- Low variance and highly interpretable  

**Polynomial Regression:**  
- Lower bias  
- Slightly higher variance (risk of overfitting if degree is high)  

**Business Perspective:**  
- Accurate prediction helps insurance companies in premium pricing  
- Assists in risk assessment and customer segmentation  
- Supports financial planning and decision-making  

---

## ▶️ STEP-BY-STEP EXECUTION  

1. Open the Google Colab notebook  
2. Upload the dataset (insurance.csv)  
3. Import required libraries  
4. Perform data preprocessing and encoding  
5. Define features (X) and target (y)  
6. Split dataset into training and testing sets  
7. Train Linear Regression model  
8. Apply Polynomial transformation and train Nonlinear model  
9. Evaluate models using performance metrics  
10. Compare results and draw conclusions  

---

## 📝 NOTES  

- Ensure dataset contains no missing values before training  
- Avoid using high polynomial degree to prevent overfitting  
- Use multiple evaluation metrics for proper comparison  
- Feature encoding is necessary for categorical variables  

---

## 🎓 VIVA-VOCE KEY POINTS  

- Difference between Linear and Polynomial Regression  
- Meaning of R² Score, MAE, MSE, RMSE  
- Concept of overfitting and underfitting  
- Importance of encoding categorical variables  
- Applications of regression in real-world industries  

---

## 🏁 CONCLUSION  

This experiment successfully demonstrated the implementation of both Linear and Nonlinear Regression techniques on the Medical Insurance dataset. While Linear Regression provided a simple and interpretable model, Polynomial Regression improved predictive accuracy by capturing nonlinear relationships. The experiment highlights the importance of selecting appropriate regression models based on data complexity and business objectives.

---

## 🔗 PROJECT & DATASET LINKS  

**Google Colab Notebook:** 👉 [https://colab.research.google.com/drive/13pIGt7cZv7mj8DQ3_PMWenSCQkOyllPG]

**Dataset:** 👉 [https://www.kaggle.com/datasets/mirichoi0218/insurance]

---

## 🙌 ACKNOWLEDGEMENT  

This project was carried out as part of the Machine Learning Laboratory to gain practical understanding of regression techniques using real-world healthcare cost data.

