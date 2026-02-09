# 🏠 Experiment 3: Bike Demand Forecasting using Ensemble Regression (Bagging, Subagging & Boosting)
---

## 📌 Project Objective
The objective of this project is to forecast hourly bike rental demand (`cnt`) using ensemble regression models including Bagging (Random Forest), Subagging (BaggingRegressor with `max_samples < 1.0`), and Boosting (GradientBoostingRegressor). Prediction accuracy is evaluated using 5-Fold cross validation.

---

## 🧩 Problem Statement
Urban mobility teams need accurate bike demand predictions to improve fleet distribution and minimize shortages. This experiment uses the UCI Bike Sharing dataset to build and compare ensemble regression models and determine which approach generalizes best for forecasting hourly bike rentals.

---

## 📊 Dataset Description
Dataset used: **hour.csv** from the UCI Bike Sharing Dataset (UCI Machine Learning Repository).  
It contains hourly records of bike rental counts along with weather and seasonal attributes collected over two years from the Capital Bikeshare system in Washington, D.C., USA. :contentReference[oaicite:1]{index=1}

The dataset includes 17,389 instances with features such as hour of the day, season, temperature, humidity, windspeed, and target rental count (`cnt`). :contentReference[oaicite:2]{index=2}

---

## 🎯 Target Variable
- **cnt** – Total count of bike rentals (sum of casual + registered) per hour.

---

## 📥 Input Features
The model uses the following input features after preprocessing:
- season – Season of the year
- yr – Year (0: 2011, 1: 2012)
- mnth – Month of the year
- hr – Hour of the day
- holiday – Whether the day is a holiday
- weekday – Day of the week
- workingday – Whether it’s a working day
- weathersit – Weather situation
- temp – Normalized temperature
- atemp – Normalized feeling temperature
- hum – Normalized humidity
- windspeed – Normalized wind speed

*Note:* `instant`, `dteday`, `casual`, and `registered` features are removed before modeling.

---

## 📂 Folder Contents

|           File Name              |                      Description                      |
|----------------------------------|-------------------------------------------------------|
| `hour.csv`                       | UCI Bike Sharing dataset used for regression analysis |
| `Ensemble_Bike_Regression.ipynb` |     Colab notebook with implementation and outputs    |
| `cv_regression_results.csv`      |            Cross validation metrics per model         |
| `final_predictions.csv`          |             Predicted vs actual bike counts           |
| `README.md`                      |                   Project documentation               |

---

## 🛠️ Tools & Technologies
- 🐍 Python 3  
- ☁️ Google Colab  
- 📊 Pandas  
- 🔢 NumPy  
- 🤖 Scikit-Learn  
- 📈 Matplotlib / Seaborn  

---

## 📋 Methodology
1. Load and preprocess the dataset (drop unused columns, encode categorical variables).
2. Define 5-Fold Cross Validation.
3. Train and evaluate:
   - RandomForestRegressor (Bagging)
   - BaggingRegressor with DecisionTree (Subagging)
   - GradientBoostingRegressor (Boosting)
4. Use cross validation to calculate mean RMSE and MAE with standard deviations.
5. Train best model on full dataset.
6. Save predictions and feature importance.

---

## 📈 Results & Insights
- Cross-validation results include average RMSE and MAE for each model.
- Ensemble methods show better generalization compared to single tree models.
- Models with weaker bias and proper complexity show stronger performance.
- Feature importance shows which inputs most influence bike rental counts (e.g., hour, temperature, humidity).

---

## ⚖️ Bias & Variance Analysis
- **Bagging (Random Forest)**: Reduces variance by averaging many decorrelated trees.
- **Subagging**: Introduces additional randomness via subsampling to further reduce variance.
- **Boosting (Gradient)**: Reduces bias by sequentially learning from residuals.
- Optimal model selection depends on the balance between bias and variance for this dataset.

---

## 🧠 Step-by-Step Execution

1. Upload `hour.csv` to Colab.
2. Load data and inspect initial rows.
3. Drop irrelevant columns (`instant`, `casual`, `registered`, etc.).
4. Define features (X) and target (`cnt`).
5. Create 5-Fold cross validator.
6. Define ensemble models (Random Forest, Subagging, Boosting).
7. Perform cross validation to compute RMSE and MAE.
8. Save `cv_regression_results.csv`.
9. Train best model on full dataset and save `final_predictions.csv`.
10. Plot performance and feature importance.

---

## 📝 Notes
- Ensure categorical variables are numeric before modeling.
- Scale/normalize features if needed for some algorithms.
- Keep cross validation consistent for fair comparison.
- Use error metrics with both mean and standard deviation.

---

## 🎓 Viva-Voce Key Points
- Difference between Bagging, Subagging, and Boosting.
- Meaning and interpretation of cross-validation metrics (RMSE, MAE).
- How ensembles reduce bias/variance.
- Feature importance and its significance.
- Real-world relevance of demand forecasting.

---

## 🏁 Conclusion
The ensemble regression models show that Boosting (or the best performing model in your results) provides the most accurate hourly bike demand forecasting based on cross validation metrics. Ensemble approaches reduce error reliably and improve generalization compared to basic regression techniques, thus making them suitable for real-world forecasting problems like bike rental demand.

---

## 🔗 Project & Dataset Links
**Google Colab Notebook:** [https://colab.research.google.com/drive/1cCB5qMoc0lz-v-Sz6jnFmt2b9TSvTWIu?usp=sharing]

**Dataset:** [https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset :contentReference[oaicite:3]{index=3}]

---

## 🙌 Acknowledgement
This project was conducted as part of a machine learning practicum to understand and compare ensemble regression models using real-world bike sharing data. Thanks to the UCI Machine Learning Repository for providing open-access datasets for educational purposes.

---
