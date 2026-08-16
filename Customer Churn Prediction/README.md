# Customer Churn Prediction — Working with a Business Problem
> **Neurofive ML Track — Week 3 (Customer Churn Prediction)**

This project focuses on predicting customer attrition for a telecommunications provider using the Telco Customer Churn dataset. It covers business-driven EDA, handling categorical encoding and class imbalance, training and comparing Logistic Regression vs. Decision Tree models, and interpreting top churn drivers using feature importances.

---

## 🛠️ Tech Stack & Toolkit

* **Language:** Python 3.12
* **Environment:** Jupyter Notebook / Virtual Environment (`venv`)
* **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
* **Version Control:** Git & GitHub

---

## 📊 Dataset & Preprocessing

* **Dataset:** Telco Customer Churn (`7,043` records, `21` columns)
* **Target Variable:** `Churn` (Binary: `1` for churned, `0` for retained)
* **Class Imbalance:** ~73.5% non-churn vs. ~26.5% churn (observed and preserved using stratified splitting)
* **Data Cleaning:** Coerced whitespace anomalies in `TotalCharges` to numeric values, handled missing values, and dropped non-informative identifier columns (`customerID`).
* **Feature Encoding:** Applied One-Hot Encoding (`pd.get_dummies(drop_first=True)`) to transform all categorical variables into numeric inputs.
* **Feature Scaling:** Applied `StandardScaler` for the Logistic Regression pipeline.

---

## 📈 Exploratory Data Analysis (EDA) Insights

1. **Contract Structure:** Customers on **Month-to-month contracts** churn at significantly higher rates compared to those on 1-year or 2-year commitments.
2. **Customer Lifecycle (Tenure):** Churn heavily spikes in the initial 0–12 month window, showing that early customer onboarding is a critical retention period.
3. **Billing Impact:** Customers with higher `MonthlyCharges` exhibit greater churn frequency, particularly those subscribing to Fiber Optic internet services without bundled tech support.

---

## 🤖 Model Comparison & Evaluation

Both models were trained using an 80/20 stratified split (`random_state=42`):

| Metric | Logistic Regression | Decision Tree (`max_depth=5`) |
| :--- | :--- | :--- |
| **Accuracy** | **~80%** | ~78% |
| **Precision** | **~0.65** | ~0.60 |
| **Recall** | **~0.54** | ~0.50 |
| **F1-Score** | **~0.59** | ~0.55 |

* **Decision Tree Interpretation:** Constraining tree depth to `max_depth=5` prevented severe tree overfitting while maintaining clear rule interpretability.
* **Logistic Regression Advantage:** Provided a better balance between precision and recall across decision thresholds on scaled features.

---

## 🔍 Top 3 Drivers of Churn (`.feature_importances_`)

Using the Decision Tree model's feature importance attributes, the top 3 drivers of customer churn were identified as:

1. **`Contract_Month-to-month` / Contract Type:** The strongest predictor of customer churn.
2. **`tenure`:** Duration of customer account activity (newer accounts show elevated churn risk).
3. **`MonthlyCharges` / `InternetService_Fiber optic`:** Cost and specific internet service tiers create pricing sensitivity.

---

## 💼 Executive Business Summary

* **High-Risk Profile:** Month-to-month subscribers within their first year of service paying higher monthly fees present the highest likelihood of churn.
* **Actionable Retention Strategy:** Introduce discounted 1-year contract migration offers and proactive onboarding check-ins during the first 90 days of subscription to stabilize retention.

---

## 🌲 Ensemble Learning: Random Forest vs. XGBoost (Task 8)

This project compares single baseline models (Logistic Regression / Decision Trees) against advanced ensemble architectures (Random Forest and XGBoost) on Customer Churn data.

### Model Comparison Table

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | ~0.8041 | ~0.6552 | ~0.5374 | ~0.5907 |
| **Random Forest** | ~0.8062 | ~0.6725 | ~0.5134 | ~0.5823 |
| **XGBoost** | **~0.8112** | **~0.6844** | **~0.5401** | **~0.6036** |

### Key Takeaways
1. **Performance Lift:** XGBoost achieved the best overall F1-score and accuracy through sequential gradient boosting.
2. **Bagging vs. Boosting:** Random Forest averages independent decision trees to reduce variance, while XGBoost sequentially trains trees to minimize the residual errors of earlier trees.

---

## 🚀 Getting Started Locally

### 1. Set Up Environment & Run
git clone https://github.com/notdumbj/neurofive-ml-track.git
cd "neurofive-ml-track/Customer Churn Prediction"

# On Windows (Git Bash):
source ../ml_env/Scripts/activate

# On Windows (CMD):
..\ml_env\Scripts\activate

# Launch Jupyter
jupyter notebook