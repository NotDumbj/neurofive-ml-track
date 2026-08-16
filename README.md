# NeuroFive ML Track — Weeks 1, 2, 3, 4 & 5
> Machine Learning fundamentals through hands-on projects

This repository contains projects from the NeuroFive ML track, covering supervised learning techniques, regression modeling, advanced classification evaluation, hyperparameter tuning, and imbalanced learning on real-world datasets.

---

## Projects

### 1. Titanic Logistic Regression & Model Tuning
`Titanic Logistic Regression/`

End-to-end ML pipeline on the Kaggle Titanic dataset covering EDA, data cleaning, feature encoding, baseline logistic regression classification, and systematic hyperparameter tuning using `GridSearchCV`.

**Tasks covered:** Tasks 1 & 2 (Week 1), Task 3 (Week 2), Task 5 (Week 3)

### 2. California Housing Linear Regression
`California Housing Linear Regression/`

Linear Regression model predicting California housing prices using scikit-learn's built-in California Housing dataset. Uses a 4-feature subset (`MedInc`, `HouseAge`, `AveRooms`, `Population`) to predict median house values, achieving an R² of ~0.50.

**Task covered:** Task 4 (Week 2)

### 3. Customer Churn Prediction
`Customer Churn Prediction/`

Binary classification project predicting customer attrition for a telecom provider using the Telco Customer Churn dataset (7,043 records). Covers business-driven EDA, categorical encoding, class imbalance awareness, Logistic Regression vs. Decision Tree comparison, feature importance analysis to identify top churn drivers, and ensemble learning with Random Forest and XGBoost.

**Tasks covered:** Task 6 (Week 3), Task 8 (Week 4)

### 4. Titanic ML Pipeline
`Titanic ML Pipeline/`

Production-style sklearn `Pipeline` with `ColumnTransformer` for the Titanic dataset. Introduces feature engineering (`FamilySize`, `IsAlone`), unified preprocessing (imputation + scaling + encoding in one object), and model persistence via `joblib` for end-to-end inference on raw inputs.

**Task covered:** Task 7 (Week 4)

### 5. Credit Card Fraud Detection
`Credit Card Fraud Detection/`

Binary classification on the ULB Credit Card Fraud dataset (284,807 transactions, 99.83% legitimate vs. 0.17% fraud). Demonstrates the accuracy paradox in extreme class imbalance, compares baseline Logistic Regression against class-weighted and SMOTE-resampled strategies, and optimizes for Recall to maximize fraud catch rate.

**Task covered:** Task 9 (Week 5)

---

## Week Overview

| Week | Tasks | Project | Focus Areas |
|------|-------|---------|-------------|
| Week 1 | Tasks 1 & 2 | Titanic | EDA, Data Profiling, Outliers & Missing Value Imputation |
| Week 2 | Tasks 3 & 4 | Titanic & California Housing | One-Hot Encoding, Logistic Regression, Linear Regression (RMSE & R²) |
| Week 3 | Tasks 5 & 6 | Titanic & Customer Churn | Beyond Accuracy (Precision, Recall, F1), GridSearchCV, Business-Driven Churn Prediction |
| Week 4 | Tasks 7 & 8 | Titanic Pipeline & Customer Churn | sklearn Pipelines, ColumnTransformer, Feature Engineering, Model Persistence & Ensemble Learning (Random Forest, XGBoost) |
| Week 5 | Task 9 | Credit Card Fraud Detection | Extreme Class Imbalance, Accuracy Paradox, SMOTE, Class Weighting, Recall Optimization |

---

## Tech Stack & Toolkit

* **Language:** Python 3.12
* **Environment:** Jupyter Notebook / Virtual Environment (`venv`)
* **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`, `xgboost`, `imbalanced-learn`
* **Version Control:** Git & GitHub

---

## Getting Started Locally

### 1. Clone the Repository
git clone https://github.com/notdumbj/neurofive-ml-track.git
cd neurofive-ml-track

### 2. Set Up Virtual Environment
python -m venv ml_env

# On Windows (Git Bash):
source ml_env/Scripts/activate

# On Windows (CMD):
ml_env\Scripts\activate

# On Linux/macOS:
source ml_env/bin/activate

### 3. Install Dependencies & Launch
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
cd "Titanic Logistic Regression"    # or "California Housing Linear Regression"
jupyter notebook

---

## Repository Structure
```
neurofive-ml-track/
├── .gitignore
├── README.md
├── ml_env/                                  # Shared virtual environment (gitignored)
├── Titanic Logistic Regression/
│   ├── data/                                # Raw CSVs (gitignored, download separately)
│   │   └── .gitkeep
│   └── main.ipynb                           # Notebook containing Tasks 1-3 & Task 5
├── California Housing Linear Regression/
│   └── main.ipynb                           # Notebook containing Task 4
├── Customer Churn Prediction/
│   ├── data/                                # Telco churn CSV (gitignored)
│   │   └── .gitkeep
│   ├── README.md                            # Project-specific documentation
│   └── main.ipynb                           # Notebook containing Tasks 6 & 8
├── Credit Card Fraud Detection/
│   ├── data/                                # Fraud dataset CSV (gitignored)
│   │   └── .gitkeep
│   ├── README.md                            # Project-specific documentation
│   └── main.ipynb                           # Notebook containing Task 9
└── Titanic ML Pipeline/
    ├── main.ipynb                           # Notebook containing Task 7
    └── titanic_pipeline_model.joblib        # Saved sklearn Pipeline artifact
```
---

## Titanic Logistic Regression & Hyperparameter Tuning — Detail

### Data Cleaning & Preprocessing Strategy (Task 2)

1. **`Age` Imputation:** Missing values (~20%) were filled using the column **median**. Median was preferred over mean because the age distribution exhibits slight right-skewness, making the median more resilient to extreme values.
2. **`Embarked` Imputation:** Missing values (2 records) were filled using the **mode** (`'S'`), preserving categorical integrity without altering class proportions.
3. **`Cabin` Removal:** Over 77% of `Cabin` entries were missing. The feature was dropped entirely to prevent introducing synthetic noise.
4. **Outlier Identification:** Boxplot profiling on `Fare` revealed severe right-skewness and outliers ranging beyond `$500`. These entries reflect high-tier first-class suites and combined family tickets rather than bad data.

### Key Visual Insights ("Data Story")

* **Primary Survival Driver (`Sex`):** Visualizing survival rate by gender reveals that females had a **~74%** survival probability compared to **~19%** for males, demonstrating the strong impact of the "women and children first" evacuation protocol.
* **Socioeconomic Influence (`Pclass`):** Passenger class serves as a strong secondary predictor. First-class passengers achieved higher survival rates due to proximity to the upper deck and priority lifeboat access.
* **Correlation Highlights:** Strong negative correlation exists between `Pclass` and `Fare`, confirming that higher fare values strongly map to tier 1 accommodations.

### Model Implementation & Hyperparameter Tuning (Tasks 3 & 5)

#### Pipeline Steps
1. **Categorical Encoding:** Converted categorical variables (`Sex`, `Embarked`) using One-Hot Encoding (`pd.get_dummies(drop_first=True)`).
2. **Train-Test Split:** Partitioned the dataset into 80% training and 20% test subsets using stratified sampling to maintain class proportions.
3. **Feature Scaling:** Applied `StandardScaler` to normalize continuous distributions (`Age`, `Fare`) prior to training.
4. **Baseline Algorithm:** Trained a **Logistic Regression** baseline (`C=1.0`).
5. **Hyperparameter Tuning:** Applied `GridSearchCV` with 5-fold cross-validation searching over `C`: `[0.01, 0.1, 1, 10, 100]` and `solver`: `['liblinear', 'lbfgs']`.

#### Performance & Metrics Comparison

| Metric | Baseline Model (`C=1.0`) | Tuned Model (`C=0.1`, `solver='liblinear'`) |
| :--- | :--- | :--- |
| **Accuracy** | **0.8045** | 0.7877 |
| **Precision** | **0.7931** | 0.7541 |
| **Recall** | **0.6667** | 0.6667 |
| **F1-Score** | **0.7244** | 0.7077 |

* **Key Takeaway:** The default Logistic Regression parameters (`C=1.0`) provided the optimal balance on the unseen test set. Applying stronger regularization (`C=0.1`) slightly underfit the training features on this dataset scale, illustrating that hyperparameter tuning does not automatically guarantee higher accuracy on holdout test splits.

---

## California Housing Linear Regression — Detail

### Dataset
* **Source:** `sklearn.datasets.fetch_california_housing` (built-in, no download required)
* **Size:** 20,640 samples, 8 features + 1 target
* **Target:** `MedHouseVal` — median house value (scaled to dollars)

### Pipeline Steps (Task 4)
1. **Feature Selection:** Selected 4 features — `MedInc`, `HouseAge`, `AveRooms`, `Population`.
2. **Target Scaling:** Multiplied `MedHouseVal` by 100,000 to convert to dollar units.
3. **Train-Test Split:** 80/20 split with `random_state=42`.
4. **Algorithm:** Trained a **Linear Regression** model via `sklearn.linear_model.LinearRegression`.

### Performance & Evaluation
* **RMSE:** **$81,104.84**
* **R² Score:** **~0.50** — the model explains 49% of the variance in house prices using the selected 4-feature subset. The remaining variance is attributed to factors not included (e.g., proximity to the ocean, property condition).

---

## Customer Churn Prediction — Detail

### Dataset
* **Source:** Telco Customer Churn dataset (`7,043` records, `21` columns)
* **Target:** `Churn` (Binary: `1` = churned, `0` = retained)
* **Class Imbalance:** ~73.5% non-churn vs. ~26.5% churn

### Pipeline Steps (Task 6)
1. **Data Cleaning:** Coerced whitespace anomalies in `TotalCharges` to numeric, handled missing values, dropped `customerID`.
2. **Feature Encoding:** Applied One-Hot Encoding (`pd.get_dummies(drop_first=True)`) across all categorical variables (30 features post-encoding).
3. **Train-Test Split:** 80/20 stratified split (`random_state=42`).
4. **Feature Scaling:** `StandardScaler` applied for the Logistic Regression pipeline.
5. **Models Trained:** Logistic Regression (baseline) and Decision Tree (`max_depth=5`).

### Performance & Evaluation

| Metric | Logistic Regression | Decision Tree (`max_depth=5`) |
| :--- | :--- | :--- |
| **Accuracy** | **0.8070** | 0.7942 |
| **Precision** | **0.6584** | 0.6312 |
| **Recall** | **0.5668** | 0.5401 |
| **F1-Score** | **0.6092** | 0.5821 |

### Top 3 Churn Drivers (Decision Tree Feature Importances)
1. **`Contract_Month-to-month`** — strongest predictor of churn.
2. **`tenure`** — newer accounts show elevated churn risk.
3. **`MonthlyCharges` / `InternetService_Fiber optic`** — pricing sensitivity and service tier.

### Business Recommendation
Target month-to-month subscribers in their first year with discounted annual contract offers and proactive onboarding check-ins during the first 90 days.

### Ensemble Learning: Random Forest vs. XGBoost (Task 8)

Extended the baseline models with two ensemble architectures to evaluate performance lift on the same churn dataset.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | ~0.8041 | ~0.6552 | ~0.5374 | ~0.5907 |
| Random Forest | ~0.8062 | ~0.6725 | ~0.5134 | ~0.5823 |
| **XGBoost** | **~0.8112** | **~0.6844** | **~0.5401** | **~0.6036** |

* **XGBoost** achieved the best overall F1-score and accuracy through sequential gradient boosting.
* **Bagging vs. Boosting:** Random Forest averages independent decision trees to reduce variance, while XGBoost sequentially trains trees to minimize the residual errors of earlier trees.

---

## Titanic ML Pipeline — Detail

### Concept
A **Pipeline** bundles preprocessing steps (imputation, scaling, encoding) and the estimator into a single object, preventing data leakage and eliminating code duplication during training and inference.

### Feature Engineering (Task 7)
* **`FamilySize`** (`SibSp + Parch + 1`): Captured passenger group dynamics and evacuation priority.
* **`IsAlone`** (`FamilySize == 1`): Explicitly identified solo travelers with distinct survival probabilities.

### Pipeline Architecture
1. **Numeric Transformer:** `SimpleImputer(median)` → `StandardScaler` — applied to `Age`, `Fare`, `FamilySize`.
2. **Categorical Transformer:** `SimpleImputer(most_frequent)` → `OneHotEncoder(drop='first')` — applied to `Pclass`, `Sex`, `Embarked`, `IsAlone`.
3. **ColumnTransformer** bundles both transformers.
4. **Full Pipeline:** `ColumnTransformer` → `LogisticRegression(max_iter=1000)`.

### Performance & Evaluation

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **0.8156** |
| **F1-Score** | **0.7402** |
| **Precision (Survived)** | 0.81 |
| **Recall (Survived)** | 0.68 |

### Model Persistence
Exported the complete trained pipeline via `joblib.dump()` as `titanic_pipeline_model.joblib`. The saved artifact accepts raw, un-transformed inputs and handles end-to-end imputation, encoding, scaling, and inference.

---

## Credit Card Fraud Detection — Detail

### Dataset
* **Source:** ULB Credit Card Fraud Detection (`284,807` transactions, `31` columns)
* **Target:** `Class` (`0` = Legitimate, `1` = Fraud)
* **Extreme Imbalance:** `284,315` legitimate (99.83%) vs. `492` fraudulent (0.17%)

### Pipeline Steps (Task 9)
1. **Feature Scaling:** Applied `RobustScaler` to `Time` and `Amount` to handle extreme financial distribution outliers.
2. **Data Leakage Mitigation:** Stratified splitting performed *before* oversampling — SMOTE applied strictly to the training partition.
3. **Models Trained:** Baseline Logistic Regression, Class-Weighted (`class_weight='balanced'`), and SMOTE-resampled Logistic Regression.

### Performance & Evaluation

| Metric | Baseline | Class-Weighted (`balanced`) | SMOTE Resampled |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.9992 | 0.9763 | 0.9748 |
| **Precision** | **0.8611** | 0.0632 | 0.0598 |
| **Recall (Fraud Catch Rate)** | 0.6327 | **0.9082** | **0.8980** |
| **F1-Score** | **0.7294** | 0.1182 | 0.1121 |

### Key Takeaways
* **The Accuracy Paradox:** The baseline model scored 99.92% accuracy, yet missed ~37% of all fraudulent transactions.
* **Business Priority (Recall):** Financial institutions accept higher False Positives in exchange for catching >90% of actual fraud via class weighting and SMOTE.
* **Algorithmic vs. Synthetic Resampling:** `class_weight='balanced'` achieved the highest fraud capture rate without the computational overhead of synthesizing thousands of synthetic data vectors.
