# Titanic Dataset - Machine Learning Pipeline
> **Neurofive ML Track — Tasks 1, 2 & 3**

This repository tracks the complete end-to-end data science lifecycle on the Kaggle Titanic dataset, covering exploratory data analysis, data cleaning, feature encoding, baseline classification modeling, and performance evaluation.

---

## 🛠️ Tech Stack & Toolkit

* **Language:** Python 3.12
* **Environment:** Jupyter Notebook / Virtual Environment (`venv`)
* **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
* **Version Control:** Git & GitHub

---

## 🧹 Data Cleaning & Preprocessing Strategy (Task 2)

1. **`Age` Imputation:** Missing values (~20%) were filled using the column **median**. Median was preferred over mean because the age distribution exhibits slight right-skewness, making the median more resilient to extreme values.
2. **`Embarked` Imputation:** Missing values (2 records) were filled using the **mode** (`'S'`), preserving categorical integrity without altering class proportions.
3. **`Cabin` Removal:** Over 77% of `Cabin` entries were missing. The feature was dropped entirely to prevent introducing synthetic noise.
4. **Outlier Identification:** Boxplot profiling on `Fare` revealed severe right-skewness and outliers ranging beyond `$500`. These entries reflect high-tier first-class suites and combined family tickets rather than bad data.

---

## 📈 Key Visual Insights ("Data Story")

* **Primary Survival Driver (`Sex`):** Visualizing survival rate by gender reveals that females had a **~74%** survival probability compared to **~19%** for males, demonstrating the strong impact of the "women and children first" evacuation protocol.
* **Socioeconomic Influence (`Pclass`):** Passenger class serves as a strong secondary predictor. First-class passengers achieved higher survival rates due to proximity to the upper deck and priority lifeboat access.
* **Correlation Highlights:** Strong negative correlation exists between `Pclass` and `Fare`, confirming that higher fare values strongly map to tier 1 accommodations.

---

## 🤖 Model Implementation & Evaluation (Task 3)

### Pipeline Steps
1. **Categorical Encoding:** Converted categorical variables (`Sex`, `Embarked`) using One-Hot Encoding (`pd.get_dummies(drop_first=True)`).
2. **Train-Test Split:** Partitioned the dataset into 80% training and 20% test subsets using stratified sampling to maintain class proportions.
3. **Feature Scaling:** Applied `StandardScaler` to normalize continuous distributions (`Age`, `Fare`) prior to training.
4. **Baseline Algorithm:** Trained a **Logistic Regression** model on scaled features.

### Performance & Evaluation
* **Test Set Accuracy:** **~80%**
* **Evaluation Metric:** Evaluated using `accuracy_score` and `confusion_matrix`. The model effectively leverages `Sex` and `Pclass` to separate binary survival outcomes with balanced true positive/negative ratios.

---

## 🚀 Getting Started Locally

### 1. Clone the Repository
```bash
git clone https://github.com/notdumbj/neurofive-ml-track.git
cd neurofive-ml-track
```

### 2. Set Up Virtual Environment
```bash
python -m venv ml_env

# On Windows (Git Bash):
source ml_env/Scripts/activate

# On Windows (CMD):
ml_env\Scripts\activate

# On Linux/macOS:
source ml_env/bin/activate
```
### 3. Install Dependencies & Launch
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook
```
---

## 📂 Project Structure

```
neurofive-ml-track/
├── .gitignore          # Excludes ml_env, checkpoints, and raw CSVs
├── main.ipynb          # Notebook containing EDA, cleaning, and model training
└── README.md           # Project documentation
```
