# Titanic Dataset - Exploratory Data Analysis (EDA)
> **Neurofive ML Track — Task 1**

An initial data profiling and exploratory analysis of the classic Kaggle Titanic dataset. This repository establishes the baseline environment, data loading patterns, and preliminary findings before applying machine learning algorithms.

---

## 🛠️ Tech Stack & Toolkit

* **Language:** Python 3.12
* **Environment:** Jupyter Notebook / Virtual Environment (`venv`)
* **Libraries:** `pandas`, `numpy`
* **Version Control:** Git & GitHub

---

## 📊 Dataset Overview

* **Source:** Kaggle (*Titanic - Machine Learning from Disaster*)
* **Scale:** 891 rows, 12 columns
* **Target Feature:** `Survived` (Binary classification: `0` = No, `1` = Yes)

---

## 🔍 Key EDA Findings ("Data Story")

* **Class Distribution & Survival Rate:** The average survival rate across the training set is **~38.4%**.
* **Missing Value Profile:**
  * `Cabin`: Highly sparse; missing the vast majority of entries (~77%). Requires dropping or deck feature extraction.
  * `Age`: Missing ~20% of values. Requires an imputation strategy (e.g., median by Pclass/Sex) prior to modeling.
  * `Embarked`: Missing only 2 values, which can easily be imputed using the mode.
* **Feature Skewness:** The `Fare` feature exhibits extreme right-skewness, ranging from `$0.00` to a maximum outlier of `$512.33`, reflecting distinct class/ticket tier distributions.

---

## 🚀 Getting Started Locally

### 1. Clone the Repository
git clone https://github.com/notdumbj/neurofive-ml-track.git
cd neurofive-ml-track

### 2. Set Up Virtual Environment
python -m venv ml_env

# On Windows (Git Bash):
source ml_env/bin/activate
# On Linux/macOS:
source ml_env/bin/activate
# On Windows (CMD):
ml_env\Scripts\activate

### 3. Install Dependencies & Launch
pip install pandas numpy jupyter
jupyter notebook

---

## 📂 Project Structure

neurofive-ml-track/
├── .gitignore          # Excludes ml_env, checkpoints, and raw CSVs
├── main.ipynb          # Primary EDA notebook with profiling output
└── README.md           # Project documentation
