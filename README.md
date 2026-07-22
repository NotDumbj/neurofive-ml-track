# Titanic Dataset - Exploratory Data Analysis & Data Cleaning
> **Neurofive ML Track — Tasks 1 & 2**

This repository contains the exploratory data analysis (EDA), data cleaning pipeline, and visual diagnostics for the classic Kaggle Titanic dataset. It establishes a baseline workflow for data preprocessing before training machine learning models.

---

## 🛠️ Tech Stack & Toolkit

* **Language:** Python 3.12
* **Environment:** Jupyter Notebook / Virtual Environment (`venv`)
* **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Version Control:** Git & GitHub

---

## 📊 Dataset Overview

* **Source:** Kaggle (*Titanic - Machine Learning from Disaster*)
* **Scale:** 891 rows, 12 columns
* **Target Feature:** `Survived` (Binary classification: `0` = No, `1` = Yes)

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

## 🚀 Getting Started Locally

### 1. Clone the Repository
git clone https://github.com/notdumbj/neurofive-ml-track.git
cd neurofive-ml-track

### 2. Set Up Virtual Environment
python -m venv ml_env

# Activate Environment (Git Bash):
source ml_env/bin/activate

### 3. Install Dependencies & Launch
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook

---

## 📂 Project Structure

neurofive-ml-track/
├── .gitignore          # Excludes ml_env, checkpoints, and raw CSVs
├── main.ipynb          # Notebook containing Task 1 & Task 2 code
└── README.md           # Project documentation
