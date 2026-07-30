<<<<<<< HEAD
# Titanic Dataset - Machine Learning Pipeline
> **Neurofive ML Track — Tasks 1, 2 & 3**

This repository tracks the complete end-to-end data science lifecycle on the Kaggle Titanic dataset, covering exploratory data analysis, data cleaning, feature encoding, and baseline classification modeling.
=======
# Titanic Dataset - Exploratory Data Analysis & Data Cleaning
> **Neurofive ML Track — Tasks 1 & 2**

This repository contains the exploratory data analysis (EDA), data cleaning pipeline, and visual diagnostics for the classic Kaggle Titanic dataset. It establishes a baseline workflow for data preprocessing before training machine learning models.
>>>>>>> 2954879d445e3e62200cd33f40bcc9ac44d4cbb6

---

## 🛠️ Tech Stack & Toolkit

* **Language:** Python 3.12
* **Environment:** Jupyter Notebook / Virtual Environment (`venv`)
<<<<<<< HEAD
* **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
=======
* **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`
>>>>>>> 2954879d445e3e62200cd33f40bcc9ac44d4cbb6
* **Version Control:** Git & GitHub

---

## 🤖 Model Implementation (Task 3)

### Pipeline Steps
1. **Categorical Encoding:** Converted categorical variables (`Sex`, `Embarked`) using One-Hot Encoding (`pd.get_dummies(drop_first=True)`).
2. **Train-Test Split:** Partitioned the dataset into 80% training and 20% test subsets using stratified sampling to maintain class proportions.
3. **Feature Scaling:** Applied `StandardScaler` to normalize continuous distributions (`Age`, `Fare`) prior to training.
4. **Baseline Algorithm:** Trained a **Logistic Regression** model on scaled features.

<<<<<<< HEAD
### Performance & Evaluation
* **Test Set Accuracy:** **~80%**
* **Evaluation Metric:** Evaluated using `accuracy_score` and `confusion_matrix`. The model effectively leverages `Sex` and `Pclass` to separate binary survival outcomes with balanced true positive/negative ratios.
=======
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
>>>>>>> 2954879d445e3e62200cd33f40bcc9ac44d4cbb6

---

## 🚀 Getting Started Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/notdumbj/neurofive-ml-track.git](https://github.com/notdumbj/neurofive-ml-track.git)
cd neurofive-ml-track
```

### 2. Set Up Virtual Environment
```bash
python -m venv ml_env

# Activate Environment (Git Bash):
source ml_env/bin/activate
<<<<<<< HEAD

# On Windows (CMD):
ml_env\Scripts\activate
```

### 3. Install Dependencies & Launch
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
=======

### 3. Install Dependencies & Launch
pip install pandas numpy matplotlib seaborn jupyter
>>>>>>> 2954879d445e3e62200cd33f40bcc9ac44d4cbb6
jupyter notebook
```

---

## 📂 Project Structure

neurofive-ml-track/
├── .gitignore          # Excludes ml_env, checkpoints, and raw CSVs
<<<<<<< HEAD
├── main.ipynb          # Notebook containing EDA, cleaning, and model training
=======
├── main.ipynb          # Notebook containing Task 1 & Task 2 code
>>>>>>> 2954879d445e3e62200cd33f40bcc9ac44d4cbb6
└── README.md           # Project documentation

---