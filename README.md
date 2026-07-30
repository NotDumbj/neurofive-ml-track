# Titanic Dataset - Machine Learning Pipeline
> **Neurofive ML Track — Tasks 1, 2 & 3**

This repository tracks the complete end-to-end data science lifecycle on the Kaggle Titanic dataset, covering exploratory data analysis, data cleaning, feature encoding, and baseline classification modeling.

---

## 🛠️ Tech Stack & Toolkit

* **Language:** Python 3.12
* **Environment:** Jupyter Notebook / Virtual Environment (`venv`)
* **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
* **Version Control:** Git & GitHub

---

## 🤖 Model Implementation (Task 3)

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
git clone [https://github.com/notdumbj/neurofive-ml-track.git](https://github.com/notdumbj/neurofive-ml-track.git)
cd neurofive-ml-track
```

### 2. Set Up Virtual Environment
```bash
python -m venv ml_env

# On Windows (Git Bash):
source ml_env/bin/activate

# On Windows (CMD):
ml_env\Scripts\activate
```

### 3. Install Dependencies & Launch
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

---

## 📂 Project Structure

neurofive-ml-track/
├── .gitignore          # Excludes ml_env, checkpoints, and raw CSVs
├── main.ipynb          # Notebook containing EDA, cleaning, and model training
└── README.md           # Project documentation

---