# Credit Card Fraud Detection — Handling Extreme Class Imbalance
> **Neurofive ML Track — Week 5 (Task 9)**

This project focuses on identifying fraudulent credit card transactions within an extremely imbalanced dataset (99.83% legitimate vs. 0.17% fraud). It evaluates data-level resampling (SMOTE) and algorithmic cost-weighting strategies to maximize fraud catch rates (Recall).

---

## 🛠️ Tech Stack & Toolkit

* **Language:** Python 3.12
* **Environment:** Jupyter Notebook / Virtual Environment (`venv`)
* **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`
* **Dataset:** ULB Credit Card Fraud Detection (284,807 transactions, 30 features)

---

## ⚖️ Class Balance & Preprocessing

* **Target Variable:** `Class` (`0` for Legitimate, `1` for Fraud)
* **Distribution:** `284,315` legitimate vs. `492` fraudulent entries
* **Data Leakage Mitigation:** Stratified splitting was performed *prior* to oversampling. SMOTE was applied strictly to the training partition.
* **Feature Scaling:** Applied `RobustScaler` to `Time` and `Amount` to handle extreme financial distribution outliers.

---

## 📊 Performance & Strategy Comparison

| Metric | 1. Baseline Model | 2. Class-Weighted (`balanced`) | 3. SMOTE Resampled |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.9992 | 0.9763 | 0.9748 |
| **Precision** | **0.8611** | 0.0632 | 0.0598 |
| **Recall (Fraud Catch Rate)** | 0.6327 | **0.9082** | **0.8980** |
| **F1-Score** | **0.7294** | 0.1182 | 0.1121 |

---

## 💡 Key Takeaways

1. **The Accuracy Paradox:** The baseline model scored **99.92% accuracy**, yet missed **~37%** of all fraudulent transactions.
2. **Business Priority (Recall):** Financial institutions accept higher False Positives (sending SMS verification alerts) in exchange for catching **>90%** of actual fraud attempts via Class Weighting and SMOTE.
3. **Algorithmic vs. Synthetic Resampling:** `class_weight='balanced'` achieved the highest fraud capture rate without the computational overhead of synthesizing thousands of synthetic data vectors.