# Titanic Survival Prediction App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://titanic-ml-track.streamlit.app)

Interactive Streamlit web application that serves the trained Titanic sklearn Pipeline model (from Task 7) as a user-facing prediction tool. Users input passenger details through an intuitive form and receive a survival prediction with probability score.

> **Neurofive ML Track — Week 5 (Task 10)**

## Features

- Input passenger class, sex, age, fare, embarkation port, and family details
- Automatic feature engineering (`FamilySize`, `IsAlone`) matching the training pipeline
- Survival prediction with probability percentage via the saved sklearn Pipeline artifact

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or try it live: **https://titanic-ml-track.streamlit.app**
