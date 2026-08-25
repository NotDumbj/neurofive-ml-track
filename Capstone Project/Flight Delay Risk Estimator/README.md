# ✈️ Flight Delay Risk Estimator

An end-to-end machine learning project that estimates the probability a US
domestic flight will arrive **15+ minutes late**, based on airline, route,
scheduled time, and seasonality. Built as a capstone project for the
Neurofive Solutions Machine Learning Fundamentals course.

## Problem Statement

Flight delays cost travelers time and money and cost airlines and airports
operational efficiency. Most delay information available to a passenger
before booking is a single historical on-time percentage for a carrier, not
a personalized risk estimate for their specific route, date, and time. This
project builds a lightweight risk-scoring model that a traveler (or a travel
app) could use at booking time to flag higher-risk itineraries.

## Approach

1. **Data**: A sample of US domestic flight records (150,000 rows, 32
   columns) covering flight schedules, actual times, delay reason codes,
   and cancellation/diversion flags.
2. **Cleaning**: Removed cancelled and diverted flights (delay isn't
   defined for these) and rows missing an arrival delay value.
3. **Target**: `IS_DELAYED` = 1 if arrival delay ≥ 15 minutes, matching the
   FAA's official definition of a delayed flight.
4. **Feature engineering**:
   - Calendar features: month, day of week, weekend flag, peak-season flag
     (Jun–Aug, Nov–Dec)
   - Scheduled departure hour, extracted from `CRS_DEP_TIME`
   - Distance tier (Short/Medium/Long haul) via binning
   - **Smoothed route delay rate**: a target-encoded feature giving each
     `ORIGIN → DEST` route its historical delay rate, shrunk toward the
     global average for low-volume routes to avoid overfitting on sparse
     routes
   - Filtered to the top 20 origin airports, top 20 destination airports,
     and top 10 carriers by volume to control categorical cardinality
5. **Modeling**: Benchmarked three classifiers inside a single
   `scikit-learn` `Pipeline` (median/most-frequent imputation → scaling /
   one-hot encoding → classifier), with class weighting to handle the
   ~82/18 class imbalance:
   - Logistic Regression (baseline, `class_weight='balanced'`)
   - Random Forest (`class_weight='balanced'`)
   - XGBoost (`scale_pos_weight` tuned to the imbalance ratio)
6. **Evaluation**: Stratified 80/20 train/test split; compared Accuracy,
   Precision, Recall, F1, and ROC-AUC.
7. **Deployment**: Champion model, route-rate lookup table, and dropdown
   option lists are bundled into a single `joblib` artifact and served
   through a Streamlit app for live predictions.

## Results

After cleaning and filtering to the top airports/carriers, the modeling
dataset was **26,343 flights**, with a base delay rate of **18.4%**
(class balance 81.5% on-time / 18.4% delayed).

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|---------:|----------:|-------:|---------:|--------:|
| Logistic Regression | 0.5931   | 0.2367    | 0.5422 | 0.3296   | 0.6035  |
| Random Forest        | 0.6367   | 0.2500    | 0.4846 | 0.3298   | 0.6140  |
| XGBoost              | 0.6403   | 0.2531    | 0.4866 | 0.3330   | 0.6094  |

`train_model.py` automatically selects the champion model by **highest
ROC-AUC** (Random Forest, 0.6140), since ROC-AUC best reflects a model's
ability to *rank* flights by risk — which is how this tool is meant to be
used (as a probability/risk score, not a hard yes/no call). XGBoost is a
close second and actually leads on accuracy, precision, recall, and F1,
which makes for a good model-selection trade-off discussion.

**Honest take on model quality**: ROC-AUC in the 0.60–0.62 range is modest
— these models are meaningfully better than a coin flip but far from a
precise predictor. That's expected given the available features: flight
delays are driven heavily by same-day weather, air traffic control
conditions, and mechanical issues, none of which are in this dataset. This
tool is best framed as an early risk-flagging signal, not a guarantee.

## Project Structure

```
Flight Delay Risk Estimator/
├── data/
│   └── flights_sample.csv       # raw input data (not committed if large)
├── eda and training.ipynb       # exploratory analysis + model benchmarking
├── train_model.py               # reproducible training script -> saves model_artifacts.joblib
├── app.py                       # Streamlit app for live predictions
├── requirements.txt
└── README.md
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (reads data/flights_sample.csv, writes model_artifacts.joblib)
python train_model.py

# 3. Launch the app
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`),
pick an airline, route, date, and departure hour, and get a live delay-risk
estimate.

## Limitations & Future Work

- No weather, air traffic control, or mechanical data — the single biggest
  lever for improving accuracy would be joining in a weather API by
  route/date.
- Trained only on the top 20 airports / 10 carriers by volume; smaller
  regional airports fall back to the global average delay rate.
- Route delay rate is computed once at training time; in production it
  would need periodic recomputation as travel patterns shift.
- Next steps: hyperparameter tuning (grid/Bayesian search), richer temporal
  features (holiday calendar, time-of-day congestion), and probability
  calibration (e.g. `CalibratedClassifierCV`) so the output percentages are
  better calibrated to true frequencies.

## Business Value

Even a modest risk signal (AUC ~0.61) is more informative than no signal at
all. Embedded into a booking flow, this kind of estimator could nudge
travelers toward earlier connections/buffer time, or feed into a travel
insurance pricing model — a small edge that compounds across millions of
bookings.
