"""
train_model.py
================
Flight Delay Risk Estimator - Training Script

Reproduces the cleaning / feature engineering / modeling steps from
`eda and training.ipynb`, benchmarks Logistic Regression, Random Forest and
XGBoost, automatically selects the best model (highest ROC-AUC), and saves
everything the Streamlit app needs to make live predictions into a single
artifact file: model_artifacts.joblib

Run:
    python train_model.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

DATA_PATH = "data/flights_sample.csv"
ARTIFACT_PATH = "model_artifacts.joblib"
N_ROWS = 150_000  # matches the notebook's initial read

TOP_N_AIRPORTS = 20
TOP_N_DESTS = 20
TOP_N_CARRIERS = 10
ROUTE_SMOOTHING_WEIGHT = 10

FEATURES = [
    "AIRLINE", "ORIGIN", "DEST", "MONTH", "DAY_OF_WEEK",
    "DEP_HOUR", "IS_WEEKEND", "DISTANCE", "DISTANCE_TIER", "ROUTE",
    "IS_PEAK_SEASON",
]
NUM_FEATURES = ["MONTH", "DAY_OF_WEEK", "DEP_HOUR", "DISTANCE", "ROUTE_DELAY_RATE"]
CAT_FEATURES = ["AIRLINE", "ORIGIN", "DEST", "IS_WEEKEND", "IS_PEAK_SEASON", "DISTANCE_TIER"]
DISTANCE_TIER_BINS = [-np.inf, 500, 1500, np.inf]
DISTANCE_TIER_LABELS = ["Short_Haul", "Medium_Haul", "Long_Haul"]
PEAK_MONTHS = [6, 7, 8, 11, 12]
WEEKEND_DAYS = [6, 7]


def build_route_distance_lookup(df_clean: pd.DataFrame):
    """
    Map each unordered airport pair (e.g. 'ATL_BOS', sorted alphabetically so
    direction doesn't matter) to its average historical distance. Used by the
    app to compute distance automatically from origin+destination instead of
    asking the user to type a number that could contradict the real route.
    """
    pair_key = df_clean.apply(lambda r: "_".join(sorted([r["ORIGIN"], r["DEST"]])), axis=1)
    route_distances = df_clean.groupby(pair_key)["DISTANCE"].mean().to_dict()
    avg_distance = float(df_clean["DISTANCE"].mean())
    return route_distances, avg_distance


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=N_ROWS)
    df.columns = df.columns.str.upper().str.strip()

    df = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)].copy()
    df = df.dropna(subset=["ARR_DELAY"]).copy()
    df["IS_DELAYED"] = (df["ARR_DELAY"] >= 15).astype(int)

    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])
    df["MONTH"] = df["FL_DATE"].dt.month
    df["DAY_OF_WEEK"] = df["FL_DATE"].dt.dayofweek + 1
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin(WEEKEND_DAYS).astype(int)
    df["DEP_HOUR"] = (df["CRS_DEP_TIME"] // 100).astype(int).clip(0, 23)
    df["DISTANCE_TIER"] = pd.cut(df["DISTANCE"], bins=DISTANCE_TIER_BINS, labels=DISTANCE_TIER_LABELS)
    df["ROUTE"] = df["ORIGIN"].astype(str) + "_" + df["DEST"].astype(str)
    df["IS_PEAK_SEASON"] = df["MONTH"].isin(PEAK_MONTHS).astype(int)

    top_airports = df["ORIGIN"].value_counts().nlargest(TOP_N_AIRPORTS).index
    top_dests = df["DEST"].value_counts().nlargest(TOP_N_DESTS).index
    top_carriers = df["AIRLINE"].value_counts().nlargest(TOP_N_CARRIERS).index

    df_clean = df[
        df["ORIGIN"].isin(top_airports)
        & df["DEST"].isin(top_dests)
        & df["AIRLINE"].isin(top_carriers)
    ].copy()
    df_clean = df_clean[FEATURES + ["IS_DELAYED"]].reset_index(drop=True)

    meta = {
        "top_airports": sorted(top_airports.tolist()),
        "top_destinations": sorted(top_dests.tolist()),
        "top_carriers": sorted(top_carriers.tolist()),
    }
    return df_clean, meta


def add_route_delay_rate(X_train, X_test, y_train):
    global_prior = y_train.mean()
    train_stats = y_train.groupby(X_train["ROUTE"]).agg(["count", "mean"])
    train_rates = (
        (train_stats["count"] * train_stats["mean"] + ROUTE_SMOOTHING_WEIGHT * global_prior)
        / (train_stats["count"] + ROUTE_SMOOTHING_WEIGHT)
    )
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["ROUTE_DELAY_RATE"] = X_train["ROUTE"].map(train_rates).fillna(global_prior)
    X_test["ROUTE_DELAY_RATE"] = X_test["ROUTE"].map(train_rates).fillna(global_prior)
    X_train = X_train.drop(columns=["ROUTE"])
    X_test = X_test.drop(columns=["ROUTE"])
    return X_train, X_test, train_rates.to_dict(), float(global_prior)


def build_preprocessor():
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", num_transformer, NUM_FEATURES),
        ("cat", cat_transformer, CAT_FEATURES),
    ])


def main():
    print("Loading and cleaning data...")
    df_clean, meta = load_and_clean(DATA_PATH)
    print(f"Cleaned dataset shape: {df_clean.shape}")

    route_distances, avg_distance = build_route_distance_lookup(df_clean)
    print(f"Built distance lookup for {len(route_distances)} unordered airport pairs "
          f"(fallback average: {avg_distance:.0f} miles)")

    X = df_clean.drop(columns=["IS_DELAYED"])
    y = df_clean["IS_DELAYED"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test, route_rates, global_prior = add_route_delay_rate(X_train, X_test, y_train)

    preprocessor = build_preprocessor()
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced", n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42,
            scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
            eval_metric="logloss", n_jobs=-1,
        ),
    }

    results_list = []
    fitted_pipelines = {}
    for name, clf in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        results_list.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1-Score": round(f1_score(y_test, y_pred), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4),
        })

    results_df = pd.DataFrame(results_list)
    print("\n=== Model Benchmark Results ===")
    print(results_df.to_string(index=False))

    best_row = results_df.sort_values("ROC-AUC", ascending=False).iloc[0]
    best_name = best_row["Model"]
    print(f"\nSelected champion model (highest ROC-AUC): {best_name}")

    artifact = {
        "pipeline": fitted_pipelines[best_name],
        "best_model_name": best_name,
        "route_rates": route_rates,
        "route_distances": route_distances,
        "avg_distance": avg_distance,
        "global_prior": global_prior,
        "top_airports": meta["top_airports"],
        "top_destinations": meta["top_destinations"],
        "top_carriers": meta["top_carriers"],
        "num_features": NUM_FEATURES,
        "cat_features": CAT_FEATURES,
        "distance_tier_bins": DISTANCE_TIER_BINS,
        "distance_tier_labels": DISTANCE_TIER_LABELS,
        "peak_months": PEAK_MONTHS,
        "weekend_days": WEEKEND_DAYS,
        "results_df": results_df,
        "base_delay_rate": global_prior,
    }
    joblib.dump(artifact, ARTIFACT_PATH)
    print(f"\nSaved deployable artifact -> {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
