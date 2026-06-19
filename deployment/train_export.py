"""
Training & Export Script - Prediksi Churn Pelanggan
=====================================================
Mereplikasi pipeline preprocessing dari notebook Week 3 (Preprocessing.ipynb),
melatih model terbaik (Random Forest hasil tuning Week 4), lalu menyimpan
seluruh artefak (model + scaler + metadata fitur) ke dalam satu file .joblib
agar dapat dimuat oleh aplikasi Streamlit.

Jalankan dari root proyek:
    python deployment/train_export.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Path ---
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "dataset" / "Sales - Marketing customer dataset.csv"
MODEL_PATH = Path(__file__).resolve().parent / "churn_model.joblib"

# Kolom yang dipakai untuk feature engineering tanggal
DATE_COLS = ["signup_date", "last_purchase_date"]
# Kolom yang di-drop (ID / kardinalitas tinggi / banyak missing) - sama dgn notebook
DROP_COLS = ["customer_id", "coupon_code", "country", "city"]
# Kategorikal yang di-one-hot-encode (kardinalitas rendah)
CAT_TO_ENCODE = [
    "gender",
    "acquisition_channel",
    "device_type",
    "subscription_type",
    "payment_method",
]
TARGET = "churn"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing identik dengan notebook: imputasi, tanggal->numerik, one-hot."""
    df = df.copy()

    # 1. Missing value: modus (kategorikal) & median (numerik)
    df["gender"] = df["gender"].fillna(df["gender"].mode()[0])
    df["age"] = df["age"].fillna(df["age"].median())
    df["total_spent"] = df["total_spent"].fillna(df["total_spent"].median())
    df["satisfaction_score"] = df["satisfaction_score"].fillna(
        df["satisfaction_score"].median()
    )

    # 2. Outlier capping (IQR) untuk fitur yang sama spt notebook
    for col in ["age", "total_spent", "satisfaction_score"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = np.where(df[col] < low, low, df[col])
        df[col] = np.where(df[col] > high, high, df[col])

    # 3. Feature engineering tanggal -> jumlah hari
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"])
    ref_date = df["last_purchase_date"].max()
    df["days_since_signup"] = (ref_date - df["signup_date"]).dt.days
    df["days_since_last_purchase"] = (ref_date - df["last_purchase_date"]).dt.days

    # 4. One-hot encoding (drop_first=True spt notebook)
    df = df.drop(columns=DATE_COLS)
    df = pd.get_dummies(df, columns=CAT_TO_ENCODE, drop_first=True, dtype=int)

    return df


def main() -> None:
    print(f"Memuat dataset dari: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    df_encoded = build_features(df)

    # Pisahkan X dan y (drop kolom tidak relevan)
    X = df_encoded.drop(columns=[TARGET] + DROP_COLS, errors="ignore")
    y = df_encoded[TARGET]
    feature_columns = X.columns.tolist()

    # Train-test split (proporsi sama dengan notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling: fit hanya pada train (mencegah data leakage)
    # Kolom biner (0/1) hasil one-hot tidak ikut di-scale
    features_to_scale = [
        c
        for c in X_train.select_dtypes(include=np.number).columns
        if not (X_train[c].nunique() == 2 and X_train[c].min() == 0 and X_train[c].max() == 1)
    ]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
    X_test_scaled[features_to_scale] = scaler.transform(X_test[features_to_scale])

    # Model terbaik: Random Forest dengan best params hasil tuning Week 4
    # (RandomizedSearchCV: n_estimators=100, min_samples_split=5,
    #  min_samples_leaf=1, max_depth=None)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluasi singkat
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    y_pred = model.predict(X_test_scaled)
    print("\n=== Evaluasi Model (data uji) ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred, zero_division=0):.4f}")

    # Simpan bundle: model + scaler + metadata fitur
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "features_to_scale": features_to_scale,
        "cat_to_encode": CAT_TO_ENCODE,
        "drop_cols": DROP_COLS,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nModel tersimpan di: {MODEL_PATH}")


if __name__ == "__main__":
    main()
