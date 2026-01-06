# ============================================
# Iris Dataset Preprocessing Script
# ============================================

# 1. Import Library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================
# 2. Load Dataset
# ============================================
DATA_PATH = "../iris_raw.csv"
OUTPUT_DIR = "iris_preprocessing"

df = pd.read_csv(DATA_PATH)

print("=== SEBELUM PREPROCESSING ===")
print(f"Jumlah data           : {df.shape}")
print(f"Total missing values  : {df.isnull().sum().sum()}")
print(f"Total duplikat        : {df.duplicated().sum()}")

# ============================================
# 3. Handling Duplicates
# ============================================
df_clean = df.drop_duplicates()

print("\n=== SETELAH HAPUS DUPLIKAT ===")
print(f"Jumlah data           : {df_clean.shape}")
print(f"Total duplikat        : {df_clean.duplicated().sum()}")

# ============================================
# 4. Feature Selection
# ============================================
FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]
TARGET = "species"

X = df_clean[FEATURES]
y = df_clean[TARGET]

print("\n=== FEATURE & TARGET ===")
print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")

# ============================================
# 5. Train-Test Split
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n=== DATA SPLIT ===")
print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y_train : {y_train.shape}")
print(f"y_test  : {y_test.shape}")

# ============================================
# 6. Feature Scaling (Standardization)
# ============================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Kembalikan ke DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=FEATURES)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=FEATURES)

print("\n=== CONTOH DATA SETELAH SCALING ===")
print(X_train_scaled.head())

# ============================================
# 7. Save Preprocessed Data
# ============================================
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_preprocessed = X_train_scaled.copy()
train_preprocessed[TARGET] = y_train.values

test_preprocessed = X_test_scaled.copy()
test_preprocessed[TARGET] = y_test.values

train_preprocessed.to_csv(
    f"{OUTPUT_DIR}/iris_train_preprocessed.csv",
    index=False
)
test_preprocessed.to_csv(
    f"{OUTPUT_DIR}/iris_test_preprocessed.csv",
    index=False
)

import joblib

# Simpan scaler untuk digunakan saat inference
joblib.dump(scaler, 'scaler.pkl')

print("Scaler berhasil disimpan sebagai 'scaler.pkl'")

print("\n" + "=" * 50)
print("PREPROCESSING SELESAI ✅")
print("=" * 50)
print("File tersimpan:")
print("- iris_preprocessing/iris_train_preprocessed.csv")
print("- iris_preprocessing/iris_test_preprocessed.csv")
