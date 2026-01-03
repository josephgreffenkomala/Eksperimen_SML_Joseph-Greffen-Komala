"""
Automation Script for Iris Dataset Preprocessing
Author: AsahDicoding
Description: Script untuk melakukan preprocessing data Iris secara otomatis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os


def load_data(filepath):
    """
    Load dataset dari file CSV
    
    Args:
        filepath (str): Path ke file CSV
        
    Returns:
        pd.DataFrame: Dataset yang telah dimuat
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def check_missing_values(df):
    """
    Cek dan handle missing values
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        pd.DataFrame: Dataset setelah handling missing values
    """
    print("\nChecking missing values...")
    missing_count = df.isnull().sum().sum()
    
    if missing_count > 0:
        print(f"Found {missing_count} missing values. Handling...")
        # Untuk numerik: isi dengan median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Untuk kategorikal: isi dengan modus
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        print("No missing values found.")
    
    return df


def remove_duplicates(df):
    """
    Hapus data duplikat
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        pd.DataFrame: Dataset tanpa duplikat
    """
    print("\nChecking duplicates...")
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed_rows = initial_rows - len(df)
    
    if removed_rows > 0:
        print(f"Removed {removed_rows} duplicate rows.")
    else:
        print("No duplicates found.")
    
    return df


def detect_outliers(df, feature_cols):
    """
    Deteksi outlier menggunakan IQR method
    
    Args:
        df (pd.DataFrame): Dataset
        feature_cols (list): Nama kolom fitur
        
    Returns:
        pd.DataFrame: Dataset dengan informasi outlier
    """
    print("\nDetecting outliers using IQR method...")
    outlier_info = {}
    
    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = len(outliers)
        
        if len(outliers) > 0:
            print(f"  {col}: {len(outliers)} outliers detected")
    
    return df, outlier_info


def split_data(df, target_col='species', test_size=0.2, random_state=42):
    """
    Split data menjadi training dan testing set
    
    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Nama kolom target
        test_size (float): Proporsi data testing
        random_state (int): Random state untuk reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print(f"\nSplitting data into train and test sets (test_size={test_size})...")
    
    # Pisahkan fitur dan target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, scaler_path=None):
    """
    Standarisasi fitur menggunakan StandardScaler
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        scaler_path (str): Path untuk menyimpan scaler
        
    Returns:
        tuple: X_train_scaled, X_test_scaled, scaler
    """
    print("\nScaling features using StandardScaler...")
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Simpan scaler jika path diberikan
    if scaler_path:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_path}")
    
    print("Features scaled successfully.")
    return X_train_scaled, X_test_scaled, scaler


def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir):
    """
    Simpan data yang sudah dipreprocess
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        y_train (pd.Series): Training target
        y_test (pd.Series): Testing target
        output_dir (str): Direktori output
    """
    print(f"\nSaving preprocessed data to {output_dir}...")
    
    # Buat direktori jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Gabungkan X dan y
    train_preprocessed = X_train.copy()
    train_preprocessed['species'] = y_train.values
    
    test_preprocessed = X_test.copy()
    test_preprocessed['species'] = y_test.values
    
    # Simpan ke CSV
    train_path = os.path.join(output_dir, 'iris_train_preprocessed.csv')
    test_path = os.path.join(output_dir, 'iris_test_preprocessed.csv')
    
    train_preprocessed.to_csv(train_path, index=False)
    test_preprocessed.to_csv(test_path, index=False)
    
    print(f"Training data saved to {train_path}")
    print(f"Testing data saved to {test_path}")


def preprocess_pipeline(raw_data_path, output_dir, target_col='species'):
    """
    Pipeline lengkap untuk preprocessing data
    
    Args:
        raw_data_path (str): Path ke raw data
        output_dir (str): Direktori untuk menyimpan hasil
        target_col (str): Nama kolom target
        
    Returns:
        dict: Dictionary berisi semua hasil preprocessing
    """
    print("="*60)
    print("Starting Preprocessing Pipeline")
    print("="*60)
    
    # 1. Load data
    df = load_data(raw_data_path)
    
    # 2. Check missing values
    df = check_missing_values(df)
    
    # 3. Remove duplicates
    df = remove_duplicates(df)
    
    # 4. Detect outliers (tidak dihapus, hanya deteksi)
    feature_cols = df.drop(columns=[target_col]).columns.tolist()
    df, outlier_info = detect_outliers(df, feature_cols)
    
    # 5. Split data
    X_train, X_test, y_train, y_test = split_data(df, target_col=target_col)
    
    # 6. Scale features
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, scaler_path=scaler_path
    )
    
    # 7. Save preprocessed data
    save_preprocessed_data(X_train_scaled, X_test_scaled, y_train, y_test, output_dir)
    
    print("\n" + "="*60)
    print("Preprocessing Pipeline Completed Successfully!")
    print("="*60)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'outlier_info': outlier_info
    }


if __name__ == "__main__":
    # Konfigurasi path
    raw_data_path = "iris_raw.csv"
    output_dir = "."
    
    # Jalankan pipeline
    results = preprocess_pipeline(raw_data_path, output_dir)
    
    print("\nPreprocessing Summary:")
    print(f"- Training samples: {len(results['X_train'])}")
    print(f"- Testing samples: {len(results['X_test'])}")
    print(f"- Number of features: {results['X_train'].shape[1]}")
    print(f"- Target classes: {results['y_train'].unique()}")
