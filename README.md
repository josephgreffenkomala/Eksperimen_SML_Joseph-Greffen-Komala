# Eksperimen_SML_AsahDicoding

Repository untuk submission Machine Learning Systems and Operations (MLOps)

## 📁 Struktur Folder

```
Eksperimen_SML_AsahDicoding/
├── .github/
│   └── workflows/
│       └── preprocessing.yml          # GitHub Actions workflow untuk preprocessing otomatis
├── preprocessing/
│   ├── Eksperimen_AsahDicoding.ipynb  # Notebook eksperimen manual
│   ├── automate_AsahDicoding.py       # Script preprocessing otomatis
│   ├── iris_raw.csv                   # Dataset mentah
│   ├── iris_train_preprocessed.csv    # Dataset training hasil preprocessing
│   ├── iris_test_preprocessed.csv     # Dataset testing hasil preprocessing
│   └── scaler.pkl                     # Scaler yang telah di-fit
```

## 📊 Dataset

**Nama**: Iris Dataset  
**Sumber**: UCI Machine Learning Repository  
**URL**: https://archive.ics.uci.edu/ml/datasets/iris

**Deskripsi**: Dataset klasifikasi bunga iris dengan 3 spesies (setosa, versicolor, virginica) berdasarkan 4 fitur pengukuran.

**Karakteristik**:
- Jumlah sampel: 150
- Jumlah fitur: 4 (sepal length, sepal width, petal length, petal width)
- Target: 3 kelas (species)

## 🔬 Kriteria 1: Eksperimen Dataset

### ✅ Basic (2 pts)
- [x] Melakukan data loading pada notebook
- [x] Melakukan EDA pada notebook
- [x] Melakukan preprocessing pada notebook

### ✅ Skilled (3 pts)
- [x] Membuat file `automate_AsahDicoding.py` untuk preprocessing otomatis
- [x] Konversi dari eksperimen notebook ke automation script

### ✅ Advance (4 pts)
- [x] Membuat GitHub Actions workflow untuk preprocessing otomatis
- [x] Workflow menghasilkan dataset yang telah diproses
- [x] Auto-commit hasil preprocessing ke repository

## 🚀 Cara Menggunakan

### Manual Preprocessing (Notebook)
```bash
# Buka notebook di Jupyter/VS Code
jupyter notebook preprocessing/Eksperimen_AsahDicoding.ipynb
```

### Automated Preprocessing (Script)
```bash
cd preprocessing
python automate_AsahDicoding.py
```

### GitHub Actions (Otomatis)
Workflow akan berjalan otomatis ketika:
- Push ke branch `main` yang mengubah file dataset atau script preprocessing
- Manual trigger melalui GitHub Actions UI

## 📝 Tahapan Preprocessing

1. **Data Loading**: Load dataset dari CSV
2. **Missing Values Check**: Deteksi dan handle missing values
3. **Duplicate Removal**: Hapus data duplikat
4. **Outlier Detection**: Deteksi outlier menggunakan IQR method
5. **Data Splitting**: Split menjadi training (80%) dan testing (20%)
6. **Feature Scaling**: Standarisasi fitur menggunakan StandardScaler
7. **Save Results**: Simpan dataset yang telah dipreprocess

## 🛠️ Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## 👤 Author

**AsahDicoding**  
Submission: Machine Learning Systems and Operations

## 📅 Created

January 2, 2026
