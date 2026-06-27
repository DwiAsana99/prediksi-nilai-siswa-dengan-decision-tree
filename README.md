# Prediksi Nilai Siswa dengan Decision Tree

Sistem peringatan dini kelulusan siswa menggunakan **Decision Tree Classifier** -- dari data mentah hingga model siap deploy.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DwiAsana99/prediksi-nilai-siswa-dengan-decision-tree/blob/main/Prediksi_Nilai_Siswa_Decision_Tree.ipynb)

## Ringkasan Proyek

| Aspek | Detail |
|-------|--------|
| **Model** | Decision Tree Classifier (scikit-learn) |
| **Bidang** | Pendidikan -- Educational Data Mining |
| **Dataset** | [UCI Student Performance](https://archive.ics.uci.edu/dataset/320) (395 siswa, 32 variabel) |
| **Target** | Lulus (G3 ≥ 10) vs Gagal (G3 < 10) |
| **Akurasi** | **89.9%** pada data uji (setelah pruning) |

## Struktur Notebook

Notebook `Prediksi_Nilai_Siswa_Decision_Tree.ipynb` disusun mengikuti alur ebook secara berurutan:

```
Bab 1  Pengantar           → Konteks prediktif analitik pendidikan
Bab 2  Teori               → Gini impurity, entropy, information gain (+ visualisasi kurva)
Bab 3  Dataset              → Memuat UCI Student Performance, EDA, distribusi target
Bab 4  Preprocessing       → Encoding biner & one-hot, split fitur/target, train-test split
Bab 5  Pemodelan (Baseline) → Pohon tanpa batasan → overfitting (100% train, 85.9% test)
Bab 6  Pruning             → Analisis max_depth, GridSearchCV → model optimal
Bab 7  Evaluasi & Insight  → Pohon final, confusion matrix, feature importance, learning curve
Bab 8  Interpretasi        → Aturan keputusan → peta intervensi pendidikan
       Ekspor Model        → Simpan model & daftar fitur ke file .pkl
       Uji Deployment      → Muat model dari .pkl, prediksi siswa baru (satuan & batch)
```

## Penjelasan Kode

### 1. Import & Setup
```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```
Library utama: `pandas` (manipulasi data), `scikit-learn` (model & evaluasi), `matplotlib`/`seaborn` (visualisasi).

### 2. Memuat Dataset
```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
# Download → ekstrak student-mat.csv → baca dengan separator titik-koma
df = pd.read_csv(f, sep=';')
df["lulus"] = (df["G3"] >= 10).astype(int)  # Target biner: 1=Lulus, 0=Gagal
```
Dataset diunduh langsung dari UCI Repository. Nilai akhir `G3` dikonversi ke label biner dengan ambang 10.

### 3. Preprocessing
```python
# Encoding biner: yes/no → 1/0
for kol in ["schoolsup", "famsup", "paid", ...]:
    df[kol] = df[kol].map({"yes": 1, "no": 0})

# One-Hot Encoding: variabel nominal multi-kategori
df = pd.get_dummies(df, columns=["Mjob", "Fjob", "reason", "guardian"])

# Pisahkan fitur dan target (G3 dibuang untuk hindari data leakage)
X = df.drop(columns=["G3", "lulus"])
y = df["lulus"]
```

### 4. Baseline Model → Overfitting
```python
tree_baseline = DecisionTreeClassifier(random_state=42)
tree_baseline.fit(X_train, y_train)
# Akurasi latih: 100% | Akurasi uji: 85.9% → OVERFITTING
```

### 5. Pruning dengan GridSearchCV
```python
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, 6, 8, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 5, 10, 20],
}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring="f1")
grid.fit(X_train, y_train)
model = grid.best_estimator_
# Hasil: criterion=gini, max_depth=4, min_samples_leaf=10
# Akurasi uji NAIK dari 85.9% → 89.9% dengan pohon lebih sederhana
```

### 6. Ekspor & Deployment
```python
import joblib
# Simpan
joblib.dump(model, 'model_decision_tree_siswa.pkl')
joblib.dump(list(X.columns), 'fitur_kolom.pkl')

# Muat kembali (simulasi deployment)
model_loaded = joblib.load('model_decision_tree_siswa.pkl')
fitur_loaded = joblib.load('fitur_kolom.pkl')

# Prediksi siswa baru
prediksi = model_loaded.predict(input_df)
probabilitas = model_loaded.predict_proba(input_df)
```
Model diekspor ke `.pkl`, lalu dimuat kembali untuk membuktikan portabilitas. Uji deployment mencakup prediksi satuan (4 skenario siswa) dan batch (5 siswa sekaligus).

## Cara Menjalankan

### Opsi 1: Google Colab (Tanpa Instalasi)

1. Klik badge **Open in Colab** di atas, atau buka langsung:
   ```
   https://colab.research.google.com/github/DwiAsana99/prediksi-nilai-siswa-dengan-decision-tree/blob/main/Prediksi_Nilai_Siswa_Decision_Tree.ipynb
   ```
2. Jalankan semua sel secara berurutan (`Runtime` → `Run all`)
3. Dataset otomatis diunduh dari UCI Repository

### Opsi 2: Lokal dengan Virtual Environment

```bash
# 1. Clone repositori
git clone https://github.com/DwiAsana99/prediksi-nilai-siswa-dengan-decision-tree.git
cd prediksi-nilai-siswa-dengan-decision-tree

# 2. Buat dan aktifkan virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 3. Install dependensi
pip install pandas scikit-learn matplotlib seaborn joblib ipykernel

# 4. Jalankan Jupyter Notebook
jupyter notebook Prediksi_Nilai_Siswa_Decision_Tree.ipynb
```

### Opsi 3: VS Code

1. Buka folder proyek di VS Code
2. Install ekstensi **Jupyter** (jika belum)
3. Buka file `.ipynb`
4. Pilih kernel Python dari virtual environment `.venv`
5. Jalankan semua sel (`Run All`)

## Hasil Utama

| Metrik | Sebelum Pruning | Sesudah Pruning |
|--------|:-:|:-:|
| Akurasi latih | 100% | 94.3% |
| **Akurasi uji** | **85.9%** | **89.9%** |
| Kedalaman pohon | 7 | 4 |
| Jumlah daun | 21 | 9 |

**Aturan emas**: Split akar `G2 ≤ 9.5` (nilai periode 2) adalah pemisah paling kuat antara siswa berisiko dan aman.

## Dependensi

| Library | Fungsi |
|---------|--------|
| `pandas` | Manipulasi dan analisis data |
| `scikit-learn` | Model, preprocessing, evaluasi |
| `matplotlib` | Visualisasi grafik dan pohon |
| `seaborn` | Visualisasi statistik |
| `joblib` | Ekspor/impor model (.pkl) |

---

## Tentang Ebook

Notebook ini adalah **pendamping praktik** dari ebook berikut:

| | |
|---|---|
| **Judul** | Prediksi Nilai Siswa dengan Decision Tree |
| **Seri** | Praktik Data Science -- B5 (Tier Pemula) |
| **Penerbit** | **Menata Data Lab** |
| **Tagline** | *Boost Your Insight* -- belajar data science dengan praktik nyata |
| **Edisi** | Pertama, 2026 |

**Menata Data Lab** adalah studio edukasi data science berbahasa Indonesia. Setiap ebook mengikuti formula: **1 Model ML + 1 Bidang Nyata + 1 Dataset Publik = Insight**.

### Peta Seri Ebook

| Tier | Fokus | Contoh Topik |
|------|-------|-------------|
| **Pemula** | Model klasik, dataset bersih | Regresi, Klasifikasi, Clustering, Decision Tree |
| **Menengah** | Ensemble & time series | Random Forest, XGBoost, ARIMA, LightGBM |
| **Mahir** | Deep learning | CNN, Transfer Learning, LSTM, IndoBERT |

### Dapatkan Ebook & Terhubung

| Platform | Link |
|----------|------|
| Semua produk & link | [lynk.id/menatadata](https://lynk.id/menatadata) |
| Instagram | [@menatadata.lab](https://instagram.com/menatadata.lab) |
| TikTok | [@menata.data](https://tiktok.com/@menata.data) |

### Referensi Utama

1. Cortez, P., & Silva, A. (2008). *Using data mining to predict secondary school student performance.* FUBUTEC 2008.
2. Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python.* JMLR, 12, 2825-2830.

---

*© 2026 Menata Data Lab · Seri Praktik Data Science · B5 · Edisi Pertama*
