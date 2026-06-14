# Sistem Rekomendasi Produk Manufaktur

Dashboard interaktif untuk rekomendasi produk menggunakan **Hybrid Filtering** (CF + CBF + SVD).

## 📁 Struktur Proyek

```
TugasProjekPakRon/
├── data/
│   ├── online_retail_II.csv              # Dataset mentah
│   ├── transactions_cleaned.csv           # Data bersih
│   ├── user_item_matrix.csv              # User-item matrix
│   ├── user_item_ratings.csv             # Rating data (SVD)
│   └── products_cleaned.csv              # Katalog produk
├── notebooks/
│   ├── 01_EDA.ipynb                      # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb            # Data cleaning & preparation
│   ├── 03_collaborative_filtering.ipynb  # CF model
│   ├── 04_content_based.ipynb            # CBF model
│   ├── 05_svd_matrix_factorization.ipynb # SVD model
│   └── 06_hybrid_evaluation.ipynb        # Hybrid system & evaluation
├── models/
│   ├── user_similarity_matrix.csv        # CF model
│   ├── tfidf_vectorizer.pkl              # TF-IDF (CBF)
│   ├── tfidf_matrix.pkl                  # TF-IDF matrix
│   ├── product_similarity_matrix.csv     # CBF model
│   ├── svd_model.pkl                     # SVD model
│   └── *_metrics_summary.csv             # Evaluation results
├── app.py                                # Streamlit dashboard
├── recommender.py                        # Core recommendation logic
├── requirements.txt                      # Dependencies
└── README.md                             # Panduan ini
```

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Notebooks (Berurutan)

```bash
# Harus dijalankan urut untuk generate model
cd notebooks
jupyter notebook

# Jalankan notebooks dalam urutan:
# 1. 01_EDA.ipynb              → Eksplorasi data
# 2. 02_preprocessing.ipynb    → Bersihkan data, buat matrices
# 3. 03_collaborative_filtering.ipynb  → Train CF, save model
# 4. 04_content_based.ipynb    → Train CBF, save model
# 5. 05_svd_matrix_factorization.ipynb → Train SVD, save model
# 6. 06_hybrid_evaluation.ipynb → Gabung semua, evaluasi
```

### 3. Jalankan Dashboard Streamlit

```bash
streamlit run app.py
```

Dashboard akan membuka di browser: `http://localhost:8501`

## 🎯 Fitur Dashboard

### Sidebar (Panel Kiri)
- **Pilih Customer ID** - Dropdown untuk memilih customer
- **Bobot Algoritma** - Slider untuk mengatur CF/CBF/SVD weights
- **Generate Recommendations** - Tombol untuk generate rekomendasi

### Main Panel (4 Tabs)

**Tab 1: Rekomendasi**
- Top 10 produk yang direkomendasikan
- Skor hybrid dari setiap produk
- Download hasil (CSV)

**Tab 2: Riwayat Pembelian**
- Daftar produk yang pernah dibeli customer
- Statistik: total produk, quantity, revenue
- Tabel detail dengan price dan revenue

**Tab 3: Perbandingan Algoritma**
- Top 5 rekomendasi dari CF
- Top 5 rekomendasi dari CBF
- Top 5 rekomendasi dari SVD

**Tab 4: Visualisasi**
- Bar chart: Score setiap rekomendasi
- Scatter plot: Quantity vs Price dari history
- Pie chart: Bobot algoritma yang digunakan

## 📊 Algoritma yang Digunakan

### 1. Collaborative Filtering (CF)
- **Metode**: User-Based, Cosine Similarity
- **Prinsip**: "Buyer mirip denganmu suka produk yang sama"
- **Kelebihan**: Tidak butuh fitur produk
- **Kekurangan**: Cold start problem

### 2. Content-Based Filtering (CBF)
- **Metode**: TF-IDF + Cosine Similarity
- **Prinsip**: "Rekomendasikan produk mirip dengan yang pernah dibeli"
- **Kelebihan**: Tidak ada cold start
- **Kekurangan**: Limited diversity

### 3. SVD Matrix Factorization
- **Metode**: Singular Value Decomposition (Surprise)
- **Prinsip**: "Temukan pola laten dalam data"
- **Kelebihan**: Akurasi tinggi, scalable
- **Kekurangan**: Kurang interpretable

### 4. Hybrid Filtering
- **Formula**: `Score = α×CF + β×CBF + γ×SVD`
- **Default bobot**: α=0.4, β=0.3, γ=0.3
- **Keuntungan**: Menggabung kelebihan semua algoritma

## 📈 Metrik Evaluasi

- **RMSE** (Root Mean Square Error) - Akurasi prediksi SVD
- **Precision@K** - % rekomendasi yang relevan
- **Recall@K** - % item relevan yang terrekomendasikan
- **Coverage** - Beragam produk yang direkomendasikan
- **F1-Score** - Harmonic mean precision & recall

## ⚙️ Konfigurasi Model

### CF Parameters
- Neighbors (K): 10
- Similarity metric: Cosine Similarity
- User-item matrix: Quantity-based

### CBF Parameters
- TF-IDF max_features: 500
- N-grams: (1, 2)
- Min_df: 2
- Max_df: 0.8

### SVD Parameters
- Latent factors: 100
- Epochs: 20
- Learning rate: 0.005
- Regularization: 0.02

## 🔧 Troubleshooting

### Model tidak ditemukan
```bash
# Pastikan sudah menjalankan semua notebooks terlebih dahulu
# Models disimpan di folder: models/
ls models/
```

### Streamlit tidak menemukan recommender
```bash
# Pastikan di folder yang sama dengan app.py
# run dari root directory project
cd TugasProjekPakRon
streamlit run app.py
```

### Memory error saat training
```bash
# Kurangi ukuran data atau gunakan subset
# Edit notebook, filter data sebelum training
```

## 📝 Lisensi & Attribution

- Dataset: Online Retail II (UCI Machine Learning Repository)
- Libraries: pandas, scikit-learn, Surprise, Streamlit, Plotly

## 👤 Author

Tugas Proyek Sistem Rekomendasi
Mata Kuliah: Sistem Rekomendasi
Dosen Pengampu: Pak Ronaldus

---

**Last Updated**: 2026-05-10
