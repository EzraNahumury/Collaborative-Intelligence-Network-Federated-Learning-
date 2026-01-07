# Bank J - Broker Saham & Kripto

## Deskripsi

Bank J merupakan bagian dari **Federated Learning Iterasi Ke-2**, yang menggunakan model global hasil pelatihan iterasi pertama dari Bank A-F sebagai titik awal. Bank J berfokus pada layanan **broker saham dan kripto**, yang menghadirkan tantangan unik dalam pendeteksian fraud.

## Karakteristik Bank J

### Profil Bank
- **Segmen**: Broker Saham & Cryptocurrency
- **Layanan Utama**: Trading saham, cryptocurrency, dan instrumen investasi
- **Tipe Transaksi**: Pembelian/penjualan aset (BBCA, BTC, ETH, dll.)

### Tantangan Teknis

#### 1. **Kardinalitas Tinggi & Data Tekstual**
Bank J menghadapi tantangan **high cardinality** pada fitur kategorikal:

- **Kolom `merchant_category` diganti dengan `asset_ticker`**
  - Contoh: `BBCA` (saham Bank BCA), `BTC` (Bitcoin), `ETH` (Ethereum), `TLKM` (Telkom), dll.
  - Berbeda dengan bank konvensional yang hanya memiliki beberapa kategori merchant (misalnya: groceries, retail, travel)
  
- **Implikasi**:
  - Model tidak bisa lagi mengandalkan beberapa kategori tetap
  - Harus menangani **ratusan atau ribuan ticker** yang terus berubah
  - Ticker baru dapat muncul seiring dengan IPO baru, listing crypto baru, dll.

#### 2. **Kebutuhan Feature Engineering**
Untuk mengatasi masalah kardinalitas tinggi, diperlukan teknik **feature engineering** khusus:

- **Embedding**: Merepresentasikan ticker dalam vektor berdimensi rendah
- **Hashing**: Menggunakan hash function untuk memetakan ticker ke ruang fitur yang lebih kecil
- **Target Encoding**: Encoding berbasis statistik dari target variable
- **Frequency Encoding**: Encoding berbasis frekuensi kemunculan ticker

Teknik-teknik ini memungkinkan model untuk:
- Generalisasi ke ticker yang belum pernah dilihat sebelumnya
- Mengurangi dimensi fitur tanpa kehilangan informasi penting
- Meningkatkan efisiensi komputasi dan performa model

## Cara Menjalankan

### Langkah 1: Training Model Bank J

1. **Masuk ke environment** (WSL/Virtual Environment):
   ```bash
   # Jika menggunakan WSL
   wsl
   
   # Aktivasi virtual environment (jika ada)
   source venv/bin/activate
   # atau
   conda activate <env_name>
   ```

2. **Jalankan training script**:
   ```bash
   python bankJ.py
   ```

   Script ini akan:
   - Memuat model global iterasi pertama (dari Bank A-F)
   - Melatih model dengan data spesifik Bank J
   - Melakukan federated learning dengan TensorFlow Federated (TFF)
   - Menyimpan model hasil training

### Langkah 2: Testing & Evaluasi

1. **Keluar dari environment/WSL**:
   ```bash
   # Deactivate virtual environment
   deactivate
   
   # Keluar dari WSL (jika menggunakan WSL)
   exit
   ```

2. **Jalankan testing script**:
   ```bash
   python test.py
   ```

   Script ini akan:
   - Memuat model yang telah dilatih
   - Melakukan evaluasi pada data test
   - Menampilkan metrik performa (accuracy, precision, recall, F1-score)
   - Menyimpan hasil evaluasi

## Output & Model yang Tersimpan

Hasil training dan testing disimpan di direktori:

```
models_round2\saved_bank_J_tff\
```

### Isi Folder Saved Model

Folder `saved_bank_J_tff` berisi:

1. **Model Weights**:
   - `model.h5` atau `model.keras`: Model TensorFlow/Keras yang telah dilatih
   - Berisi bobot (weights) dan bias dari semua layer neural network

2. **Model Architecture**:
   - `model_config.json`: Konfigurasi arsitektur model (jumlah layer, aktivasi, dll.)
   - Informasi tentang input shape, output shape, dan struktur model

3. **Training History**:
   - `history.pkl` atau `history.json`: Riwayat training
   - Berisi metrik per epoch: loss, accuracy, validation loss, validation accuracy

4. **Preprocessing Objects** (jika ada):
   - `scaler.pkl`: StandardScaler atau MinMaxScaler untuk normalisasi data
   - `encoder.pkl`: LabelEncoder atau OneHotEncoder untuk encoding kategorikal
   - `feature_hasher.pkl`: Feature hasher untuk encoding ticker dengan kardinalitas tinggi

5. **Metadata**:
   - `model_metadata.json`: Informasi tambahan
     - Tanggal training
     - Hyperparameters yang digunakan
     - Fitur engineering yang diterapkan
     - Performa model pada data validasi
     - Jumlah rounds federated learning
     - Informasi tentang global model iterasi pertama yang digunakan

6. **Evaluation Results**:
   - `test_results.json`: Hasil evaluasi pada test set
     - Accuracy, Precision, Recall, F1-Score
     - Confusion Matrix
     - Classification Report

## Federated Learning Iterasi Ke-2

Bank J merupakan bagian dari iterasi kedua dalam federated learning:

- **Input**: Model global dari iterasi pertama (Bank A-F)
- **Proses**: Fine-tuning model dengan data lokal Bank J yang memiliki karakteristik unik
- **Output**: Model yang sudah disesuaikan dengan pola transaksi broker saham & kripto
- **Keuntungan**: 
  - Memanfaatkan knowledge dari bank-bank lain
  - Tetap menjaga privasi data lokal Bank J
  - Meningkatkan kemampuan generalisasi model

## Pola Fraud yang Dideteksi

Pola fraud pada broker saham & kripto umumnya meliputi:

- **Pump and Dump**: Transaksi masif pada ticker tertentu dalam waktu singkat
- **Wash Trading**: Transaksi jual-beli berulang untuk menciptakan volume palsu
- **Front Running**: Transaksi mendahului order besar untuk mengambil keuntungan
- **Account Takeover**: Akun yang diambil alih untuk melakukan transaksi tidak sah
- **Unusual Asset Selection**: Pembelian aset yang tidak sesuai dengan profil investor

## Kesimpulan

Bank J menghadirkan tantangan unik dalam federated learning karena **kardinalitas tinggi** pada fitur `asset_ticker`. Hal ini memaksa penggunaan teknik feature engineering yang sophisticated seperti embedding dan hashing, yang merupakan pembelajaran penting untuk menangani data kategorikal dengan variabilitas tinggi dalam skenario dunia nyata.
