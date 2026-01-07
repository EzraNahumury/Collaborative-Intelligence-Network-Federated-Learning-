# 🏦 Bank C - Dokumentasi Model Federated Learning

## 📋 Deskripsi Bank C

**Bank C ("Bank Ritel Konvensional")**

Bank C merupakan bank ritel konvensional dengan karakteristik sebagai berikut:

### Karakteristik Transaksi
- **Distribusi Nilai Transaksi**: Nilai transaksi tersebar secara merata
- **Lokasi Transaksi**: Mayoritas transaksi terjadi secara **fisik** di berbagai kota seperti:
  - Jakarta
  - Bandung
  - Surabaya
  - Dan kota-kota lainnya di Indonesia
- **Kategori Dominan**: Transaksi didominasi oleh kategori:
  - **Groceries** (Belanja kebutuhan sehari-hari)
  - **Retail** (Pembelian ritel)

### 🚨 Pola Penipuan Lokal
Penipuan pada Bank C memiliki pola khas:
- Terjadi pada **transaksi fisik**
- Transaksi dilakukan di **luar kota domisili nasabah**
- Penipuan memanfaatkan transaksi fisik yang tidak biasa dari lokasi yang berbeda

---

## 🚀 Cara Running Program

### Prerequisites
- Python 3.x
- Virtual Environment (venv) atau WSL
- TensorFlow & TensorFlow Federated
- Dependencies lainnya (lihat requirements jika ada)

### Tahapan Eksekusi

#### **Tahap 1: Training Model (Dalam Environment/WSL)**

1. **Masuk ke Virtual Environment atau WSL**
   ```bash
   # Untuk Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   
   # Untuk Linux/WSL/Mac
   source venv/bin/activate
   ```

2. **Jalankan Script Training `bankC.py`**
   ```bash
   python bankC.py
   ```
   
   Script ini akan:
   - Memuat data dari `data/bank_C_DATA.csv`
   - Memuat konfigurasi fitur global dari `models_global/fitur_global.pkl`
   - Melakukan preprocessing data
   - Membagi data menjadi beberapa klien federated (default: 5 klien)
   - Melatih model menggunakan Weighted Federated Averaging
   - Menyimpan model dan metadata ke folder `Models/saved_bank_C_DATA_tff`

3. **Keluar dari Virtual Environment/WSL**
   ```bash
   deactivate
   ```

#### **Tahap 2: Testing Model (Di Luar Environment)**

4. **Jalankan Script Testing `test.py`**
   ```bash
   python test.py
   ```
   
   Script ini akan:
   - Memuat model yang sudah dilatih dari `Models/saved_bank_C_DATA_tff`
   - Menguji model dengan test cases dari berbagai bank (A-F)
   - Menghitung akurasi keseluruhan
   - Menyimpan hasil akurasi terbaik ke `best_accuracy.txt`

---

## 📁 Struktur Folder `Models\saved_bank_C_DATA_tff`

Setelah menjalankan kedua script, hasil model akan disimpan di folder **`Models\saved_bank_C_DATA_tff`** dengan struktur sebagai berikut:

### File-file yang Tersimpan:

| File/Folder | Deskripsi |
|-------------|-----------|
| **`saved_model.pb`** | Model TensorFlow dalam format SavedModel (Protocol Buffer). File utama yang berisi arsitektur dan graph model. |
| **`variables/`** | Folder berisi bobot (weights) dan biases dari model neural network dalam format TensorFlow checkpoint. |
| **`keras_metadata.pb`** | Metadata Keras yang berisi informasi tentang layer, konfigurasi model, dan versi. |
| **`fingerprint.pb`** | Fingerprint unik untuk verifikasi integritas model. |
| **`assets/`** | Folder untuk menyimpan aset tambahan (jika ada). |
| **`<timestamp>.npz`** | File bobot model dalam format NumPy compressed (NPZ) dengan timestamp (contoh: `20260105_115837.npz`). Berguna untuk backup atau analisis offline. |
| **`preprocess_bank_C_DATA.pkl`** | File pickle berisi metadata preprocessing seperti:<br>- `MODE`: Mode fitur (DICT/LIST)<br>- `NUM_COLS`: Kolom numerik<br>- `CAT_COLS`: Kolom kategorikal<br>- `HASHER_DIM`: Dimensi hashing<br>- `SCALER`: Parameter scaler (min, range)<br>- `FEATURE_DIM`: Dimensi fitur total |
| **`history_bank_C_DATA.json`** | History training dalam format JSON berisi:<br>- Round number<br>- Accuracy per round<br>- PR-AUC (Precision-Recall Area Under Curve)<br>- Loss per round |
| **`accuracy_history.txt`** | File teks berisi log lengkap history akurasi dalam format tab-separated:<br>- Bank name<br>- Round<br>- Accuracy<br>- PR-AUC<br>- Loss<br>- Timestamp |
| **`best_accuracy.txt`** | File berisi akurasi terbaik dari hasil testing global (diupdate oleh `test.py`). Format: satu angka float (0.0 - 1.0). |

### Penjelasan Detail Isi Model

#### 1. **Model SavedModel** (`saved_model.pb` + `variables/`)
- Berisi model neural network lengkap yang siap digunakan untuk inference
- Arsitektur model:
  - Input Layer (dimensi sesuai jumlah fitur)
  - Batch Normalization
  - Dense Layer 128 unit + ReLU + L2 Regularization + Dropout 30%
  - Dense Layer 64 unit + ReLU + L2 Regularization + Dropout 20%
  - Output Layer 1 unit + Sigmoid (untuk klasifikasi biner fraud/non-fraud)

#### 2. **Preprocessing Metadata** (`preprocess_bank_C_DATA.pkl`)
- Diperlukan untuk memproses data baru dengan cara yang sama seperti saat training
- Menyimpan parameter scaler (MinMax) untuk normalisasi fitur numerik
- Informasi kolom kategorikal dan konfigurasi hashing (jika digunakan)

#### 3. **Training History** (`history_bank_C_DATA.json` & `accuracy_history.txt`)
- Mencatat performa model di setiap round training
- Berguna untuk analisis konvergensi dan debugging
- Dapat digunakan untuk visualisasi kurva learning

#### 4. **Best Accuracy** (`best_accuracy.txt`)
- Menyimpan akurasi terbaik dari hasil testing pada test cases global
- Diperbarui oleh script `test.py` setelah evaluasi
- Format: floating point 6 digit (contoh: 0.833333)

---

## 🔬 Parameter Training

Parameter yang digunakan dalam training (dapat diubah melalui argumen):

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `--bank` | `C_data` | Kode bank |
| `--n_clients` | `5` | Jumlah klien federated |
| `--batch_size` | `32` | Ukuran batch untuk training |
| `--rounds` | `10` | Jumlah round federated learning |
| `--lr_client` | `5e-4` | Learning rate client optimizer |
| `--lr_server` | `1e-3` | Learning rate server optimizer |

---

## 📊 Evaluasi Model

Script `test.py` mengevaluasi model dengan test cases dari 6 bank berbeda (A-F), termasuk:

**Test Cases Bank C:**
- Case 1: Transaksi crypto online internasional (Fraud)
- Case 2: Transaksi retail Surabaya lokal (Non-fraud)
- Case 3: Transaksi loan Singapore internasional (Fraud)
- Case 4: Transaksi payment gateway Jakarta lokal (Non-fraud)

Model menggunakan **threshold otomatis** yang dihitung berdasarkan:
- Precision-Recall curve (F1-score maksimal)
- ROC curve (Youden's Index)
- Rata-rata threshold dari kedua metode

---

## 📝 Catatan

- **Early Stopping**: Training akan berhenti otomatis jika akurasi sudah stabil (perubahan < 1e-4) setelah round ke-3
- **Federated Learning**: Model dilatih secara terdistribusi di beberapa klien, kemudian diagregasi menggunakan Weighted FedAvg
- **Format Model**: Model disimpan dalam format TensorFlow SavedModel untuk kompatibilitas maksimal
- **Backup Weights**: File NPZ berisi backup bobot model dengan timestamp untuk tracking versi

---

## 🛠️ Troubleshooting

### Error: File tidak ditemukan
- Pastikan file `data/bank_C_DATA.csv` tersedia
- Pastikan file `models_global/fitur_global.pkl` tersedia

### Error: Model tidak bisa dimuat
- Pastikan sudah menjalankan `bankC.py` terlebih dahulu
- Periksa path `Models/saved_bank_C_DATA_tff` apakah sudah terisi

### Error: Import module tidak ditemukan
- Pastikan semua dependencies terinstall
- Aktifkan virtual environment sebelum running

---

