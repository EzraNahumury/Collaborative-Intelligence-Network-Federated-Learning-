# 📘 Bank D - Dokumentasi Sistem Deteksi Fraud

## 🏦 Tentang Bank D

**Bank D** adalah bank yang berfokus pada **Fintech P2P/Paylater** (Peer-to-Peer Lending dan Pembayaran Cicilan). Bank ini mengkhususkan diri dalam layanan pinjaman digital dan pembayaran tagihan secara online.

### Karakteristik Utama

- **Fokus Layanan**: Pinjaman digital dan pembayaran cicilan
- **Transaksi Utama**: 
  - Pencairan pinjaman (loan disbursement)
  - Pembayaran tagihan (bill payment)
  - Pembayaran cicilan (installment payment)
- **Platform**: Berbasis digital dan aplikasi mobile

### 🚨 Pola Penipuan Lokal

Bank D rentan terhadap **penipuan aplikasi pinjaman palsu**. Pola penipuan yang umum terjadi meliputi:

- **Aplikasi Pinjaman Palsu**: Aplikasi yang menyamar sebagai platform pinjaman resmi
- **Pencairan Dana Mencurigakan**: Terlihat seperti transaksi `loan_disbursement` dengan frekuensi yang tidak wajar
- **Akun-akun Baru**: Transaksi penipuan sering dilakukan ke akun-akun yang baru dibuat
- **Indikator Kecurigaan**:
  - Frekuensi pencairan dana yang terlalu tinggi dalam waktu singkat
  - Transaksi ke akun rekening yang baru terdaftar
  - Pola pengajuan pinjaman yang tidak realistis
  - Pencairan ke lokasi yang tidak biasa atau mencurigakan

---

## 🚀 Cara Menjalankan Sistem

Sistem deteksi fraud Bank D menggunakan **Federated Learning** dengan TensorFlow Federated (TFF). Proses pelatihan dan pengujian dilakukan dalam dua tahap terpisah.

### Langkah 1: Training Model (Dalam Environment)

1. **Masuk ke Environment/WSL**
   
   Aktifkan virtual environment Python:
   ```bash
   # Windows
   .\venv\Scripts\activate
   
   # WSL/Linux
   source venv/bin/activate
   ```

2. **Jalankan Script Training**
   
   Jalankan `bankD.py` untuk melatih model federated learning:
   ```bash
   python bankD.py --bank D_data --rounds 10 --n_clients 5
   ```
   
   **Parameter yang dapat disesuaikan:**
   - `--bank`: Kode bank (default: `D_data`)
   - `--rounds`: Jumlah round federated training (default: `10`)
   - `--n_clients`: Jumlah klien untuk federated learning (default: `5`)
   - `--batch_size`: Ukuran batch (default: `32`)
   - `--lr_client`: Learning rate client (default: `5e-4`)
   - `--lr_server`: Learning rate server (default: `1e-3`)

3. **Proses Training**
   
   Script akan:
   - Memuat data dari `data/bank_D_DATA.csv`
   - Memuat konfigurasi fitur global dari `models_global/fitur_global.pkl`
   - Melakukan preprocessing data
   - Membagi data menjadi beberapa klien federated
   - Melatih model menggunakan Weighted FedAvg
   - Menyimpan hasil ke folder `Models/saved_bank_D_DATA_tff`

4. **Keluar dari Environment**
   
   Setelah training selesai, keluar dari environment:
   ```bash
   deactivate
   ```

### Langkah 2: Testing Model (Di Luar Environment/WSL)

1. **Jalankan Script Testing**
   
   Setelah keluar dari environment/WSL, jalankan script testing:
   ```bash
   python test.py
   ```

2. **Proses Testing**
   
   Script `test.py` akan:
   - Memuat model yang telah dilatih dari `Models/saved_bank_D_DATA_tff`
   - Memuat konfigurasi preprocessing dari `models_global/fitur_global_test.pkl`
   - Menguji model dengan test cases untuk semua bank (A-F)
   - Menghitung akurasi untuk setiap bank
   - Menyimpan hasil akurasi terbaik ke `best_accuracy.txt`

3. **Konfigurasi Testing**
   
   Anda dapat mengubah konfigurasi di `test.py`:
   ```python
   THRESHOLD_MODE = "AUTO"   # "AUTO" atau "MANUAL"
   THRESHOLD_MANUAL = 0.5    # threshold manual jika dipilih MANUAL
   ```

---

## 📁 Struktur Saved Model

Hasil training disimpan di folder **`Models/saved_bank_D_DATA_tff`** dengan struktur sebagai berikut:

### Daftar File dan Penjelasan

#### 1. **`saved_model.pb`**
- **Deskripsi**: File protokol buffer yang berisi arsitektur model Keras
- **Ukuran**: ~120 KB
- **Fungsi**: Menyimpan graph komputasi dan struktur layer model neural network

#### 2. **`variables/`** (Folder)
- **Deskripsi**: Folder yang berisi bobot (weights) dan bias model
- **Isi**: 
  - `variables.data-00000-of-00001`: Data bobot model
  - `variables.index`: Index untuk mengakses bobot
- **Fungsi**: Menyimpan parameter yang telah dipelajari selama training

#### 3. **`[timestamp].npz`** (contoh: `20260105_120956.npz`)
- **Deskripsi**: Snapshot bobot model dalam format NumPy compressed
- **Timestamp**: Waktu ketika model disimpan
- **Fungsi**: Backup bobot model untuk versioning dan recovery

#### 4. **`preprocess_bank_D_DATA.pkl`**
- **Deskripsi**: File pickle yang berisi metadata preprocessing
- **Ukuran**: ~1 KB
- **Isi**:
  - Mode preprocessing (DICT atau LIST)
  - Kolom numerik dan kategorikal
  - Parameter scaler (MinMaxScaler)
  - Dimensi fitur hasil preprocessing
  - Konfigurasi feature hashing (jika menggunakan mode DICT)
- **Fungsi**: Memastikan konsistensi preprocessing antara training dan inference

#### 5. **`history_bank_D_DATA.json`**
- **Deskripsi**: Riwayat training dalam format JSON
- **Ukuran**: ~1-2 KB
- **Isi**:
  - Akurasi per round
  - PR-AUC (Precision-Recall Area Under Curve) per round
  - Loss per round
  - Nomor round
- **Fungsi**: Tracking performa model selama proses training

#### 6. **`accuracy_history.txt`**
- **Deskripsi**: Log akurasi lengkap dalam format tab-separated
- **Format**: `bank \t round \t acc \t pr_auc \t loss \t timestamp`
- **Fungsi**: Logging untuk analisis historis dan debugging

#### 7. **`best_accuracy.txt`**
- **Deskripsi**: File yang menyimpan akurasi terbaik model
- **Isi**: Satu baris berisi nilai akurasi (format: `0.XXXXXX`)
- **Sumber**: Akurasi dari hasil testing menggunakan `test.py`
- **Fungsi**: Referensi cepat untuk evaluasi performa model

#### 8. **`keras_metadata.pb`**
- **Deskripsi**: Metadata Keras model
- **Ukuran**: ~12 KB
- **Fungsi**: Informasi konfigurasi model untuk loading dan serving

#### 9. **`fingerprint.pb`**
- **Deskripsi**: Fingerprint unik model
- **Ukuran**: ~55 bytes
- **Fungsi**: Identifikasi versi dan validasi integritas model

#### 10. **`assets/`** (Folder)
- **Deskripsi**: Folder untuk aset tambahan model
- **Fungsi**: Menyimpan file pendukung jika diperlukan

---

## 🔍 Arsitektur Model

Model menggunakan **Dense Neural Network** dengan konfigurasi:

```
Input Layer (dimensi sesuai jumlah fitur)
    ↓
Batch Normalization
    ↓
Dense Layer (128 units, ReLU activation, L2 regularization)
    ↓
Dropout (30%)
    ↓
Dense Layer (64 units, ReLU activation, L2 regularization)
    ↓
Dropout (20%)
    ↓
Output Layer (1 unit, Sigmoid activation)
```

**Loss Function**: Binary Crossentropy  
**Optimizer**: Adam (dengan learning rate terpisah untuk client dan server)  
**Metrics**: Binary Accuracy, PR-AUC

---

## 📊 Evaluasi Model

### Metrik yang Digunakan

1. **Binary Accuracy**: Persentase prediksi yang benar
2. **PR-AUC**: Area under Precision-Recall curve (khususnya penting untuk imbalanced data)
3. **Loss**: Binary crossentropy loss

### Threshold Otomatis

Sistem menggunakan threshold otomatis yang dihitung dari:
- F1-Score optimal dari Precision-Recall curve
- Youden's Index dari ROC curve
- Rata-rata kedua threshold di atas

---

## 📝 Catatan Penting

1. **Environment Separation**: Training dilakukan di dalam environment untuk memastikan dependency yang konsisten, sedangkan testing dapat dilakukan di luar environment
2. **Data Path**: Pastikan file `data/bank_D_DATA.csv` tersedia sebelum training
3. **Global Features**: File `fitur_global.pkl` dan `fitur_global_test.pkl` harus tersedia di folder `models_global/`
4. **Early Stopping**: Training akan berhenti otomatis jika akurasi sudah stabil (perubahan < 0.0001 setelah round ke-3)

---

## 🛠️ Troubleshooting

### Error: "Data tidak ditemukan"
- Pastikan file `data/bank_D_DATA.csv` ada dan berformat CSV yang valid
- Periksa kolom `is_fraud` ada dalam dataset

### Error: "fitur_global.pkl tidak ditemukan"
- Pastikan file `models_global/fitur_global.pkl` tersedia
- File ini berisi konfigurasi preprocessing yang harus sama antara training dan testing

### Model tidak konvergen
- Coba tingkatkan jumlah `--rounds`
- Sesuaikan learning rate dengan `--lr_client` dan `--lr_server`
- Periksa kualitas dan distribusi data

