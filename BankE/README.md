# 🏦 Bank E - Wealth Management / Private Bank

## 📋 Deskripsi Bank E

**Bank E** adalah bank yang fokus pada layanan **Wealth Management** dan **Private Banking**, yang melayani nasabah **high-net-worth** (nasabah dengan kekayaan tinggi).

### Karakteristik Utama:

- **Volume Transaksi**: Sangat rendah (low transaction volume)
- **Nilai Transaksi**: Ekstrim tinggi per transaksi (extremely high amount per transaction)
- **Kategori Dominan**: 
  - `investment` (investasi)
  - `asset_management` (manajemen aset)
- **Profil Nasabah**: High-net-worth individuals (HNWI)

### 🚨 Pola Penipuan Lokal

Bank E memiliki karakteristik penipuan yang unik:

> **Penipuan berupa satu atau dua transfer internasional bernilai masif**

⚠️ **Tantangan Deteksi:**
- Model **tidak bisa hanya mengandalkan frekuensi tinggi** sebagai sinyal penipuan
- Fokus deteksi pada **nilai transaksi yang sangat besar** dan **pola transaksi internasional yang mencurigakan**
- Perlu mempertimbangkan **anomali pada kategori transaksi** yang tidak sesuai dengan profil nasabah

---

## 🚀 Cara Menjalankan

### Tahapan Eksekusi

#### **Tahap 1: Training Model (bankE.py)**

1. **Masuk ke environment/WSL:**
   ```bash
   # Aktivasi virtual environment
   source venv/bin/activate
   # atau di Windows:
   venv\Scripts\activate
   ```

2. **Jalankan script training:**
   ```bash
   python bankE.py
   ```

   Script ini akan:
   - Membaca data dari folder `data/`
   - Memuat fitur global dari `models_global/fitur_global.pkl`
   - Melakukan preprocessing data
   - Membagi data menjadi beberapa klien federated
   - Melatih model menggunakan **TensorFlow Federated (TFF)** dengan algoritma **Weighted FedAvg**
   - Menyimpan model dan metadata ke folder `Models/saved_bank_E_DATA_tff/`

#### **Tahap 2: Testing Model (test.py)**

1. **Keluar dari environment/WSL** (jika masih di dalam environment)
   ```bash
   deactivate
   ```

2. **Jalankan script testing:**
   ```bash
   python test.py
   ```

   Script ini akan:
   - Memuat model yang telah dilatih dari `Models/saved_bank_E_DATA_tff/`
   - Menguji model menggunakan test cases global untuk semua bank (A, B, C, D, E, F, G)
   - Menghitung akurasi per bank dan total accuracy
   - Menyimpan hasil akurasi ke `best_accuracy.txt`

---

## 📁 Struktur Hasil Model

Setelah menjalankan `bankE.py` dan `test.py`, hasil akan disimpan di:

```
Models/saved_bank_E_DATA_tff/
```

### Isi Folder Saved Model:

| File/Folder | Deskripsi |
|-------------|-----------|
| **`saved_model.pb`** | Model TensorFlow dalam format SavedModel (Protocol Buffer) - berisi arsitektur dan graph computation |
| **`keras_metadata.pb`** | Metadata Keras yang berisi informasi tentang layer, konfigurasi model, dan versi Keras |
| **`fingerprint.pb`** | Fingerprint unik untuk verifikasi integritas model |
| **`variables/`** | Folder berisi variabel model (weights dan biases) dalam format checkpoint |
| **`assets/`** | Folder untuk asset tambahan yang dibutuhkan model (jika ada) |
| **`YYYYMMDD_HHMMSS.npz`** | Snapshot bobot model dengan timestamp (format NumPy NPZ) untuk versioning |
| **`preprocess_bank_E_DATA.pkl`** | Preprocessor (scaler, encoder) yang digunakan untuk transformasi data - **penting untuk inference** |
| **`history_bank_E_DATA.json`** | Riwayat training berisi metrik per round federated (accuracy, loss, dll) |
| **`accuracy_history.txt`** | Log akurasi training dalam format teks (human-readable) |
| **`best_accuracy.txt`** | Akurasi terbaik yang dicapai model pada test cases global |

### 🔍 Penjelasan Detail:

1. **Model Files (`saved_model.pb`, `keras_metadata.pb`, `variables/`)**
   - Menyimpan arsitektur neural network dan learned parameters
   - Dapat dimuat ulang untuk inference tanpa perlu re-training

2. **Preprocessing (`preprocess_bank_E_DATA.pkl`)**
   - ⚠️ **CRITICAL**: File ini **wajib** digunakan saat melakukan prediksi pada data baru
   - Berisi transformasi yang sama yang digunakan saat training (feature encoding, scaling, dll)
   
3. **Training History (`history_bank_E_DATA.json`, `accuracy_history.txt`)**
   - Tracking progress training untuk monitoring dan debugging
   - Berguna untuk analisis performa model dari waktu ke waktu

4. **Weight Snapshots (`YYYYMMDD_HHMMSS.npz`)**
   - Backup bobot model dengan timestamp untuk rollback jika diperlukan
   - Memungkinkan perbandingan performa antar versi model

5. **Accuracy Metrics (`best_accuracy.txt`)**
   - Diperbarui setiap kali `test.py` dijalankan dengan hasil terbaru
   - Digunakan untuk tracking performa model pada global test cases

---

## 📊 Catatan Penting

- Model dilatih menggunakan **Federated Learning** dengan **TensorFlow Federated (TFF)**
- Menggunakan algoritma **Weighted FedAvg** untuk agregasi model dari multiple clients
- Data nasabah tetap terdistribusi dan tidak perlu dikumpulkan di satu tempat (privacy-preserving)
- Threshold klasifikasi fraud ditentukan secara otomatis menggunakan **F1-Score optimization**


