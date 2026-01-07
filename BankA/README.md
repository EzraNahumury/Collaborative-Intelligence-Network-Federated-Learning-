# 🏦 Bank A - Bank Digital Inovatif

## 📋 Deskripsi Bank A

**Bank A** adalah bank digital inovatif dengan karakteristik transaksi yang unik:

### Karakteristik Transaksi
- ✅ **Volume Transaksi Tinggi**: Bank A memproses jumlah transaksi yang sangat banyak setiap harinya
- 💰 **Nilai Rata-Rata Rendah**: Meskipun volume tinggi, nilai rata-rata (amount) per transaksi relatif rendah
- 🛒 **Dominasi Transaksi Online**: Mayoritas transaksi terjadi secara **online** di merchant **e-commerce**
- 🌐 **Fokus Nasional**: Sebagian besar transaksi bersifat domestik (lokal)

### 🚨 Pola Penipuan Lokal

Bank A memiliki pola penipuan yang khas dan berbeda dari bank lain:

> **Karakteristik Fraud**: Penipuan pada Bank A seringkali berupa **serangkaian transaksi kecil yang dilakukan dengan cepat**.

Indikator utama penipuan:
- 📊 **`transaction_frequency_24h` tinggi**: Banyak transaksi dalam waktu 24 jam
- 💳 **Transaksi kecil berulang**: Nilai kecil namun frekuensi sangat tinggi
- ⚡ **Kecepatan tinggi**: Transaksi dilakukan dalam rentang waktu singkat
- 🛍️ **E-commerce**: Mayoritas terjadi di merchant online/e-commerce

---

## 🚀 Cara Menjalankan Program

### Prasyarat
- Python 3.x terinstal
- Virtual environment (venv) atau WSL
- Dependencies yang diperlukan sudah terinstal

### Tahapan Menjalankan

#### 1️⃣ **Masuk ke Environment & Menjalankan Training Model**

Pertama, masuk ke dalam virtual environment atau WSL:

```bash
# Untuk Windows (PowerShell/Command Prompt)
.\venv\Scripts\activate

# Untuk WSL/Linux
source venv/bin/activate
```

Kemudian jalankan file **`bankA.py`** untuk melatih model federated learning:

```bash
python bankA.py
```

**Proses yang terjadi:**
- 📂 Membaca data transaksi dari folder `data/`
- 🔄 Memproses data dengan fitur global dari `models_global/fitur_global.pkl`
- 🧠 Melatih model menggunakan **TensorFlow Federated (TFF)** dengan algoritma Weighted FedAvg
- 💾 Menyimpan model dan metadata ke folder `Models/saved_bank_A_DATA_tff/`

#### 2️⃣ **Keluar dari Environment**

Setelah training selesai, keluar dari environment:

```bash
# Keluar dari virtual environment
deactivate
```

#### 3️⃣ **Menjalankan Testing Model**

Setelah keluar dari environment/WSL, jalankan file **`test.py`**:

```bash
python test.py
```

**Proses yang terjadi:**
- 🧪 Memuat model yang sudah dilatih dari `Models/saved_bank_A_DATA_tff/`
- 🔍 Menguji model dengan test cases untuk semua bank (A, B, C, D, E)
- 📊 Menghitung akurasi, precision, recall untuk setiap bank
- 📝 Menyimpan hasil testing ke `best_accuracy.txt`

---

## 📦 Isi Folder `Models/saved_bank_A_DATA_tff`

Setelah menjalankan `bankA.py` dan `test.py`, hasil model dan metadata akan tersimpan di folder berikut:

```
Models/saved_bank_A_DATA_tff/
```

### File dan Folder yang Disimpan:

| File/Folder | Deskripsi |
|-------------|-----------|
| 📄 **`saved_model.pb`** | Model utama TensorFlow dalam format Protocol Buffer |
| 📄 **`keras_metadata.pb`** | Metadata Keras untuk konfigurasi model |
| 📄 **`fingerprint.pb`** | Fingerprint model untuk verifikasi integritas |
| 📂 **`variables/`** | Folder berisi bobot (weights) model neural network |
| 📂 **`assets/`** | Folder berisi aset tambahan model (jika ada) |
| 📄 **`YYYYMMDD_HHMMSS.npz`** | File bobot model dengan timestamp (contoh: `20260105_114811.npz`) |
| 📄 **`preprocess_bank_A_DATA.pkl`** | File preprocessing metadata (scaler, encoder, fitur yang digunakan) |
| 📄 **`history_bank_A_DATA.json`** | Riwayat training (akurasi, loss, metrics per round) |
| 📄 **`accuracy_history.txt`** | Riwayat akurasi dalam format teks |
| 📄 **`best_accuracy.txt`** | Akurasi terbaik yang dicapai model setelah testing |

### Penjelasan Detail:

#### 🧠 Model Files
- **`saved_model.pb`**, **`keras_metadata.pb`**, **`fingerprint.pb`**, **`variables/`**, **`assets/`**: 
  - File-file standar TensorFlow SavedModel format
  - Berisi arsitektur neural network, konfigurasi, dan bobot model
  - Digunakan untuk deployment dan inference

#### ⚙️ Preprocessing Files
- **`preprocess_bank_A_DATA.pkl`**: 
  - Berisi informasi preprocessing yang digunakan saat training
  - Menyimpan:
    - Daftar fitur yang digunakan
    - Feature engineering metadata
    - Informasi dimensi input model
  - **Sangat penting** untuk memastikan data testing diproses dengan cara yang sama seperti saat training

#### 📊 Training History Files
- **`YYYYMMDD_HHMMSS.npz`**: 
  - Snapshot bobot model dengan timestamp
  - Berguna untuk tracking versi model dan rollback jika diperlukan

- **`history_bank_A_DATA.json`**: 
  - Riwayat lengkap proses training federated
  - Berisi metrik per round: akurasi, loss, dll.
  - Format JSON memudahkan analisis dan visualisasi

- **`accuracy_history.txt`**: 
  - Format teks sederhana untuk tracking akurasi per round
  - Mudah dibaca untuk monitoring cepat

#### 🎯 Testing Result
- **`best_accuracy.txt`**: 
  - Akurasi final dari hasil testing dengan `test.py`
  - Diupdate setiap kali testing dilakukan
  - Berisi nilai akurasi terbaik dalam format desimal (contoh: `0.923077`)

---

## 📝 Catatan Penting

1. **Urutan Eksekusi**: Pastikan menjalankan `bankA.py` terlebih dahulu sebelum `test.py`
2. **Environment**: `bankA.py` dijalankan **di dalam** environment, sedangkan `test.py` dapat dijalankan **di luar** environment
3. **Model Persistence**: Semua file di `saved_bank_A_DATA_tff/` diperlukan untuk inference yang benar
4. **Preprocessing Consistency**: File `preprocess_bank_A_DATA.pkl` harus selalu digunakan saat melakukan prediksi pada data baru

---

## 🔗 File Terkait

- 📄 [`bankA.py`](bankA.py) - Script training model federated
- 📄 [`test.py`](test.py) - Script testing model
- 📂 `data/` - Folder data transaksi Bank A
- 📂 `models_global/` - Folder fitur global untuk preprocessing
- 📂 `Models/saved_bank_A_DATA_tff/` - Folder output model dan metadata
