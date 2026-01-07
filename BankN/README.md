# Bank N - Federated Learning Round 2

## 📌 Deskripsi Bank N

**Bank N** merupakan **"Bank dengan Data Terenkripsi"** yang berpartisipasi dalam iterasi kedua federated learning. Bank ini menggunakan model global hasil pelatihan dari iterasi pertama (Round 1) yang telah dilatih menggunakan data dari Bank A hingga Bank F.

### 🔐 Tantangan Utama: **Privasi Tingkat Lanjut**

Bank N menghadapi tantangan unik dalam hal privasi data:
- **Fitur Sensitif Terenkripsi**: Beberapa fitur sensitif seperti `merchant_id` dan `customer_id` sudah di-hash atau dienkripsi di level data sumber
- **Data Tidak Terbaca Manusia**: Semua representasi data yang digunakan tidak dapat dibaca dalam bentuk aslinya

### 💡 Implikasi untuk Model

Model federated learning pada Bank N harus memiliki kemampuan khusus:
- ✅ Mampu belajar dari representasi data yang telah terenkripsi
- ✅ Tidak bergantung pada nilai asli dari fitur sensitif
- ✅ Mensimulasikan skenario real-world di mana peserta konsorsium sangat protektif terhadap data mereka
- ✅ Membuktikan bahwa federated learning tetap efektif bahkan dengan data yang sudah dienkripsi sebelumnya

> **Catatan:** Ini mensimulasikan skenario di mana peserta konsorsium finansial sangat protektif terhadap privasi data mereka, bahkan dalam bentuk terenkripsi sekalipun. Model harus tetap mampu mendeteksi fraud patterns tanpa mengakses data asli.

---

## 🚀 Cara Menjalankan

### Prasyarat
- Python environment dengan TensorFlow Federated (TFF) terinstal
- WSL (Windows Subsystem for Linux) atau environment Linux
- Dataset Bank N tersimpan di folder `data/`
- Model global Round 1 tersedia di `models_global_round1/`

### Langkah 1: Training Model (di dalam Environment/WSL)

Masuk ke environment atau WSL terlebih dahulu, kemudian jalankan:

```bash
# Aktifkan virtual environment (jika menggunakan venv)
source venv/bin/activate  # Linux/WSL
# atau
.\venv\Scripts\activate   # Windows

# Jalankan training script
python bankN.py
```

**Opsi tambahan:**
```bash
# Training dengan konfigurasi custom
python bankN.py --bank N --n_clients 3 --rounds 10 --batch_size 32

# Resume training dari checkpoint
python bankN.py --resume
```

**Parameter yang tersedia:**
- `--bank`: Kode bank (default: N)
- `--data_dir`: Direktori data (default: data)
- `--models_dir`: Direktori output model (default: models_round2)
- `--global_dir`: Direktori fitur global (default: models_global)
- `--global_model_r1`: Path model global Round 1 (default: models_global_round1/global_savedmodel)
- `--n_clients`: Jumlah klien federated (default: 3)
- `--batch_size`: Ukuran batch (default: 32)
- `--rounds`: Jumlah round training (default: 10)
- `--lr_client`: Learning rate klien (default: 5e-4)
- `--lr_server`: Learning rate server (default: 1e-3)
- `--resume`: Lanjutkan dari checkpoint

### Langkah 2: Testing Model (di luar Environment/WSL)

Keluar dari environment/WSL, kemudian jalankan:

```bash
# Deaktivasi environment (jika masih aktif)
deactivate

# Jalankan testing script
python test.py
```

Script `test.py` akan:
- Memuat model yang telah dilatih dari `models_round2/saved_bank_N_tff`
- Menjalankan test cases dari Bank A-F
- Menghitung akurasi per bank dan akurasi keseluruhan
- Menyimpan hasil akurasi terbaik

---

## 💾 Output Model

Hasil training dan testing disimpan di direktori:
```
models_round2/saved_bank_N_tff/
```

### 📁 Isi Folder Saved Model

Folder `saved_bank_N_tff` berisi berbagai artefak hasil training:

#### 1. **Keras SavedModel (Folder Standard)**
- `saved_model.pb`: Model dalam format Protocol Buffer
- `variables/`: Direktori berisi weights model
  - `variables.data-00000-of-00001`: File data weights
  - `variables.index`: Index file untuk weights
- `assets/`: Aset tambahan (jika ada)

**Kegunaan:** Format standar TensorFlow untuk deployment dan inference

#### 2. **Weights NPZ (File Timestamped)**
- Format: `YYYYMMDD_HHMMSS.npz`
- Contoh: `20260106_101530.npz`

**Isi:** Compressed numpy array berisi semua weights model

**Kegunaan:** Backup weights untuk loading manual atau transfer learning

#### 3. **Preprocessing Metadata**
- File: `preprocess_bank_N.pkl`

**Isi:**
- `MODE`: Mode preprocessing (DICT/LIST)
- `NUM_COLS`: Daftar kolom numerik
- `CAT_COLS`: Daftar kolom kategorikal (yang di-hash)
- `HASHER_DIM`: Dimensi feature hashing
- `SCALER`: Parameter scaler (min-max atau standard)
- `FEATURE_DIM`: Total dimensi fitur hasil preprocessing

**Kegunaan:** Memastikan preprocessing data baru konsisten dengan training data

#### 4. **Training History**
- File: `history_bank_N.json`

**Format:**
```json
[
  {
    "round": 1,
    "acc": 0.8234,
    "pr_auc": 0.7845,
    "loss": 0.3421
  },
  ...
]
```

**Kegunaan:** Tracking performa model sepanjang training rounds

#### 5. **Accuracy History Log**
- File: `accuracy_history.txt`

**Format:** Tab-separated values
```
bank    round   acc       pr_auc    loss      timestamp
bank_N  1       0.823400  0.784500  0.342100  2026-01-06T03:15:30Z
bank_N  2       0.841200  0.798300  0.312400  2026-01-06T03:16:45Z
...
```

**Kegunaan:** Log lengkap untuk analisis historis dan monitoring

#### 6. **Best Accuracy**
- File: `best_accuracy.txt`

**Isi:** Akurasi terbaik dalam format desimal (0-1)
```
0.875000
```

**Kegunaan:** Quick reference untuk model evaluation, ditulis oleh `test.py`

#### 7. **Checkpoint Directory**
- Folder: `ckpt/`
  - `server_state.npz`: State terakhir dari federated server
  - `history.json`: History untuk resume training

**Kegunaan:** Resume training jika terinterupsi dengan flag `--resume`

---

## 📊 Proses Federated Learning Round 2

### Base Model Initialization
Bank N menggunakan **model global Round 1** sebagai starting point:
- Model global R1 dilatih dari Bank A-F
- Weights pre-trained di-load sebagai base initialization
- Fine-tuning dilakukan menggunakan data lokal Bank N

### Weighted FedAvg Algorithm
- **Client Optimizer:** Adam (lr = 5e-4)
- **Server Optimizer:** Adam (lr = 1e-3)
- **Aggregation:** Weighted averaging berdasarkan jumlah data per client

### Early Stopping
Training akan berhenti lebih awal jika:
- Minimal 3 rounds telah selesai
- Perubahan akurasi < 0.0001 antara rounds

---

## 🔍 Validasi Model

Model divalidasi menggunakan test cases dari 6 bank (A-F) untuk memastikan:
- ✅ **Generalization:** Model dapat generalize ke data dari bank lain
- ✅ **Cross-Bank Performance:** Performa konsisten lintas berbagai karakteristik bank
- ✅ **Encrypted Data Handling:** Model tetap efektif meskipun fitur sensitif terenkripsi

**Threshold Detection:** Otomatis menggunakan kombinasi precision-recall curve dan ROC curve untuk menentukan optimal threshold.

---

## 📈 Metrik Evaluasi

Model dievaluasi menggunakan:
- **Binary Accuracy:** Akurasi klasifikasi fraud/non-fraud
- **PR-AUC:** Area under Precision-Recall curve (penting untuk imbalanced data)
- **Binary Cross-Entropy Loss:** Loss function untuk optimisasi

---

## 🔄 Integrasi dengan Global Model

Bank N merupakan bagian dari ekosistem federated learning yang lebih besar:
1. **Round 1:** Bank A-F melatih model global
2. **Round 2:** Bank N menggunakan model global R1 sebagai base
3. **Future Rounds:** Model Bank N dapat berkontribusi ke model global R3

Ini memungkinkan **continuous learning** sambil menjaga **privasi data** masing-masing bank.

---

## 🛡️ Keamanan & Privasi

Bank N mendemonstrasikan best practices dalam federated learning:
- 🔒 Data sensitif di-hash sebelum training
- 🔒 Model hanya mengakses representasi terenkripsi
- 🔒 Tidak ada raw data yang dishare antar banks
- 🔒 Only model weights aggregated di server global

---

## 📝 Catatan Penting

1. **Environment Setup:** Pastikan menggunakan environment yang sama untuk training dan testing
2. **Memory Management:** Threading dibatasi untuk menghindari crash (`OMP_NUM_THREADS=1`)
3. **CPU-Only:** Default menggunakan CPU (`CUDA_VISIBLE_DEVICES=""`)
4. **Checkpoint:** Gunakan flag `--resume` untuk melanjutkan training yang terinterupsi
5. **Preprocessing:** Pastikan `fitur_global.pkl` konsisten antara training dan testing

---

## 🎯 Kesimpulan

Bank N membuktikan bahwa federated learning dapat bekerja efektif bahkan dengan **privacy-enhanced data** yang sudah terenkripsi. Ini membuka peluang untuk konsorsium finansial dengan regulasi privasi yang sangat ketat untuk tetap berpartisipasi dalam collaborative machine learning tanpa mengorbankan keamanan data nasabah.
