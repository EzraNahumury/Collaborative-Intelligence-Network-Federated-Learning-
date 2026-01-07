# 🏦 Bank F - Bank Syariah Modern

## 📌 Deskripsi Bank F

**Bank F** merupakan bank syariah modern yang menyediakan produk dan layanan perbankan sesuai dengan prinsip-prinsip Syariah Islam. Bank ini memiliki karakteristik unik yang membedakannya dari bank konvensional:

### Karakteristik Utama

- **🕌 Prinsip Syariah**: Semua produk dan layanan mengikuti prinsip Syariah Islam
- **📊 Kategori Transaksi Khusus**: 
  - **Zakat** - Transaksi penyaluran zakat
  - **Syariah Investment** - Investasi berbasis syariah
  - **Halal Categories** - Kategori transaksi yang sesuai dengan prinsip halal
  - **Menghindari kategori non-halal** seperti:
    - Gambling (perjudian)
    - Alcohol (minuman keras)
    - Interest-based transactions (riba)
- **👥 Basis Nasabah**: Kuat di segmen retail dan UMKM (Usaha Mikro, Kecil, dan Menengah)
- **💼 Target Market**: Masyarakat yang mengutamakan transaksi sesuai syariah

### 🔍 Pola Penipuan Lokal

Bank F menghadapi pola penipuan yang mirip dengan bank ritel pada umumnya, yaitu:

- **🎯 Account Takeover untuk E-commerce**: 
  - Pengambilalihan akun nasabah untuk melakukan transaksi belanja online
  - Serupa dengan pola penipuan di bank retail konvensional
  
#### Tujuan Deteksi
Proyek ini menguji apakah **model Federated Learning** dapat:
- Menemukan pola penipuan yang **sama** meskipun konteks kategori transaksinya **berbeda**
- Mendeteksi anomali pada transaksi syariah yang serupa dengan pola penipuan di bank konvensional
- Mengadaptasi model detection terhadap karakteristik unik transaksi syariah

---

## 🚀 Cara Menjalankan Program

### Prasyarat

- Python 3.11+
- Virtual environment (venv) atau WSL
- Dependencies yang sudah terinstall (tensorflow, tensorflow-federated, pandas, numpy, dll.)

### Langkah-Langkah Eksekusi

#### **Tahap 1: Training Model Federated (bankF.py)**

1. **Masuk ke environment/WSL**
   ```bash
   # Jika menggunakan virtual environment
   source venv/bin/activate   # Linux/Mac
   .\venv\Scripts\activate    # Windows PowerShell
   
   # Atau jika menggunakan WSL
   wsl
   ```

2. **Jalankan bankF.py untuk training**
   ```bash
   python bankF.py --bank F_data --n_clients 5 --rounds 10 --batch_size 32
   ```

   **Parameter yang dapat disesuaikan:**
   - `--bank`: Kode bank (default: `F_data`)
   - `--data_dir`: Direktori data (default: `data`)
   - `--models_dir`: Direktori model (default: `models`)
   - `--global_dir`: Direktori fitur global (default: `models_global`)
   - `--n_clients`: Jumlah klien federated (default: `5`)
   - `--batch_size`: Ukuran batch (default: `32`)
   - `--rounds`: Jumlah round training (default: `10`)
   - `--lr_client`: Learning rate client (default: `5e-4`)
   - `--lr_server`: Learning rate server (default: `1e-3`)

3. **Output Training**
   ```
   🧠 Training Federated Client: bank_F_DATA
   📂 Data: data/bank_F_DATA.csv
   🔄 Fitur Global: models_global/fitur_global.pkl
   
   ✅ Mulai Federated Training ===========================
   [bank_F_DATA] Round 01 | acc=0.8234 | pr_auc=0.7456 | loss=0.4123
   [bank_F_DATA] Round 02 | acc=0.8567 | pr_auc=0.7823 | loss=0.3456
   ...
   ✅ Model bank_F_DATA disimpan di Models\saved_bank_F_DATA_tff
   ```

#### **Tahap 2: Testing Model Global (test.py)**

4. **Keluar dari environment/WSL**
   ```bash
   deactivate   # Jika menggunakan venv
   exit         # Jika menggunakan WSL
   ```

5. **Jalankan test.py untuk evaluasi**
   ```bash
   python test.py
   ```

6. **Output Testing**
   ```
   ======================================================================
   🌍 TEST MODEL GLOBAL di Data 
   ======================================================================
   ✅ Model & fitur global berhasil dimuat!
   
   Testing BANK F:
      ✅ Case 01 | Exp=0 | Pred=0 | Prob=0.2134
      ✅ Case 02 | Exp=1 | Pred=1 | Prob=0.8567
      ...
   
   📊 Akurasi Global di BANK F: 95.00% (19/20)
   
   🏁 TOTAL AKURASI (SEMUA TEST CASE)
   Total: 87.50% (21/24)
   
   💾 Total accuracy (87.50%) disimpan ke Models\saved_bank_F_DATA_tff\best_accuracy.txt
   ```

---

## 📁 Struktur Hasil Model (`Models\saved_bank_F_DATA_tff`)

Setelah menjalankan `bankF.py` dan `test.py`, hasil model akan disimpan di direktori `Models\saved_bank_F_DATA_tff` dengan struktur sebagai berikut:

```
Models/saved_bank_F_DATA_tff/
├── saved_model.pb                      # Model TensorFlow SavedModel
├── keras_metadata.pb                   # Metadata model Keras
├── fingerprint.pb                      # Fingerprint model
├── assets/                             # Asset tambahan model
├── variables/                          # Bobot model dalam format TensorFlow
│   ├── variables.data-00000-of-00001
│   └── variables.index
├── 20260105_125758.npz                 # Bobot model (timestamped NPZ format)
├── preprocess_bank_F_DATA.pkl          # Metadata preprocessing
├── history_bank_F_DATA.json            # Riwayat training (JSON)
├── accuracy_history.txt                # Riwayat akurasi per round (TSV)
└── best_accuracy.txt                   # Akurasi terbaik dari test global
```

### 📄 Penjelasan File-File Penting

#### 1. **saved_model.pb** 
   - Format: TensorFlow SavedModel
   - Isi: Arsitektur model neural network lengkap
   - Digunakan untuk: Loading model untuk inferensi

#### 2. **variables/** 
   - Format: TensorFlow checkpoint
   - Isi: Bobot (weights) dan bias dari setiap layer model
   - File:
     - `variables.data-00000-of-00001`: Data bobot
     - `variables.index`: Index mapping bobot

#### 3. **{timestamp}.npz** (contoh: `20260105_125758.npz`)
   - Format: NumPy compressed array
   - Isi: Snapshot bobot model dengan timestamp
   - Kegunaan: Backup bobot untuk versioning dan recovery

#### 4. **preprocess_bank_F_DATA.pkl**
   - Format: Pickle (joblib)
   - Isi: Metadata preprocessing yang berisi:
     - `MODE`: Mode preprocessing (DICT/LIST)
     - `NUM_COLS`: Kolom numerik
     - `CAT_COLS`: Kolom kategorikal
     - `HASHER_DIM`: Dimensi feature hashing
     - `SCALER`: Parameter scaling (min, range)
     - `FEATURE_DIM`: Total dimensi fitur
   - Kegunaan: Konsistensi preprocessing saat testing

#### 5. **history_bank_F_DATA.json**
   - Format: JSON
   - Isi: Log training per round:
     ```json
     [
       {
         "round": 1,
         "acc": 0.8234,
         "pr_auc": 0.7456,
         "loss": 0.4123
       },
       ...
     ]
     ```
   - Kegunaan: Analisis performa training

#### 6. **accuracy_history.txt**
   - Format: Tab-separated values (TSV)
   - Isi: Log lengkap dengan timestamp:
     ```
     bank    round   acc       pr_auc    loss      timestamp
     bank_F_DATA  1  0.823400  0.745600  0.412300  2026-01-05T12:57:58Z
     ```
   - Kegunaan: Tracking historical performance

#### 7. **best_accuracy.txt**
   - Format: Plain text (single line)
   - Isi: Nilai akurasi terbaik dari test global (bukan training)
   - Contoh: `0.875000`
   - Sumber: Dihasilkan oleh `test.py` berdasarkan evaluasi terhadap global test cases
   - Kegunaan: Quick reference untuk performa model

---

## 🔬 Arsitektur Model

Model menggunakan **Federated Learning** dengan algoritma **Weighted FedAvg** (Federated Averaging):

```python
Sequential([
    Input(shape=(FEATURE_DIM,))
    BatchNormalization()
    Dense(128, activation='relu', L2 regularization)
    Dropout(0.3)
    Dense(64, activation='relu', L2 regularization)
    Dropout(0.2)
    Dense(1, activation='sigmoid')
])
```

**Loss Function**: Binary Crossentropy  
**Metrics**: Binary Accuracy, PR-AUC (Precision-Recall Area Under Curve)  
**Optimizer**: Adam (client & server)

---

## 📊 Workflow Federated Learning

```
1. Load Data Lokal (bank_F_DATA.csv)
2. Preprocessing dengan fitur global (fitur_global.pkl)
3. Split data menjadi N clients (default: 5)
4. Federated Training:
   - Setiap client training lokal
   - Server agregasi bobot dengan weighted average
   - Update model global
   - Repeat untuk R rounds (default: 10)
5. Simpan model global ke Models\saved_bank_F_DATA_tff
6. Testing dengan test cases global (test.py)
7. Simpan akurasi terbaik ke best_accuracy.txt
```

---
