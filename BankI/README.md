# 🏦 Bank I - Dokumentasi Federated Learning (Round 2)

## 📋 Deskripsi Bank I

**Bank I** merupakan **"Bank Pembangunan Daerah"** yang berpartisipasi dalam iterasi kedua federated learning. Bank ini memiliki karakteristik sistem yang lebih sederhana dibandingkan bank-bank lain dalam federasi.

### 🎯 Karakteristik Utama

#### 1️⃣ **Sistem Sederhana dengan Keterbatasan Tracking**
Bank I memiliki sistem legacy yang tidak melacak semua metrik secara komprehensif seperti bank-bank modern lainnya. Hal ini mencerminkan kondisi nyata dari bank pembangunan daerah yang masih menggunakan infrastruktur teknologi tradisional.

#### 2️⃣ **Missing Feature: `transaction_frequency_24h`**
Salah satu keterbatasan utama adalah **tidak adanya data `transaction_frequency_24h`**. Feature ini penting untuk mendeteksi pola penipuan modern yang melibatkan multiple transaksi dalam waktu singkat. Ketiadaan feature ini membuat model harus bergantung pada fitur-fitur lain yang tersedia.

#### 3️⃣ **Concept Drift: Pola Penipuan "Gaya Lama"**
Bank I mengalami **concept drift** yang signifikan. Pola penipuan yang terjadi di Bank I adalah **"old-style fraud"** dengan karakteristik:
- ✅ **Satu transaksi domestik dengan nilai sangat besar**
- ✅ **Tidak ada sinyal kompleks lainnya**
- ✅ **Pola sederhana tanpa multi-channel atau multi-device indicators**

Pola ini berbeda dengan pola penipuan modern yang biasanya melibatkan:
- ❌ Multiple small transactions (velocity attacks)
- ❌ Cross-border sophisticated schemes
- ❌ Complex behavioral patterns

> [!WARNING]
> **Concept Drift Impact**: Model global yang dilatih dengan data dari Bank A-F (Round 1) mungkin memiliki performa yang lebih rendah pada data Bank I karena perbedaan distribusi pola penipuan. Inilah alasan pentingnya federated learning round 2 - untuk mengadaptasi model terhadap pola baru ini.

---

## 🔄 Konteks Federated Learning Round 2

Bank I adalah bagian dari **Federated Learning Iterasi Ke-2**. Pada iterasi ini:

- 📦 Model global dari **Round 1** (hasil agregasi Bank A, B, C, D, E, F) digunakan sebagai **base model**
- 🔧 Model di-fine-tune dengan data lokal Bank I untuk mengatasi concept drift
- 🌐 Hasilnya akan dikontribusikan kembali ke model global untuk meningkatkan generalisasi

### 📂 Pre-trained Model Global Round 1
Model global dari iterasi pertama tersimpan di:
```
models_global_round1/global_savedmodel/
```

Model ini diload sebagai inisialisasi bobot awal sebelum training dimulai.

---

## 🚀 Cara Menjalankan

### ⚙️ Prerequisites
- Python 3.11+
- Virtual environment (venv)
- TensorFlow Federated (TFF)
- Dependencies lainnya (ada di `requirements.txt`)

---

### 📝 Step-by-Step Execution

#### **Step 1: Aktivasi Environment**

Masuk ke dalam virtual environment atau WSL:

```bash
# Untuk Windows (PowerShell/CMD)
cd "c:\KP\MATERI\BANK - FIX\BankI"
.\venv\Scripts\activate

# Untuk WSL/Linux
source venv/bin/activate
```

Setelah aktivasi berhasil, prompt Anda akan berubah menampilkan `(venv)`.

---

#### **Step 2: Training Model Federated (Dalam Environment)**

Jalankan script `bankI.py` untuk melatih model:

```bash
python bankI.py
```

**Apa yang terjadi di step ini?**
- 📂 Load data dari `data/bank_I_data_clean.csv`
- 🌍 Load fitur global dari `models_global/fitur_global.pkl`
- 🧠 Load model global Round 1 sebagai base model
- 👥 Split data menjadi 3 client federated (default)
- 🔁 Training selama 10 rounds menggunakan Weighted FedAvg
- 💾 Save model hasil training ke `models_round2/saved_bank_I_tff/`

**Output yang dihasilkan:**
```
🚀 Training Federated Client (R2): bank_I
📂 Data         : data/bank_I_data_clean.csv
🌐 Fitur Global : models_global/fitur_global.pkl
🌍 Global R1    : models_global_round1/global_savedmodel
💾 Output model : models_round2/saved_bank_I_tff

✅ Base model diinisialisasi dari Global R1 (input_dim=...)
✅ 3 klien federated data siap digunakan.

🚀 Mulai Federated Training (ROUND 2) ===========================
[bank_I] Round 01 | acc=0.8745 | pr_auc=0.6523 | loss=0.3421
[bank_I] Round 02 | acc=0.8892 | pr_auc=0.6891 | loss=0.3102
...
```

---

#### **Step 3: Keluar dari Environment**

Setelah training selesai, keluar dari virtual environment:

```bash
deactivate
```

> [!IMPORTANT]
> Pastikan Anda keluar dari environment/WSL sebelum menjalankan `test.py` untuk menghindari konflik dependency.

---

#### **Step 4: Testing Model (Di Luar Environment)**

Jalankan script `test.py` untuk mengevaluasi model:

```bash
python test.py
```

**Apa yang terjadi di step ini?**
- 🧠 Load model yang sudah dilatih dari `models_round2/saved_bank_I_tff/`
- 🔎 Load preprocessing config dari `models_global/fitur_global_test.pkl`
- 🧪 Menguji model dengan test cases dari Bank A, B, C, D, E, F
- ⚙️ Menggunakan automatic threshold optimization (Precision-Recall + ROC)
- 📊 Menghitung akurasi per bank dan total akurasi
- 💾 Menyimpan hasil akurasi terbaik ke `best_accuracy.txt`

**Output yang dihasilkan:**
```
======================================================================
🌍 TEST MODEL GLOBAL di Data 
======================================================================
✅ Model & fitur global berhasil dimuat!

   ✅ Case 01 | Exp=0 | Pred=0 | Prob=0.1234
   ✅ Case 02 | Exp=1 | Pred=1 | Prob=0.8765
   ...

 Akurasi Global di BANK A: 87.50% (7/8)
----------------------------------------------------------------------

======================================================================
🏁 TOTAL AKURASI (SEMUA TEST CASE)
======================================================================
 Total: 85.42% (41/48)
----------------------------------------------------------------------

💾 Total accuracy (85.42%) disimpan ke models_round2/saved_bank_I_tff/best_accuracy.txt
```

---

## 📦 Isi Folder `models_round2\saved_bank_I_tff`

Setelah menjalankan kedua script, folder ini akan berisi beberapa file penting:

### 1. **Keras SavedModel Format** 📁
| File/Folder | Deskripsi |
|-------------|-----------|
| `saved_model.pb` | Protocol Buffer yang berisi graph model TensorFlow |
| `keras_metadata.pb` | Metadata Keras untuk model architecture |
| `variables/` | Folder berisi bobot model (weights) dalam format TensorFlow |
| `variables/variables.data-00000-of-00001` | File data bobot aktual |
| `variables/variables.index` | Index file untuk variables |
| `assets/` | Folder untuk asset tambahan (biasanya kosong) |
| `fingerprint.pb` | Fingerprint untuk tracking versi model |

> [!TIP]
> Format SavedModel ini dapat langsung di-load dengan `tf.keras.models.load_model()` atau `TFSMLayer()` untuk inference.

### 2. **Bobot Model (NPZ Format)** 💾
| File | Deskripsi |
|------|-----------|
| `YYYYMMDD_HHMMSS.npz` | Bobot model dalam format NumPy compressed archive dengan timestamp (contoh: `20260105_144332.npz`) |

Format NPZ berguna untuk:
- Backup bobot model yang lebih portabel
- Loading cepat tanpa perlu load seluruh graph
- Kompatibilitas dengan NumPy ecosystem

### 3. **Preprocessing Configuration** 🔧
| File | Deskripsi |
|------|-----------|
| `preprocess_bank_I.pkl` | Metadata preprocessing yang berisi informasi kolom numerik, kategorikal, dimensi hasher, dan scaler |

**Isi dari `preprocess_bank_I.pkl`:**
```python
{
    "MODE": "DICT",  # atau "LIST" tergantung fitur global
    "NUM_COLS": [...],  # Daftar kolom numerik
    "CAT_COLS": [...],  # Daftar kolom kategorikal
    "HASHER_DIM": 10,   # Dimensi feature hashing
    "SCALER": {...},    # MinMaxScaler params atau sklearn object
    "FEATURE_DIM": 47   # Total dimensi fitur setelah preprocessing
}
```

### 4. **Training History & Metrics** 📊
| File | Deskripsi |
|------|-----------|
| `history_bank_I.json` | History lengkap per round: accuracy, PR-AUC, dan loss |
| `accuracy_history.txt` | Log akurasi dalam format TSV dengan timestamp |
| `best_accuracy.txt` | Total akurasi terbaik dari evaluasi test.py (format: `0.854200`) |

**Contoh `history_bank_I.json`:**
```json
[
  {"round": 1, "acc": 0.8745, "pr_auc": 0.6523, "loss": 0.3421},
  {"round": 2, "acc": 0.8892, "pr_auc": 0.6891, "loss": 0.3102},
  ...
]
```

### 5. **Checkpoint untuk Resume Training** 🔄
| Folder | Deskripsi |
|--------|-----------|
| `ckpt/` | Folder checkpoint untuk melanjutkan training |
| `ckpt/server_state.npz` | State server federated untuk resume |
| `ckpt/history.json` | Copy history untuk resume |

> [!NOTE]
> Untuk melanjutkan training dari checkpoint, gunakan flag `--resume`:
> ```bash
> python bankI.py --resume
> ```

---

## 📈 Evaluasi dan Interpretasi

### 🎯 Metrik yang Digunakan

1. **Binary Accuracy**: Akurasi klasifikasi binary (fraud vs non-fraud)
2. **PR-AUC** (Precision-Recall Area Under Curve): Lebih cocok untuk imbalanced dataset
3. **Loss**: Binary crossentropy loss

### 🔍 Hal yang Perlu Diperhatikan

- **Performa pada data Bank I sendiri** mungkin lebih tinggi karena model sudah di-fine-tune
- **Performa pada test cases Bank A-F** menunjukkan kemampuan generalisasi model
- **Concept drift** dapat terlihat dari perbedaan performa antara Bank I dan bank lain
- Model yang baik seharusnya tetap mempertahankan performa pada pola fraud lama (Bank A-F) sambil belajar pola baru (Bank I)

---

## 🔧 Konfigurasi Lanjutan

### Parameter Training yang Dapat Diubah

```bash
python bankI.py \
  --n_clients 5 \           # Jumlah client federated (default: 3)
  --batch_size 64 \         # Batch size (default: 32)
  --rounds 20 \             # Jumlah rounds training (default: 10)
  --lr_client 0.001 \       # Learning rate client (default: 5e-4)
  --lr_server 0.01 \        # Learning rate server (default: 1e-3)
  --resume                  # Resume dari checkpoint
```

### Parameter Testing yang Dapat Diubah

Edit file `test.py` pada bagian konfigurasi:

```python
THRESHOLD_MODE = "AUTO"   # "AUTO" atau "MANUAL"
THRESHOLD_MANUAL = 0.5    # Threshold jika mode MANUAL
MODEL_PATH = "models_round2/saved_bank_I_tff"
PREPROC_PATH = "models_global/fitur_global_test.pkl"
```

---



## ⚠️ Troubleshooting

### Problem: Model tidak bisa load Global R1
**Solusi**: Pastikan folder `models_global_round1/global_savedmodel/` ada dan berisi model yang valid. Jika tidak ada, script akan otomatis menggunakan model baru.

### Problem: Fitur tidak sesuai dengan global features
**Solusi**: Pastikan `models_global/fitur_global.pkl` kompatibel dengan data Bank I. Script sudah handle missing columns dengan mengisi nilai 0.

### Problem: Test.py error "Module not found"
**Solusi**: Pastikan Anda sudah keluar dari environment sebelum menjalankan `test.py`.

---
