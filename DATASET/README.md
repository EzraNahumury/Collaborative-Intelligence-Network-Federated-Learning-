# 🏦 Dataset Generator untuk Bank A - N

Proyek ini menjelaskan alur lengkap untuk menghasilkan dataset dari 14 bank berbeda (Bank A hingga Bank N), membersihkan dataset untuk Bank G-N, dan membuat fitur global untuk Federated Learning.

---

## 📋 Daftar Isi

1. [Overview](#-overview)
2. [Struktur Direktori](#-struktur-direktori)
3. [Langkah-langkah Eksekusi](#-langkah-langkah-eksekusi)
   - [Step 1: Generate Dataset Bank A-N](#step-1-generate-dataset-bank-a-n)
   - [Step 2: Clean Dataset Bank G-N](#step-2-clean-dataset-bank-g-n)
   - [Step 3: Buat Fitur Global](#step-3-buat-fitur-global)
4. [Karakteristik Setiap Bank](#-karakteristik-setiap-bank)
5. [Output yang Dihasilkan](#-output-yang-dihasilkan)

---

## 🌟 Overview

Proyek ini mensimulasikan 14 bank berbeda dengan karakteristik unik masing-masing untuk keperluan Federated Learning dalam deteksi fraud. Setiap bank memiliki:
- **Pola transaksi unik**: Volume, nilai, dan frekuensi yang berbeda
- **Fitur khusus**: Beberapa bank memiliki fitur eksklusif
- **Pola fraud lokal**: Setiap bank memiliki jenis fraud yang berbeda
- **Kualitas data**: Beberapa bank memiliki sistem legacy dengan missing data

---

## 📁 Struktur Direktori

```
DATASET/
├── generate_dataset/          # Script untuk generate dan clean dataset
│   ├── bankA.py              # Generate dataset Bank A
│   ├── bankB.py              # Generate dataset Bank B
│   ├── ...                   # Generate dataset Bank C-N
│   ├── clean_bankG.py        # Clean dataset Bank G
│   ├── clean_bankH.py        # Clean dataset Bank H
│   ├── ...                   # Clean dataset Bank I-N
│   ├── fitur_global.py       # Generate fitur global (list)
│   └── fitur_global2.py      # Generate fitur global (dict)
├── data/                      # Dataset mentah Bank A-N
│   ├── bank_A_data.csv
│   ├── bank_B_data.csv
│   └── ...                   # bank_C_data.csv - bank_N_data.csv
├── data_cleaned/              # Dataset bersih Bank G-N
│   ├── bank_G_data_clean.csv
│   ├── bank_H_data_clean.csv
│   └── ...                   # bank_I_data_clean.csv - bank_N_data_clean.csv
├── models_global/             # Fitur global untuk FL
│   ├── fitur_global.pkl      # List fitur global
│   └── fitur_global_test.pkl # Dict fitur global lengkap
└── README.md                  # Dokumentasi ini
```

---

## 🚀 Langkah-langkah Eksekusi

### **Step 1: Generate Dataset Bank A-N**

Generate dataset untuk semua bank (Bank A hingga Bank N).

#### 📝 Cara Menjalankan:

```bash
# Aktifkan environment (jika menggunakan virtual environment)
# contoh: conda activate your_env atau source venv/bin/activate

# Masuk ke direktori generate_dataset
cd generate_dataset

# Generate dataset Bank A
python bankA.py

# Generate dataset Bank B
python bankB.py

# Generate dataset Bank C
python bankC.py

# Generate dataset Bank D
python bankD.py

# Generate dataset Bank E
python bankE.py

# Generate dataset Bank F
python bankF.py

# Generate dataset Bank G
python bankG.py

# Generate dataset Bank H
python bankH.py

# Generate dataset Bank I
python bankI.py

# Generate dataset Bank J
python bankJ.py

# Generate dataset Bank K
python bankK.py

# Generate dataset Bank L
python bankL.py

# Generate dataset Bank M
python bankM.py

# Generate dataset Bank N
python bankN.py
```

#### ✅ Output:

Setelah menjalankan semua script di atas, file-file berikut akan dibuat di folder `data/`:
- `bank_A_data.csv` - Dataset Bank A
- `bank_B_data.csv` - Dataset Bank B
- `bank_C_data.csv` - Dataset Bank C
- `bank_D_data.csv` - Dataset Bank D
- `bank_E_data.csv` - Dataset Bank E
- `bank_F_data.csv` - Dataset Bank F
- `bank_G_data.csv` - Dataset Bank G (mentah)
- `bank_H_data.csv` - Dataset Bank H (mentah)
- `bank_I_data.csv` - Dataset Bank I (mentah)
- `bank_J_data.csv` - Dataset Bank J (mentah)
- `bank_K_data.csv` - Dataset Bank K (mentah)
- `bank_L_data.csv` - Dataset Bank L (mentah)
- `bank_M_data.csv` - Dataset Bank M (mentah)
- `bank_N_data.csv` - Dataset Bank N (mentah)

---

### **Step 2: Clean Dataset Bank G-N**

Bank G hingga N memiliki data yang perlu dibersihkan karena berbagai alasan:
- **Bank G**: Sistem legacy dengan schema berbeda dan data korup
- **Bank H**: Data terstruktur berbeda (Challenger Bank Digital)
- **Bank I**: Missing features dan concept drift
- **Bank J**: High cardinality data
- **Bank K**: Non-transactional data
- **Bank L**: Extreme class imbalance / Time-series data
- **Bank M**: (sesuai karakteristik bank)
- **Bank N**: Encrypted data

#### 📝 Cara Menjalankan:

```bash
# Pastikan masih di direktori generate_dataset

# Clean dataset Bank G
python clean_bankG.py

# Clean dataset Bank H
python clean_bankH.py

# Clean dataset Bank I
python clean_bankI.py

# Clean dataset Bank J
python clean_bankJ.py

# Clean dataset Bank K
python clean_bankK.py

# Clean dataset Bank L
python clean_bankL.py

# Clean dataset Bank M
python clean_bankM.py

# Clean dataset Bank N
python clean_bankN.py
```

#### ✅ Output:

Setelah menjalankan semua script cleaning, file-file berikut akan dibuat di folder `data_cleaned/`:
- `bank_G_data_clean.csv` - Dataset Bank G yang sudah dibersihkan
- `bank_H_data_clean.csv` - Dataset Bank H yang sudah dibersihkan
- `bank_I_data_clean.csv` - Dataset Bank I yang sudah dibersihkan
- `bank_J_data_clean.csv` - Dataset Bank J yang sudah dibersihkan
- `bank_K_data_clean.csv` - Dataset Bank K yang sudah dibersihkan
- `bank_L_data_clean.csv` - Dataset Bank L yang sudah dibersihkan
- `bank_M_data_clean.csv` - Dataset Bank M yang sudah dibersihkan
- `bank_N_data_clean.csv` - Dataset Bank N yang sudah dibersihkan

---

### **Step 3: Buat Fitur Global**

Untuk Federated Learning, kita perlu membuat daftar fitur global yang konsisten di semua bank. Tersedia 2 versi:

#### **Opsi A: Fitur Global (List)**

Script `fitur_global.py` membuat list sederhana dari semua fitur yang tersedia.

```bash
# Jalankan script fitur_global
python fitur_global.py
```

**Output:**
- File: `models_global/fitur_global.pkl`
- Berisi: List nama kolom fitur setelah one-hot encoding
- Digunakan untuk: Memastikan konsistensi fitur antar client

#### **Opsi B: Fitur Global (Dict) - Recommended**

Script `fitur_global2.py` membuat dictionary lengkap dengan scaler dan metadata.

```bash
# Jalankan script fitur_global2
python fitur_global2.py
```

**Output:**
- File: `models_global/fitur_global_test.pkl`
- Berisi Dictionary dengan:
  - `FEATURE_COLS`: List semua kolom setelah encoding
  - `NUM_COLS`: List kolom numerik
  - `CAT_COLS`: List kolom kategorikal
  - `SCALER`: StandardScaler yang sudah di-fit
- Digunakan untuk: Preprocessing konsisten dan kompatibel dengan `test.py`

> **💡 Rekomendasi**: Gunakan `fitur_global2.py` karena lebih lengkap dan kompatibel dengan testing.

---

## 🏦 Karakteristik Setiap Bank

| Bank | Karakteristik | Fraud Pattern | Fitur Khusus |
|------|---------------|---------------|--------------|
| **A** | E-commerce, volume tinggi, nilai rendah | Rapid small transactions | Standard features |
| **B** | Corporate/B2B, volume rendah, nilai tinggi | Large international transactions | International focus |
| **C** | Retail physical, distribusi merata | Physical trans outside domicile | Location-based |
| **D** | Digital lending & installments | Fake loan applications | Loan features |
| **E** | Private banking, high-net-worth | Massive international transfers | Investment focus |
| **F** | Retail + UMKM + Syariah | E-commerce account takeover | Sharia products |
| **G** | Legacy system, data korup | (varies) | Different schema |
| **H** | Challenger digital bank | (varies) | device_risk_score |
| **I** | Old fraud detection, missing features | Old-style fraud (concept drift) | Simpler system |
| **J** | Stock & crypto broker | (varies) | asset_ticker |
| **K** | Asuransi jiwa & kesehatan | (varies) | Non-transactional |
| **L** | Microfinance / Payment Gateway | Extreme imbalance / Time-series | History-based |
| **M** | (sesuai implementasi) | (varies) | - |
| **N** | Encrypted data | Advanced privacy | Encrypted features |

---

## 📦 Output yang Dihasilkan

### 1. **Dataset Mentah** (`data/`)
- Format: CSV
- Jumlah: 14 file (Bank A-N)
- Konten: Data transaksi dengan kolom:
  - `transaction_id`
  - `customer_id`
  - `timestamp`
  - `amount` / `transaction_value`
  - `merchant_category` / `merchant_type`
  - `location`
  - `is_international`
  - `transaction_frequency_24h`
  - `is_fraud` (label)
  - Dan fitur lain sesuai karakteristik bank

### 2. **Dataset Bersih** (`data_cleaned/`)
- Format: CSV
- Jumlah: 8 file (Bank G-N)
- Konten: Data yang sudah:
  - Diperbaiki schema inconsistencies
  - Diisi missing values
  - Dinormalisasi format
  - Siap untuk training

### 3. **Fitur Global** (`models_global/`)

#### `fitur_global.pkl`
- **Type**: List
- **Content**: Nama-nama kolom fitur global
- **Size**: ~XX features (tergantung jumlah one-hot encoding)

#### `fitur_global_test.pkl`
- **Type**: Dictionary
- **Content**:
  ```python
  {
      "FEATURE_COLS": [...],  # List semua fitur
      "NUM_COLS": [...],      # Kolom numerik
      "CAT_COLS": [...],      # Kolom kategorikal
      "SCALER": StandardScaler()  # Fitted scaler
  }
  ```

---

## 🔧 Troubleshooting

### Error: File tidak ditemukan
**Solusi**: Pastikan Anda berada di direktori yang benar dan sudah menjalankan Step 1 sebelum Step 2.

### Error: Import module tidak ditemukan
**Solusi**: Pastikan environment Python sudah diaktifkan dan semua dependencies terinstall:
```bash
pip install pandas numpy scikit-learn joblib
```

### Error: Permission denied
**Solusi**: Pastikan folder `data/`, `data_cleaned/`, dan `models_global/` ada dan memiliki write permission.

---

## 📝 Catatan Penting

1. **Urutan Eksekusi**: Harus mengikuti urutan Step 1 → Step 2 → Step 3
2. **Bank A-F**: Tidak perlu cleaning, data sudah bersih
3. **Bank G-N**: Wajib di-clean sebelum membuat fitur global
4. **Fitur Global**: Gunakan versi 2 (`fitur_global2.py`) untuk kompatibilitas dengan testing
5. **Environment**: Pastikan menggunakan Python 3.7+ dan semua dependencies terinstall

---


