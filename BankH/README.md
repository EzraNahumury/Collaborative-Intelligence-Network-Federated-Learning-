# 🏦 Bank H - Challenger Bank Digital

## 📖 Deskripsi

**Bank H** adalah **Challenger Bank Digital** yang merupakan bagian dari implementasi **Federated Learning Iterasi ke-2**. Sebagai bank digital baru, Bank H memiliki karakteristik yang unik dibandingkan dengan bank-bank tradisional lainnya.

---

## 🎯 Karakteristik Bank H

### Masalah & Tantangan

Sebagai **bank baru dengan platform digital**, Bank H melacak **metrik modern** yang tidak dimiliki oleh bank-bank tradisional lainnya. Bank H memiliki fitur eksklusif yang menjadi pembeda utama:

- **`device_risk_score`**: Skor risiko dari perangkat yang digunakan untuk melakukan transaksi
  - Fitur ini membantu mengidentifikasi transaksi yang dilakukan dari perangkat yang tidak biasa atau mencurigakan
  - Meningkatkan kemampuan deteksi fraud dengan menganalisis perilaku perangkat pengguna

### Kualitas Data

- ✅ **Data sangat bersih** - Bank H memiliki data berkualitas tinggi karena merupakan sistem digital native
- ⚠️ **Struktur berbeda** - Memiliki skema dan fitur yang berbeda dari bank tradisional (Bank A-G)
- 🆕 **Fitur eksklusif** - Memiliki fitur modern seperti `device_risk_score` yang tidak ada di bank lain

### Posisi dalam Federated Learning

Bank H merupakan bagian dari **Round 2 Federated Learning**, yang berarti:
- Menggunakan **model global dari iterasi pertama** (hasil pembelajaran Bank A-F) sebagai base model
- Melakukan fine-tuning dengan data lokalnya yang memiliki pola fraud modern
- Kontribusi Bank H akan memperkaya model global dengan pola fraud berbasis perangkat digital

---

## 🚀 Cara Running

### Langkah 1: Training Model Bank H

Pertama, masuk ke environment dan jalankan training:

```bash
# Masuk ke environment WSL/Virtual Environment
# Contoh untuk WSL:
wsl

# Atau jika menggunakan virtual environment:
source venv/bin/activate  # Linux/Mac
# atau
.\venv\Scripts\activate   # Windows

# Jalankan training Bank H
python bankH.py
```

**Proses yang terjadi:**
- Load data Bank H dari `data/bank_H_data_clean.csv`
- Load fitur global dari model Round 1 (`models_global/fitur_global.pkl`)
- Load base model global Round 1 dari `models_global_round1/global_savedmodel`
- Preprocessing data sesuai dengan fitur global
- Split data menjadi beberapa klien federated (default: 3 klien)
- Training menggunakan TensorFlow Federated
- Menyimpan hasil ke `models_round2/saved_bank_H_tff/`

### Langkah 2: Testing Model

Keluar dari environment WSL/virtual environment, lalu jalankan testing:

```bash
# Keluar dari environment
exit           # untuk WSL
# atau
deactivate     # untuk virtual environment

# Jalankan testing
python test.py
```

**Proses yang terjadi:**
- Load model yang telah ditraining dari `models_round2/saved_bank_H_tff/`
- Load preprocessing configuration
- Testing dengan test cases dari semua bank (A-G)
- Menghitung akurasi per bank dan total akurasi
- Menyimpan hasil akurasi terbaik ke `best_accuracy.txt`

---

## 📦 Struktur Saved Model

Hasil training dan testing disimpan di direktori:
```
models_round2/saved_bank_H_tff/
```

### Isi Direktori Saved Model

| File/Folder | Deskripsi |
|-------------|-----------|
| **`saved_model.pb`** | Model TensorFlow dalam format protobuf - model utama yang telah ditraining |
| **`variables/`** | Direktori berisi weights dan parameters dari neural network |
| **`assets/`** | Asset tambahan yang diperlukan oleh model (jika ada) |
| **`keras_metadata.pb`** | Metadata Keras untuk model architecture dan configuration |
| **`fingerprint.pb`** | Fingerprint unik untuk verifikasi integritas model |
| **`preprocess_bank_H.pkl`** | Configuration preprocessing (scaler, encoder, dll) yang digunakan untuk data Bank H |
| **`history_bank_H.json`** | Riwayat training metrics (loss, accuracy) per epoch/round |
| **`best_accuracy.txt`** | Akurasi terbaik yang dicapai model saat testing |
| **`accuracy_history.txt`** | Riwayat akurasi training per round federated learning |
| **`ckpt/`** | Checkpoint weights selama proses training untuk recovery |
| **`*.npz`** | NumPy archive berisi server state atau additional training artifacts |

### Penjelasan File Penting

#### 1. **Model Files** (`saved_model.pb`, `variables/`)
Model neural network yang telah ditraining menggunakan TensorFlow Federated dengan base model dari Global Round 1.

#### 2. **Preprocessing Configuration** (`preprocess_bank_H.pkl`)
Berisi:
- Scaler untuk normalisasi fitur numerik
- Encoder untuk fitur kategorikal
- Feature names dan dimensi
- Mapping fitur lokal ke fitur global

#### 3. **Training History** (`history_bank_H.json`)
Riwayat lengkap proses training:
- Loss per round
- Accuracy per round
- Metrics lainnya selama federated learning

#### 4. **Best Accuracy** (`best_accuracy.txt`)
Akurasi terbaik dari hasil testing dengan test cases global (dari semua bank A-G).

#### 5. **Checkpoint** (`ckpt/`)
Menyimpan state model di setiap epoch/round untuk:
- Recovery jika training terinterupsi
- Evaluasi model pada berbagai tahap training
- Rollback jika diperlukan

---

## 🔧 Konfigurasi Default

Berikut adalah konfigurasi default yang digunakan:

```python
--bank_name: "H"
--data_path: "data/bank_H_data_clean.csv"
--models_dir: "models_round2"
--global_dir: "models_global"
--global_feats: "fitur_global.pkl"
--global_model_r1: "models_global_round1/global_savedmodel"
--n_clients: 3
--batch_size: 32
--epochs: 5
--learning_rate: 0.001
```

---

## 📊 Fitur Eksklusif Bank H

### device_risk_score
Fitur ini merupakan **inovasi Bank H** sebagai challenger bank digital:
- Menilai risiko dari perangkat yang digunakan untuk transaksi
- Mempertimbangkan faktor seperti:
  - Perangkat yang tidak dikenal
  - Lokasi perangkat yang mencurigakan
  - Pola penggunaan perangkat yang abnormal
  - Konfigurasi keamanan perangkat

Fitur ini sangat efektif untuk mendeteksi:
- Account takeover dari perangkat yang tidak dikenal
- Transaksi fraud menggunakan emulator atau perangkat virtual
- Serangan dari botnet atau automated tools

---

## 📝 Catatan Penting

1. **Model Global Round 1** harus tersedia di `models_global_round1/global_savedmodel`
2. **Fitur global** harus tersedia di `models_global/fitur_global.pkl`
3. Data Bank H memiliki **struktur yang berbeda** namun akan di-align dengan fitur global
4. Training menggunakan **TensorFlow Federated** untuk privacy-preserving learning
5. Model Bank H akan berkontribusi pada **Global Model Round 2**


