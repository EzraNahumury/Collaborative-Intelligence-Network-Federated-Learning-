# 📚 API Documentation - Federated Server

Dokumentasi lengkap untuk semua endpoint yang tersedia di **Federated Aggregation Server**.

---

## 🌐 Base URL
```
Production: https://your-railway-app.railway.app
Development: http://localhost:8080
```

---

## 📋 Daftar Endpoints

1. [GET /](#1-get-)
2. [POST /upload-model](#2-post-upload-model)
3. [POST /aggregate](#3-post-aggregate)
4. [GET /logs](#4-get-logs)
5. [GET /download-global](#5-get-download-global)
6. [GET /download/:filename](#6-get-downloadfilename)
7. [DELETE /delete/:filename](#7-delete-deletefilename)
8. [POST /delete-model](#8-post-delete-model)
9. [GET /accuracy/:client](#9-get-accuracyclient)

---

## 1. GET `/`

**Deskripsi**: Endpoint home untuk mengecek status server dan melihat daftar endpoint yang tersedia.

### Request
```http
GET / HTTP/1.1
```

### Response (200 OK)
```json
{
  "message": "🌍 Federated Aggregation Server aktif!",
  "status": "online",
  "endpoints": {
    "/upload-model": "Upload model lokal dari client (POST)",
    "/aggregate": "Lakukan agregasi global (POST)",
    "/logs": "Lihat file di models (GET)",
    "/download/<filename>": "Download file (GET)",
    "/delete/<filename>": "Hapus file (DELETE)",
    "/delete-model": "Hapus file via POST JSON",
    "/accuracy/<client>": "Ambil best accuracy & riwayat (GET)"
  }
}
```

---

## 2. POST `/upload-model`

**Deskripsi**: Upload model lokal dari client ke server. Endpoint ini menerima bobot model dalam format compressed NPZ yang di-encode base64, serta optional metrics (accuracy dan history).

### Request Headers
```
Content-Type: application/json
```

### Request Body
```json
{
  "client": "BANK_A",
  "compressed_weights": "<base64 encoded npz data>",
  "metrics": {
    "best_accuracy": 0.9123,
    "history": [
      {
        "round": 1,
        "accuracy": 0.8945,
        "timestamp": "2026-01-06T08:30:00Z"
      },
      {
        "round": 2,
        "accuracy": 0.9123,
        "timestamp": "2026-01-06T08:35:00Z"
      }
    ]
  }
}
```

**Atau format alternatif:**
```json
{
  "client": "BANK_B",
  "compressed_weights": "<base64 encoded npz data>",
  "accuracy": 0.9234
}
```

### Response (200 OK)
```json
{
  "status": 200,
  "client": "BANK_A",
  "saved_weights": "models/BANK_A_weights.npz",
  "message": "model uploaded",
  "metrics": {
    "best_path": "models/logs/BANK_A_best_accuracy.txt",
    "reported_accuracy": 0.9123,
    "written_best": true,
    "history_written": 2,
    "history_path": "models/logs/BANK_A_accuracy_history.txt"
  }
}
```

### Response (400 Bad Request)
```json
{
  "status": "error",
  "message": "client missing"
}
```

### Response (500 Internal Server Error)
```json
{
  "status": "error",
  "message": "failed to decode/load npz: <error details>"
}
```

---

## 3. POST `/aggregate`

**Deskripsi**: Melakukan agregasi model dari semua client menggunakan metode **Federated Averaging (FedAvg)**. Endpoint ini akan menggabungkan semua model lokal yang telah di-upload dan menghasilkan model global. Endpoint ini juga secara otomatis melakukan testing terhadap model global yang dihasilkan.

### Request Headers
```
Content-Type: application/json
```

### Request Body (Optional)
```json
{
  "data_sizes": {
    "BANK_A_weights.npz": 10000,
    "BANK_B_weights.npz": 8000,
    "BANK_C_weights.npz": 12000
  }
}
```

### Response (200 OK)
```json
{
  "status": "success",
  "method": "FedAvg",
  "num_clients": 6,
  "num_layers": 8,
  "total_parameters": 45632,
  "avg_global_weight": 0.0234567,
  "avg_global_weight_change_percent": 2.345678,
  "saved": "models/global_model_fedavg_20260106_153000.npz",
  "client_mean_weight": {
    "BANK_A_weights.npz": 0.0234,
    "BANK_B_weights.npz": 0.0245,
    "BANK_C_weights.npz": 0.0221,
    "BANK_D_weights.npz": 0.0239,
    "BANK_E_weights.npz": 0.0228,
    "BANK_F_weights.npz": 0.0237
  },
  "client_mean_weight_percentage": {
    "BANK_A_weights.npz": 16.5432,
    "BANK_B_weights.npz": 17.3214,
    "BANK_C_weights.npz": 15.6234,
    "BANK_D_weights.npz": 16.8976,
    "BANK_E_weights.npz": 16.1234,
    "BANK_F_weights.npz": 16.7910
  },
  "fedavg_data_contribution_percentage": {
    "BANK_A_weights.npz": 33.3333,
    "BANK_B_weights.npz": 26.6667,
    "BANK_C_weights.npz": 40.0000
  },
  "total_accuracy": 87.5,
  "test_results": {
    "total_accuracy": 87.5,
    "total_correct": 21,
    "total_cases": 24,
    "per_bank_results": [
      {
        "bank": "BANK A",
        "accuracy": 100.0,
        "correct": 4,
        "total": 4
      },
      {
        "bank": "BANK B",
        "accuracy": 75.0,
        "correct": 3,
        "total": 4
      },
      {
        "bank": "BANK C",
        "accuracy": 100.0,
        "correct": 4,
        "total": 4
      },
      {
        "bank": "BANK D",
        "accuracy": 75.0,
        "correct": 3,
        "total": 4
      },
      {
        "bank": "BANK E",
        "accuracy": 100.0,
        "correct": 4,
        "total": 4
      },
      {
        "bank": "BANK F",
        "accuracy": 75.0,
        "correct": 3,
        "total": 4
      }
    ],
    "test_model_used": "models/saved_bank_B_DATA_tff"
  }
}
```

### Response (400 Bad Request - Tidak Cukup Model)
```json
{
  "status": "error",
  "message": "Hanya ditemukan 1 model lokal (BANK_A_weights.npz). Minimal 2 model diperlukan untuk melakukan Federated Averaging.",
  "found_models": [
    "BANK_A_weights.npz"
  ],
  "required": 2,
  "current": 1
}
```

### Response (500 Internal Server Error)
```json
{
  "status": "error",
  "message": "<error details>"
}
```

---

## 4. GET `/logs`

**Deskripsi**: Mendapatkan daftar semua file model (.npz) yang ada di folder `models`, termasuk model client dan model global hasil agregasi. Hasil di-sort berdasarkan waktu modifikasi terbaru.

### Request
```http
GET /logs HTTP/1.1
```

### Response (200 OK)
```json
{
  "status": 200,
  "files": [
    {
      "client": "GLOBAL",
      "name": "global_model_fedavg_20260106_153000.npz",
      "message": "Model global hasil agregasi FedAvg",
      "timestamp": "2026-01-06T08:30:00Z"
    },
    {
      "client": "BANK_A",
      "name": "BANK_A_weights.npz",
      "message": "Model dari client BANK_A",
      "timestamp": "2026-01-06T08:25:00Z"
    },
    {
      "client": "BANK_B",
      "name": "BANK_B_weights.npz",
      "message": "Model dari client BANK_B",
      "timestamp": "2026-01-06T08:20:00Z"
    },
    {
      "client": "BANK_C",
      "name": "BANK_C_weights.npz",
      "message": "Model dari client BANK_C",
      "timestamp": "2026-01-06T08:15:00Z"
    }
  ],
  "message": "Daftar model berhasil diambil dari server"
}
```

### Response (500 Internal Server Error)
```json
{
  "status": "error",
  "message": "<error details>"
}
```

---

## 5. GET `/download-global`

**Deskripsi**: Download model global terbaru hasil agregasi FedAvg. File akan diunduh dengan nama yang mengandung timestamp saat download.

### Request
```http
GET /download-global HTTP/1.1
```

### Response (200 OK)
- **Content-Type**: `application/octet-stream`
- **Content-Disposition**: `attachment; filename="global_model_fedavg_20260106_153400.npz"`
- **Custom Headers**:
  - `X-File-Name`: `global_model_fedavg_20260106_153400.npz`
  - `X-File-Size`: `245632` (bytes)
  - `X-Last-Modified`: `1704537600.123456` (Unix timestamp)
  - `X-Description`: `Model global terbaru hasil Federated Averaging`

**Body**: Binary data (file NPZ)

### Response (404 Not Found)
```json
{
  "status": "error",
  "message": "Model global belum tersedia. Silakan jalankan endpoint /aggregate setelah minimal 2 model lokal terkirim.",
  "hint": "Pastikan minimal dua client telah mengunggah model mereka."
}
```

### Response (500 Internal Server Error)
```json
{
  "status": "error",
  "message": "<error details>"
}
```

---

## 6. GET `/download/:filename`

**Deskripsi**: Download file spesifik dari folder `models` berdasarkan nama file.

### Request
```http
GET /download/BANK_A_weights.npz HTTP/1.1
```

### Response (200 OK)
- **Content-Type**: `application/octet-stream`
- **Content-Disposition**: `attachment; filename="BANK_A_weights.npz"`

**Body**: Binary data (file NPZ)

### Response (404 Not Found)
```json
{
  "status": "error",
  "message": "BANK_A_weights.npz tidak ditemukan"
}
```

---

## 7. DELETE `/delete/:filename`

**Deskripsi**: Menghapus file model tertentu dari server. Endpoint ini juga akan menghapus log accuracy dan history yang terkait dengan client tersebut.

### Request
```http
DELETE /delete/BANK_A_weights.npz HTTP/1.1
```

### Response (200 OK)
```json
{
  "status": "success",
  "deleted": "BANK_A_weights.npz",
  "deleted_logs": {
    "best": true,
    "history": true,
    "folder_best": false,
    "folder_history": false
  }
}
```

### Response (400 Bad Request)
```json
{
  "status": "error",
  "message": "Invalid filename"
}
```

### Response (404 Not Found)
```json
{
  "status": "error",
  "message": "File BANK_A_weights.npz tidak ditemukan"
}
```

### Response (500 Internal Server Error)
```json
{
  "status": "error",
  "message": "<error details>"
}
```

---

## 8. POST `/delete-model`

**Deskripsi**: Menghapus model client menggunakan metode POST dengan body JSON. Alternatif dari endpoint DELETE untuk kompatibilitas dengan client yang tidak support DELETE method dengan path parameter.

### Request Headers
```
Content-Type: application/json
```

### Request Body (Option 1 - dengan client name)
```json
{
  "client": "BANK_A"
}
```

### Request Body (Option 2 - dengan filename)
```json
{
  "filename": "BANK_B_weights.npz"
}
```

### Response (200 OK)
```json
{
  "status": "success",
  "deleted": "BANK_A_weights.npz",
  "deleted_logs": {
    "best": true,
    "history": true,
    "folder_best": false,
    "folder_history": false
  }
}
```

### Response (400 Bad Request)
```json
{
  "status": "error",
  "message": "client atau filename required"
}
```

### Response (404 Not Found)
```json
{
  "status": "error",
  "message": "BANK_A_weights.npz tidak ditemukan"
}
```

### Response (500 Internal Server Error)
```json
{
  "status": "error",
  "message": "<error details>"
}
```

---

## 9. GET `/accuracy/:client`

**Deskripsi**: Mendapatkan best accuracy dan riwayat accuracy (20 entry terakhir) untuk client tertentu. Endpoint ini akan mencari di folder `models/logs/` terlebih dahulu, kemudian mencari di folder client di `models/` jika tidak ditemukan.

### Request
```http
GET /accuracy/BANK_A HTTP/1.1
```

### Response (200 OK - Data Ditemukan)
```json
{
  "client": "BANK_A",
  "best_accuracy": 0.9123,
  "history_tail": [
    "2026-01-06T08:20:00Z\t0.8945",
    "2026-01-06T08:25:00Z\t0.9050",
    "2026-01-06T08:30:00Z\t0.9123"
  ],
  "source": "models/logs/BANK_A_best_accuracy.txt"
}
```

### Response (200 OK - Data Tidak Ditemukan)
```json
{
  "client": "BANK_A",
  "best_accuracy": null,
  "history_tail": [],
  "source": null
}
```

### Response (500 Internal Server Error)
```json
{
  "status": "error",
  "message": "<error details>"
}
```

---

## 📝 Catatan Penting

### CORS Configuration
Server dikonfigurasi untuk menerima request dari:
- Frontend URL yang di-set di environment variable `FRONTEND_URL`
- `http://localhost:3000` (untuk development)

### Folder Structure
```
models/
├── logs/
│   ├── BANK_A_best_accuracy.txt
│   ├── BANK_A_accuracy_history.txt
│   └── ...
├── BANK_A_weights.npz
├── BANK_B_weights.npz
├── global_model_fedavg_20260106_153000.npz
└── last_avg_weight.json
```

### Path Safety
Semua endpoint yang menerima filename menggunakan fungsi `safe_model_path()` untuk mencegah path traversal attacks (misalnya `../../etc/passwd`).

### Automatic Testing
Ketika endpoint `/aggregate` dipanggil, server secara otomatis akan:
1. Melakukan agregasi model dengan FedAvg
2. Menjalankan testing terhadap 6 bank test cases (BANK A sampai F)
3. Menghitung total accuracy dan accuracy per bank
4. Mengembalikan hasil testing dalam response JSON

---

## 🔧 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FRONTEND_URL` | URL frontend yang diizinkan untuk CORS | `http://localhost:3000` |
| `PORT` | Port server | `8080` |

---

## 🚀 Cara Menggunakan

### 1. Upload Model dari Client
```bash
curl -X POST http://localhost:8080/upload-model \
  -H "Content-Type: application/json" \
  -d '{
    "client": "BANK_A",
    "compressed_weights": "<base64_encoded_npz>",
    "accuracy": 0.9123
  }'
```

### 2. Agregasi Model
```bash
curl -X POST http://localhost:8080/aggregate \
  -H "Content-Type: application/json"
```

### 3. Download Model Global
```bash
curl -O -J http://localhost:8080/download-global
```

### 4. Cek Accuracy Client
```bash
curl http://localhost:8080/accuracy/BANK_A
```

### 5. Lihat Semua Model
```bash
curl http://localhost:8080/logs
```

### 6. Hapus Model Client
```bash
curl -X DELETE http://localhost:8080/delete/BANK_A_weights.npz
```

---

## 📊 Test Cases

Server memiliki built-in test cases untuk 6 bank (BANK A sampai F). Setiap bank memiliki 4 test cases yang digunakan untuk mengevaluasi akurasi model global setelah agregasi.

Contoh test case untuk BANK A:
```python
[
    ({"amount": 200000, "merchant_category": "ecommerce", ...}, 0),  # Normal
    ({"amount": 7500000, "merchant_category": "travel", ...}, 1),    # Fraud
    ({"amount": 120000, "merchant_category": "subscription", ...}, 0), # Normal
    ({"amount": 9600000, "merchant_category": "investment", ...}, 1)   # Fraud
]
```

---
