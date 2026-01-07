# ============================================================
# 🌍 Membuat Fitur Global untuk Federated Learning (Safe Version)
# ============================================================
import os, pandas as pd, joblib

banks = ["A","B","C","D","E","F"]
dataframes = []

print("📂 Membaca dataset dari semua bank...")
for b in banks:
    df = pd.read_csv(f"data/bank_{b}_data.csv")
    if "is_fraud" in df.columns:
        df = df.drop(columns=["is_fraud"])
    # 🧹 buang kolom ID & timestamp (tidak perlu di-encode)
    for col in ["transaction_id", "timestamp"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    dataframes.append(df)

print(f"✅ Total bank dibaca: {len(dataframes)}")

# ============================================================
# 🔄 Gabungkan semua kolom unik
# ============================================================
print("🔄 Menggabungkan semua fitur unik...")
all_data = pd.concat(dataframes, axis=0, ignore_index=True)
print("🧮 Total baris gabungan:", len(all_data))

# ============================================================
# 💡 One-hot encoding untuk kolom kategorikal
# ============================================================
cat_cols = all_data.select_dtypes(include=["object"]).columns.tolist()
print("🪄 Kolom kategorikal yang akan di-encode:", cat_cols)

encoded = pd.get_dummies(all_data, columns=cat_cols, drop_first=False).astype("float32")

# ============================================================
# 💾 Simpan daftar fitur global
# ============================================================
feature_cols = list(encoded.columns)
os.makedirs("models_global", exist_ok=True)
joblib.dump(feature_cols, "models_global/fitur_global.pkl")

print(f"\n✅ Total fitur global: {len(feature_cols)}")
print("💾 Disimpan di: models_global/fitur_global.pkl")
