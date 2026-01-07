# ============================================================
# 🌍 BUILD GLOBAL FEATURE SET (KOMPATIBEL test.py ✅)
# ============================================================
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

banks = ["A","B","C","D","E","F"]
dataframes = []

print("📂 Membaca dataset...")
for b in banks:
    df = pd.read_csv(f"data/bank_{b}_data.csv")

    # Buang label kalau ada
    if "is_fraud" in df.columns:
        df = df.drop(columns=["is_fraud"])

    # Buang kolom yang tidak perlu
    for col in ["transaction_id", "timestamp", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    dataframes.append(df)

print(f"✅ Total bank dibaca: {len(dataframes)}")

# Gabungkan untuk menentukan kolom global
all_data = pd.concat(dataframes, axis=0, ignore_index=True)

# Deteksi kolom kategorikal & numerik
CAT_COLS = all_data.select_dtypes(include=["object"]).columns.tolist()
NUM_COLS = all_data.select_dtypes(include=["int","float"]).columns.tolist()

print("🔍 Kolom numerik:", NUM_COLS)
print("🎨 Kolom kategorikal:", CAT_COLS)

# One-hot encoding
encoded = pd.get_dummies(all_data, columns=CAT_COLS, drop_first=False).astype("float32")

# List fitur global setelah encoding
FEATURE_COLS = list(encoded.columns)

# Scaling numerik
scaler = StandardScaler()
scaler.fit(encoded)

# Simpan dalam bentuk DICTIONARY ✅
output = {
    "FEATURE_COLS": FEATURE_COLS,
    "NUM_COLS": NUM_COLS,
    "CAT_COLS": CAT_COLS,
    "SCALER": scaler
}

os.makedirs("models_global", exist_ok=True)
joblib.dump(output, "models_global/fitur_global_test.pkl")

print("\n🎉 BERHASIL!")
print("📦 fitur_global.pkl disimpan dalam format lengkap & kompatibel ✅")
print("📁 Lokasi: models_global/fitur_global.pkl")
