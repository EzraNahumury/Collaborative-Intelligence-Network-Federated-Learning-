import pandas as pd
from pathlib import Path

# ============================================
# AUTO-PATH: naik ke parent folder
# ============================================
BASE = Path(__file__).resolve().parent.parent

INPUT = BASE / "data" / "bank_N_data.csv"
OUTPUT_DIR = BASE / "data_cleaned"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT = OUTPUT_DIR / "bank_N_data_clean.csv"


def clean_bank_n():
    print(" Memproses Bank N ...")
    print(f" Membaca dari: {INPUT}")

    df = pd.read_csv(INPUT)

    # ===============================
    # 1. TAMBAHKAN MERCHANT CATEGORY
    # ===============================
    df["merchant_category"] = "encrypted_channel"

    # ===============================
    # 2. HANDLE LOCATION MISSING
    # ===============================
    df["location"] = df["location"].fillna("Unknown")
    df.loc[df["location"].astype(str).str.strip() == "", "location"] = "Unknown"

    # ===============================
    # 3. is_international
    # ===============================
    df["is_international"] = (df["location"] == "International").astype(int)

    # ===============================
    # 4. TAMBAHKAN transaction_frequency_24h
    # ===============================
    df["transaction_frequency_24h"] = 0

    # ===============================
    # 5. CAST NUMERIC
    # ===============================
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["is_fraud"] = df["is_fraud"].astype(int)

    # ===============================
    # 6. CLEAN hashed IDs (stringify)
    # ===============================
    df["hashed_merchant_id"] = df["hashed_merchant_id"].astype(str).str.strip()
    df["hashed_customer_id"] = df["hashed_customer_id"].astype(str).str.strip()

    # ===============================
    # 7. SUSUN ULANG KOLOM SESUAI CIN
    # ===============================
    df = df[
        [
            "transaction_id",
            "timestamp",
            "amount",
            "merchant_category",
            "location",
            "is_international",
            "transaction_frequency_24h",
            "hashed_merchant_id",
            "hashed_customer_id",
            "is_fraud",
        ]
    ]

    # ===============================
    # 8. SAVE OUTPUT
    # ===============================
    df.to_csv(OUTPUT, index=False)

    print(" BANK N selesai dibersihkan!")
    print(f" Hasil disimpan di: {OUTPUT}")
    print(f" Columns: {df.columns.tolist()}")
    print(f" Total rows: {df.shape[0]}")


if __name__ == "__main__":
    clean_bank_n()
