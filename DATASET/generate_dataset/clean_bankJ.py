import pandas as pd
from pathlib import Path

# ============================================
#  PATH FIX — otomatis naik ke parent folder
# ============================================
BASE = Path(__file__).resolve().parent.parent

INPUT = BASE / "data" / "bank_J_data.csv"
OUTPUT_DIR = BASE / "data_cleaned"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT = OUTPUT_DIR / "bank_J_data_clean.csv"


def clean_bank_j():
    print(" Memproses Bank J ...")
    print(f" Membaca dari: {INPUT}")

    df = pd.read_csv(INPUT)

    # ===================================
    # 1. TAMBAHKAN KOLOM YG TIDAK ADA
    # ===================================
    df["merchant_category"] = "asset_trading"
    df["location"] = "Online"
    df["is_international"] = 0
    df["transaction_frequency_24h"] = 0

    # ===================================
    # 2. CAST NUMERIC
    # ===================================
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["is_international"] = df["is_international"].astype(int)
    df["transaction_frequency_24h"] = df["transaction_frequency_24h"].astype(int)
    df["is_fraud"] = df["is_fraud"].astype(int)

    # ===================================
    # 3. SUSUN ULANG KOLOM SESUAI CIN
    # ===================================
    df = df[
        [
            "transaction_id",
            "timestamp",
            "amount",
            "merchant_category",
            "location",
            "is_international",
            "transaction_frequency_24h",
            "asset_ticker",
            "transaction_type",
            "is_fraud",
        ]
    ]

    # ===================================
    # 4. SAVE OUTPUT
    # ===================================
    df.to_csv(OUTPUT, index=False)

    print(" BANK J selesai dibersihkan!")
    print(f" Hasil disimpan di: {OUTPUT}")
    print(f" Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    clean_bank_j()
