import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================================
# AUTO-PATH: Naik ke parent folder
# ============================================
BASE = Path(__file__).resolve().parent.parent

INPUT = BASE / "data" / "bank_M_data.csv"
OUTPUT_DIR = BASE / "data_cleaned"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT = OUTPUT_DIR / "bank_M_data_clean.csv"


def convert_unix_to_string(ts):
    """Convert unix timestamp → 'YYYY-MM-DD HH:mm'"""
    try:
        return datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
    except:
        return "1970-01-01 00:00"


def clean_bank_m():
    print(" Memproses Bank M ...")
    print(f" Membaca dari: {INPUT}")

    df = pd.read_csv(INPUT)

    # ===================================
    # 1. RENAME tx_id → transaction_id
    # ===================================
    df = df.rename(columns={
        "tx_id": "transaction_id"
    })

    # ===================================
    # 2. UNIX TIMESTAMP → NORMAL TIMESTAMP
    # ===================================
    df["timestamp"] = df["unix_timestamp"].apply(convert_unix_to_string)
    df = df.drop(columns=["unix_timestamp"])

    # ===================================
    # 3. merchant_country → merchant_category
    # ===================================
    df["merchant_category"] = df["merchant_country"].astype(str)
    df["location"] = df["merchant_country"].astype(str)

    # ===================================
    # 4. is_international
    # jeśli negara ≠ US (atau selain Indonesia), kita set 1
    # karena dataset global tidak pakai country "ID"
    # ===================================
    df["is_international"] = 1

    # ===================================
    # 5. TAMBAHKAN transaction_frequency_24h
    # ===================================
    df["transaction_frequency_24h"] = 0

    # ===================================
    # 6. CAST DATA TYPES
    # ===================================
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["avg_amount_7d"] = pd.to_numeric(df["avg_amount_7d"], errors="coerce").fillna(0)
    df["time_since_last_tx_sec"] = pd.to_numeric(df["time_since_last_tx_sec"], errors="coerce").fillna(0)
    df["is_fraud"] = df["is_fraud"].astype(int)

    # ===================================
    # 7. SUSUN ULANG KOLOM SESUAI SKEMA CIN
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
            "avg_amount_7d",
            "time_since_last_tx_sec",
            "is_fraud",
        ]
    ]

    # ===================================
    # 8. SAVE OUTPUT
    # ===================================
    df.to_csv(OUTPUT, index=False)

    print(" BANK M selesai dibersihkan!")
    print(f" Hasil disimpan di: {OUTPUT}")
    print(f" Kolom: {df.columns.tolist()}")
    print(f" Total rows: {df.shape[0]}")


if __name__ == "__main__":
    clean_bank_m()
