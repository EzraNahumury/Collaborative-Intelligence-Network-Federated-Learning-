import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent   # naik 1 folder

INPUT = BASE / "data" / "bank_G_data.csv"
OUTPUT_DIR = BASE / "data_cleaned"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT = OUTPUT_DIR / "bank_G_data_clean.csv"



def clean_numeric(val):
    """Membersihkan angka yang rusak seperti '_--_500000' menjadi 500000"""
    s = str(val)
    s = s.replace(",", "")
    s = s.replace("Rp", "")
    s = s.replace(" ", "")
    s = s.replace("_", "")
    # hapus semua karakter non-digit
    s = ''.join(ch for ch in s if ch.isdigit())
    if s == "":
        return 0
    return float(s)


def clean_bank_g():
    print(" Memproses Bank G ...")

    df = pd.read_csv(INPUT)

    # ============ 1. RENAME COLUMNS ===============
    df = df.rename(columns={
        "transaction_value": "amount",
        "merchant_type": "merchant_category"
    })

    # ============ 2. CLEAN AMOUNT =================
    df["amount"] = df["amount"].apply(clean_numeric)

    # ============ 3. HANDLE MISSING LOCATION ======
    df["location"] = df["location"].fillna("Unknown")
    df.loc[df["location"].astype(str).str.strip() == "", "location"] = "Unknown"

    # ============ 4. CAST NUMERIC COLUMNS =========
    df["is_international"] = df["is_international"].astype(int)
    df["transaction_frequency_24h"] = df["transaction_frequency_24h"].astype(int)
    df["is_fraud"] = df["is_fraud"].astype(int)

    # ============ 5. SAVE OUTPUT ==================
    df.to_csv(OUTPUT, index=False)

    print(" BANK G selesai dibersihkan!")
    print(f" Saved to: {OUTPUT}")
    print(f" Shape: {df.shape}")
    print(f" Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    clean_bank_g()
