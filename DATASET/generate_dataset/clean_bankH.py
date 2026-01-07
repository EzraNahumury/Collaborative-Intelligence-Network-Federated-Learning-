import pandas as pd
from pathlib import Path

# ============================================
#  PATH 
# ============================================
BASE = Path(__file__).resolve().parent.parent

INPUT = BASE / "data" / "bank_H_data.csv"
OUTPUT_DIR = BASE / "data_cleaned"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT = OUTPUT_DIR / "bank_H_data_clean.csv"


def clean_bank_h():
    print(" Memproses Bank H ...")
    print(f" Membaca dari: {INPUT}")

    df = pd.read_csv(INPUT)

    # =============================
    # 1. HAPUS FITUR EKSKLUSIF
    # =============================
    if "device_risk_score" in df.columns:
        df = df.drop(columns=["device_risk_score"])
        print("  Kolom 'device_risk_score' dihapus.")

    # =============================
    # 2. CAST DATA TYPE YANG PERLU
    # =============================
    df["is_international"] = df["is_international"].astype(int)
    df["transaction_frequency_24h"] = df["transaction_frequency_24h"].astype(int)
    df["is_fraud"] = df["is_fraud"].astype(int)

    # =============================
    # 3. SAVE OUTPUT
    # =============================
    df.to_csv(OUTPUT, index=False)

    print(" BANK H selesai dibersihkan!")
    print(f" Hasil disimpan di: {OUTPUT}")
    print(f" Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    clean_bank_h()
