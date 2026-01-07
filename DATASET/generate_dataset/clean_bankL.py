import pandas as pd
from pathlib import Path

# ============================================
#  AUTO-PATH: Naik ke parent folder
# ============================================
BASE = Path(__file__).resolve().parent.parent

INPUT = BASE / "data" / "bank_L_data.csv"
OUTPUT_DIR = BASE / "data_cleaned"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT = OUTPUT_DIR / "bank_L_data_clean.csv"


def clean_bank_l(oversample=False):
    print(" Memproses Bank L ...")
    print(f" Membaca dari: {INPUT}")

    df = pd.read_csv(INPUT)

    # ===================================
    # 1. RENAME loan_id -> transaction_id
    # ===================================
    df = df.rename(columns={
        "loan_id": "transaction_id",
        "loan_amount": "amount",
        "location_branch": "location",
        "is_default": "is_fraud"
    })

    # ===================================
    # 2. TAMBAHKAN KOLOM YANG TIDAK ADA
    # ===================================
    df["merchant_category"] = "microfinance_loan"
    df["is_international"] = 0
    df["transaction_frequency_24h"] = 0

    # ===================================
    # 3. CAST DATA TYPES
    # ===================================
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["loan_tenor_months"] = pd.to_numeric(df["loan_tenor_months"], errors="coerce").fillna(0)
    df["is_fraud"] = df["is_fraud"].astype(int)

    # ===================================
    # 4. OPSIONAL: OVERSAMPLING FRAUD
    # ===================================
    if oversample:
        fraud_df = df[df["is_fraud"] == 1]
        if not fraud_df.empty:
            # Duplikasi fraud secara ringan
            df = pd.concat([df, fraud_df.sample(len(df)//5, replace=True)], ignore_index=True)
            print(f" Oversampling fraud dilakukan. Total rows sekarang: {len(df)}")
        else:
            print(" Tidak ada fraud untuk di-oversample.")

    # ===================================
    # 5. SUSUN ULANG KOLOM SESUAI SKEMA CIN
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
            "loan_tenor_months",
            "is_fraud",
        ]
    ]

    # ===================================
    # 6. SAVE OUTPUT
    # ===================================
    df.to_csv(OUTPUT, index=False)

    print("BANK L selesai dibersihkan!")
    print(f"Hasil disimpan di: {OUTPUT}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Total rows: {df.shape[0]}")


if __name__ == "__main__":
    clean_bank_l(oversample=False)
