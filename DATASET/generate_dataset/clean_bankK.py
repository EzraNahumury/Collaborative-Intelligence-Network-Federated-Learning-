import pandas as pd
from pathlib import Path

# ============================================
#  AUTO-ADAPT PATH KE PARENT FOLDER
# ============================================
BASE = Path(__file__).resolve().parent.parent

INPUT = BASE / "data" / "bank_K_data.csv"
OUTPUT_DIR = BASE / "data_cleaned"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT = OUTPUT_DIR / "bank_K_data_clean.csv"


def clean_bank_k():
    print("Memproses Bank K ...")
    print(f"Membaca dari: {INPUT}")

    df = pd.read_csv(INPUT)

    # ===================================
    # 1. RENAME claim_id -> transaction_id
    # ===================================
    df = df.rename(columns={
        "claim_id": "transaction_id",
        "claim_amount": "amount",
        "is_fraudulent_claim": "is_fraud",
    })

    # ===================================
    # 2. TAMBAHKAN KOLOM YANG TIDAK ADA
    # ===================================
    df["merchant_category"] = "insurance_claim"
    df["location"] = "Unknown"
    df["is_international"] = 0
    df["transaction_frequency_24h"] = 0

    # ===================================
    # 3. CAST DATA
    # ===================================
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["customer_age"] = pd.to_numeric(df["customer_age"], errors="coerce").fillna(0)
    df["is_international"] = df["is_international"].astype(int)
    df["transaction_frequency_24h"] = df["transaction_frequency_24h"].astype(int)
    df["is_fraud"] = df["is_fraud"].astype(int)

    # ===================================
    # 4. SUSUN ULANG KOLOM SESUAI SKEMA CIN
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
            "customer_age",
            "diagnosis_code",
            "is_fraud",
        ]
    ]

    # ===================================
    # 5. SAVE OUTPUT
    # ===================================
    df.to_csv(OUTPUT, index=False)

    print("BANK K selesai dibersihkan!")
    print(f"Hasil disimpan di: {OUTPUT}")
    print(f"Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    clean_bank_k()
