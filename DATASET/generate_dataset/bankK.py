import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_bank_k_dataset(n_rows=15000, fraud_ratio=0.05, seed=42):
    """
    Generator dataset Bank K (Asuransi Jiwa & Kesehatan).

    Karakter:
    - Data berupa klaim, bukan transaksi finansial
    - Fitur: claim_amount, customer_age, diagnosis_code
    - Fraud pattern:
        • Klaim besar (ratusan juta)
        • Diagnosis berat atau mencurigakan (C50.9, I21.9)
        • Age-diagnosis mismatch (usia tidak cocok dengan diagnosis)
    """

    rng = np.random.default_rng(seed)

    # ===========================================================
    # 1. FRAUD LABEL
    # ===========================================================
    is_fraud = (rng.random(n_rows) < fraud_ratio).astype(int)
    fraud_mask = is_fraud == 1

    # ===========================================================
    # 2. CUSTOMER AGE
    # ===========================================================
    # Legit: umur 18–85 (normal distribution)
    customer_age = rng.integers(18, 85, size=n_rows)

    # Fraud: kadang age mismatch (contoh: cancer di usia 20)
    age_mismatch_mask = fraud_mask & (rng.random(n_rows) < 0.25)
    customer_age[age_mismatch_mask] = rng.integers(20, 40, size=age_mismatch_mask.sum())

    # ===========================================================
    # 3. DIAGNOSIS CODE (ICD-10)
    # ===========================================================
    diagnosis_legit = np.array([
        "J45.9", "J06.9", "K29.7", "E11.9", "I10", "Z34.9", "E78.5"
    ])

    diagnosis_fraud_highrisk = np.array([
        "C50.9",  # Breast cancer
        "I21.9",  # Heart attack
    ])

    diagnosis_code = rng.choice(diagnosis_legit, size=n_rows)

    # Fraud override
    diagnosis_code[fraud_mask] = rng.choice(
        diagnosis_fraud_highrisk,
        size=fraud_mask.sum(),
        p=[0.55, 0.45]
    )

    # Age-diagnosis mismatch injection
    mismatch_idx = np.where(age_mismatch_mask)[0]
    if mismatch_idx.size > 0:
        diagnosis_code[mismatch_idx] = rng.choice(
            ["C50.9", "I21.9"], size=mismatch_idx.size
        )

    # ===========================================================
    # 4. CLAIM AMOUNT
    # ===========================================================
    # Normal claims: 5 juta – 60 juta
    claim_legit = rng.integers(5_000_000, 60_000_000, size=n_rows)

    # Fraud claims: 200 juta – 800 juta
    claim_fraud_big = rng.integers(200_000_000, 800_000_000, size=n_rows)

    claim_amount = np.where(fraud_mask, claim_fraud_big, claim_legit)

    # Log-transform agar stabil di FL
    claim_amount = np.log1p(claim_amount)

    # ===========================================================
    # 5. TIMESTAMP
    # ===========================================================
    start_dt = datetime(2025, 9, 19, 12, 30)
    timestamps = []

    for i in range(n_rows):
        # Klaim tidak secepat transaksi → interval 3–12 menit
        delta = int(rng.integers(3, 13))
        start_dt += timedelta(minutes=delta)
        timestamps.append(start_dt.strftime("%Y-%m-%d %H:%M"))

    # ===========================================================
    # 6. CLAIM ID
    # ===========================================================
    width = len(str(n_rows))
    claim_id = [f"K{str(i).zfill(width)}" for i in range(1, n_rows + 1)]

    # ===========================================================
    # 7. BUILD DATAFRAME
    # ===========================================================
    df = pd.DataFrame({
        "claim_id": claim_id,
        "timestamp": timestamps,
        "claim_amount": claim_amount,
        "customer_age": customer_age,
        "diagnosis_code": diagnosis_code,
        "is_fraudulent_claim": is_fraud,
    })

    return df


if __name__ == "__main__":
    df_k = generate_bank_k_dataset(
        n_rows=15000,
        fraud_ratio=0.05,
        seed=42
    )

    print("Total rows:", len(df_k))
    print("Fraud ratio:", df_k["is_fraudulent_claim"].mean())

    df_k.to_csv("../data/bank_K_data.csv", index=False)
    print("Saved → ../data/bank_K_data.csv")
