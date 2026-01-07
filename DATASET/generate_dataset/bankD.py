import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_bank_d_dataset(n_rows=15000, fraud_ratio=0.10, seed=42):
    """
    Bank D – Advanced Fintech Fraud Dataset (v2)
    - Pola fraud lebih kompleks dan realistis
    - Time-window fraud burst
    - Multi-loan scam pattern
    - Payment clustering (Gamma distribution)
    """
    rng = np.random.default_rng(seed)

    # ===========================================================
    # 1. FRAUD LABEL
    # ===========================================================
    is_fraud = (rng.random(n_rows) < fraud_ratio).astype(int)
    fraud_mask = is_fraud == 1

    # ===========================================================
    # 2. TRANSACTION FREQUENCY
    # ===========================================================
    freq_legit = rng.integers(1, 4, size=n_rows)               # 1–3
    freq_fraud = rng.integers(6, 15, size=n_rows)              # FREQUENCY EXTREME
    transaction_frequency_24h = np.where(fraud_mask, freq_fraud, freq_legit)

    # ===========================================================
    # 3. AMOUNT DISTRIBUTION
    # ===========================================================
    # Payment cluster (Gamma distribution → lebih smooth)
    amount_payment = np.round(rng.gamma(shape=2.0, scale=300000, size=n_rows))
    amount_payment = np.clip(amount_payment, 150000, 3000000)

    # Legit Loan moderate
    amount_loan_legit = rng.integers(5_000_000, 18_000_000, size=n_rows)

    # Fraud loan (noise besar)
    amount_loan_fraud = rng.integers(12_000_000, 35_000_000, size=n_rows)

    # Low-ball fraud → untuk pola "test fraud"
    low_fraud_mask = fraud_mask & (rng.random(n_rows) < 0.18)
    amount_loan_fraud[low_fraud_mask] = rng.integers(3_000_000, 7_000_000, size=low_fraud_mask.sum())

    # kategori dasar
    merchant_category = rng.choice(
        ["loan_disbursement", "fintech_payment"],
        size=n_rows,
        p=[0.50, 0.50]
    )

    # randomize fraud categories supaya tidak terlalu obvious
    fraud_categories = ["loan_disbursement", "instant_loan", "fast_loan", "fake_payout"]
    merchant_category[fraud_mask] = rng.choice(fraud_categories, size=fraud_mask.sum())

    # assign amount
    amount = np.where(
        merchant_category == "loan_disbursement",
        amount_loan_legit,
        amount_payment
    )

    # fraud override
    amount = np.where(fraud_mask, amount_loan_fraud, amount)

    # log transform untuk stabil FL
    amount = np.log1p(amount)

    # ===========================================================
    # 4. LOCATION (always online)
    # ===========================================================
    location = np.full(n_rows, "Online", dtype=object)
    is_international = np.zeros(n_rows, dtype=int)

    # ===========================================================
    # 5. TIMESTAMP – fraud burst window
    # ===========================================================
    start_dt = datetime(2025, 9, 19, 11, 30, 0)
    timestamps = []

    for i in range(n_rows):
        if fraud_mask[i]:
            # fraud burst waktu → spam dalam 1–2 menit
            start_dt += timedelta(minutes=int(rng.integers(1, 3)))
        else:
            # legit lebih jarang: 3–8 menit
            start_dt += timedelta(minutes=int(rng.integers(3, 8)))
        timestamps.append(start_dt.strftime("%Y-%m-%d %H:%M"))

    # ===========================================================
    # 6. TRANSACTION ID
    # ===========================================================
    width = len(str(n_rows))
    transaction_id = [f"D{str(i).zfill(width)}" for i in range(1, n_rows + 1)]

    # ===========================================================
    # 7. BUILD FINAL DF
    # ===========================================================
    df = pd.DataFrame({
        "transaction_id": transaction_id,
        "timestamp": timestamps,
        "amount": amount,
        "merchant_category": merchant_category,
        "location": location,
        "is_international": is_international,
        "transaction_frequency_24h": transaction_frequency_24h,
        "is_fraud": is_fraud,
    })

    return df


if __name__ == "__main__":
    df_d = generate_bank_d_dataset(
        n_rows=15000,
        fraud_ratio=0.15,
        seed=42
    )

    print("Total rows     :", len(df_d))
    print("Fraud ratio    :", df_d['is_fraud'].mean())
    df_d.to_csv("../data/bank_D_data.csv", index=False)
    print("Saved → ../data/bank_D_data.csv")
