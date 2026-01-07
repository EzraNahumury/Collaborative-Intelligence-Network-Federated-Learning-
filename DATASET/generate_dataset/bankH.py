import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_bank_h_dataset(n_rows=15000, fraud_ratio=0.08, seed=42):
    """
    Generator dataset Bank H (Challenger Digital Bank).

    Karakter:
    - Data sangat bersih (tidak ada NaN)
    - Struktur modern, memiliki device_risk_score (0.0 - 1.0)
    - Fraud pattern:
        • device_risk_score tinggi (>= 0.75)
        • transaction_frequency_24h tinggi (5–7)
        • mayoritas kategori e-commerce / subscription
        • amount bisa kecil atau besar (dua tipe fraud)
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
    freq_legit = rng.integers(1, 4, size=n_rows)      # 1–3
    freq_fraud = rng.integers(5, 8, size=n_rows)      # 5–7 (aggressive fraud)
    transaction_frequency_24h = np.where(fraud_mask, freq_fraud, freq_legit)

    # ===========================================================
    # 3. AMOUNT DISTRIBUTION
    # ===========================================================
    # Legit small–medium
    amount_legit = rng.integers(50_000, 600_000, size=n_rows)

    # Fraud type A: credential-takeover small fraud
    amount_fraud_small = rng.integers(80_000, 250_000, size=n_rows)

    # Fraud type B: huge international transfer (rare but critical)
    amount_fraud_big = rng.integers(5_000_000, 35_000_000, size=n_rows)
    big_fraud_mask = fraud_mask & (rng.random(n_rows) < 0.10)   # hanya 10% fraud besar

    # Combine amount
    amount = np.where(fraud_mask, amount_fraud_small, amount_legit)
    amount[big_fraud_mask] = amount_fraud_big[big_fraud_mask]

    # log transform supaya skala cocok dengan bank lain (FL-friendly)
    amount = np.log1p(amount)

    # ===========================================================
    # 4. MERCHANT CATEGORY
    # ===========================================================
    merchant_category = rng.choice(
        ["e-commerce", "subscription", "travel", "retail"],
        size=n_rows,
        p=[0.55, 0.20, 0.10, 0.15]
    )

    # Fraud override
    merchant_category[fraud_mask] = rng.choice(
        ["e-commerce", "subscription"],
        size=fraud_mask.sum(),
        p=[0.7, 0.3]
    )

    # Big fraud → mostly e-commerce International
    merchant_category[big_fraud_mask] = "e-commerce"

    # ===========================================================
    # 5. LOCATION
    # ===========================================================
    location = np.full(n_rows, "Online", dtype=object)

    # Legit offline occurrences (small portion)
    legit_offline_mask = (rng.random(n_rows) < 0.12) & (~fraud_mask)
    location[legit_offline_mask] = rng.choice(["Jakarta", "Bandung"], size=legit_offline_mask.sum())

    # Big fraud → International
    location[big_fraud_mask] = "International"

    is_international = np.where(location == "International", 1, 0)

    # ===========================================================
    # 6. DEVICE RISK SCORE (fitur utama Bank H)
    # ===========================================================
    # Legit → risk rendah (0.02 – 0.25)
    device_risk_legit = rng.uniform(0.02, 0.25, size=n_rows)

    # Fraud → risk tinggi (0.75 – 0.99)
    device_risk_fraud = rng.uniform(0.75, 0.99, size=n_rows)

    device_risk_score = np.where(fraud_mask, device_risk_fraud, device_risk_legit)

    # ===========================================================
    # 7. TIMESTAMP (fast digital bank)
    # ===========================================================
    start_dt = datetime(2025, 9, 19, 12, 1, 0)
    timestamps = []

    for i in range(n_rows):
        if fraud_mask[i]:
            start_dt += timedelta(minutes=int(rng.integers(1, 3)))  # fraud cepat
        else:
            start_dt += timedelta(minutes=int(rng.integers(2, 7)))  # legit sedikit lebih lambat
        timestamps.append(start_dt.strftime("%Y-%m-%d %H:%M"))

    # ===========================================================
    # 8. TRANSACTION ID
    # ===========================================================
    width = len(str(n_rows))
    transaction_id = [f"H{str(i).zfill(width)}" for i in range(1, n_rows + 1)]

    # ===========================================================
    # 9. BUILD FINAL DATAFRAME
    # ===========================================================
    df = pd.DataFrame({
        "transaction_id": transaction_id,
        "timestamp": timestamps,
        "amount": amount,
        "merchant_category": merchant_category,
        "location": location,
        "is_international": is_international,
        "transaction_frequency_24h": transaction_frequency_24h,
        "device_risk_score": device_risk_score,
        "is_fraud": is_fraud,
    })

    return df


if __name__ == "__main__":
    df_h = generate_bank_h_dataset(
        n_rows=15000,
        fraud_ratio=0.08,  # Challenger bank: fraud lebih tinggi dari bank Syariah
        seed=42
    )

    print("Total rows:", len(df_h))
    print("Fraud ratio:", df_h["is_fraud"].mean())

    df_h.to_csv("../data/bank_H_data.csv", index=False)
    print("Saved → ../data/bank_H_data.csv")
