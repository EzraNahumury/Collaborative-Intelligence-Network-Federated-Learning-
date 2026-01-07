import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_bank_f_dataset(n_rows=15000, fraud_ratio=0.06, seed=42):
    """
    Generator dataset Bank F (Bank Syariah Modern).

    Karakter:
    - Nasabah retail & UMKM
    - Kategori sesuai prinsip Syariah (zakat, syariah_invest, umkm_payment)
    - Pola fraud mirip bank ritel modern: akun takeover -> e-commerce spam
    - Amount kecil–menengah
    - Fraud = e-commerce amount kecil + freq tinggi
    """

    rng = np.random.default_rng(seed)

    # ===========================================================
    # 1. FRAUD LABEL
    # ===========================================================
    is_fraud = (rng.random(n_rows) < fraud_ratio).astype(int)
    fraud_mask = is_fraud == 1

    # ===========================================================
    # 2. TRANSACTION FREQUENCY
    # Legit: 1–3
    # Fraud: 4–8
    # ===========================================================
    freq_legit = rng.integers(1, 4, size=n_rows)
    freq_fraud = rng.integers(4, 9, size=n_rows)
    transaction_frequency_24h = np.where(fraud_mask, freq_fraud, freq_legit)

    # ===========================================================
    # 3. AMOUNT DISTRIBUTION
    # Legit small–medium: 50k – 2.5 juta
    # Fraud e-commerce: 150k – 400k
    # ===========================================================
    amount_legit = rng.integers(50_000, 2_500_000, size=n_rows)
    amount_fraud = rng.integers(150_000, 400_000, size=n_rows)

    # combine
    amount = np.where(fraud_mask, amount_fraud, amount_legit)

    # log transform → stabil untuk FL
    amount = np.log1p(amount)

    # ===========================================================
    # 4. MERCHANT CATEGORY
    # ===========================================================
    merchant_category = rng.choice(
        ["retail", "syariah_invest", "zakat", "e-commerce", "groceries", "umkm_payment"],
        size=n_rows,
        p=[0.30, 0.15, 0.10, 0.20, 0.15, 0.10]
    )

    # fraud override: e-commerce only
    merchant_category[fraud_mask] = "e-commerce"

    # ===========================================================
    # 5. LOCATION
    # ===========================================================
    syariah_cities = ["Bekasi", "Depok", "Bogor", "Online"]

    location = rng.choice(syariah_cities, size=n_rows, p=[0.4, 0.2, 0.15, 0.25])

    # fraud → online only
    location[fraud_mask] = "Online"

    is_international = np.zeros(n_rows, dtype=int)

    # ===========================================================
    # 6. TIMESTAMP (mirip bank ritel)
    # ===========================================================
    start_dt = datetime(2025, 9, 19, 11, 30, 0)
    timestamps = []

    for i in range(n_rows):
        # fraud lebih rapat (1–3 menit)
        if fraud_mask[i]:
            start_dt += timedelta(minutes=int(rng.integers(1, 4)))
        else:
            start_dt += timedelta(minutes=int(rng.integers(3, 9)))
        timestamps.append(start_dt.strftime("%Y-%m-%d %H:%M"))

    # ===========================================================
    # 7. TRANSACTION ID
    # ===========================================================
    width = len(str(n_rows))
    transaction_id = [f"F{str(i).zfill(width)}" for i in range(1, n_rows + 1)]

    # ===========================================================
    # 8. BUILD FINAL DATAFRAME
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
    df_f = generate_bank_f_dataset(
        n_rows=15000,
        fraud_ratio=0.06,   # fraud ritel syariah moderat
        seed=42
    )

    print("Total rows     :", len(df_f))
    print("Fraud ratio    :", df_f['is_fraud'].mean())

    df_f.to_csv("../data/bank_F_data.csv", index=False)
    print("Saved → ../data/bank_F_data.csv")
