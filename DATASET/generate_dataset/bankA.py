import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_bank_a_dataset(n_rows=20000, fraud_ratio=0.10, seed=42):
    """
    Generator dataset Bank A (versi stabil untuk Federated Learning).
    Pattern disesuaikan agar model global bisa belajar:
    - transaksi nominal kecil–menengah
    - mayoritas Online + e-commerce
    - fraud berupa transaksi kecil & cepat
    - anomaly: nilai besar + international
    """
    rng = np.random.default_rng(seed)

    # ===========================================================
    # 1. IS_FRAUD LABEL
    # ===========================================================
    is_fraud = (rng.random(n_rows) < fraud_ratio).astype(int)

    # ===========================================================
    # 2. TRANSACTION FREQUENCY
    #    Legit: 1–4
    #    Fraud (small-rapid fraud): 4–10
    # ===========================================================
    freq_legit = rng.integers(1, 5, size=n_rows)
    freq_fraud = rng.integers(4, 10, size=n_rows)
    transaction_frequency_24h = np.where(is_fraud == 1, freq_fraud, freq_legit)

    # ===========================================================
    # 3. AMOUNT DISTRIBUTION (lognormal = realistic)
    # ===========================================================
    # legit: mostly low amount
    amount_legit = np.round(
        np.exp(rng.normal(11, 0.35, size=n_rows))  # mean nominal ~ 60.000–250.000
    )

    # small-rapid fraud: similar distribution but slightly lower range
    amount_fraud = np.round(
        np.exp(rng.normal(10.8, 0.30, size=n_rows))
    )

    amount = np.where(is_fraud == 1, amount_fraud, amount_legit)

    # anomaly high-value (2% sample)
    high_mask = rng.random(n_rows) < 0.02
    amount[high_mask] = rng.integers(3_000_000, 20_000_000, high_mask.sum())

    # ===========================================================
    # 4. MERCHANT CATEGORY
    # ===========================================================
    merchant_category = rng.choice(
        ["e-commerce", "subscription", "travel", "retail"],
        size=n_rows,
        p=[0.72, 0.10, 0.08, 0.10]
    )

    # anomaly → mostly travel
    merchant_category[high_mask] = rng.choice(
        ["travel", "e-commerce"], size=high_mask.sum(), p=[0.8, 0.2]
    )

    # ===========================================================
    # 5. LOCATION & INTERNATIONAL
    # ===========================================================
    location = np.full(n_rows, "Online", dtype=object)

    # 15% offline (Jakarta)
    jkt_mask = rng.random(n_rows) < 0.15
    location[jkt_mask] = "Jakarta"

    # international occurs mostly on anomaly
    intl_mask = high_mask & (rng.random(n_rows) < 0.7)
    location[intl_mask] = "International"

    is_international = np.where(location == "International", 1, 0)

    # ===========================================================
    # 6. TIMESTAMP (lebih natural)
    # ===========================================================
    start_dt = datetime(2025, 9, 19, 10, 0, 0)

    # timestamp naik sesuai index (lebih smooth)
    timestamps = []
    for i in range(n_rows):
        delta_minutes = rng.integers(0, 5)  # tiap transaksi maju 0–5 menit
        start_dt += timedelta(minutes=int(delta_minutes))
        timestamps.append(start_dt.strftime("%Y-%m-%d %H:%M"))

    # ===========================================================
    # 7. TRANSACTION ID
    # ===========================================================
    width = len(str(n_rows))
    transaction_id = [f"A{str(i).zfill(width)}" for i in range(1, n_rows + 1)]

    # ===========================================================
    # 8. BUILD DATAFRAME
    # ===========================================================
    df = pd.DataFrame({
        "transaction_id": transaction_id,
        "timestamp": timestamps,
        "amount": amount.astype(int),
        "merchant_category": merchant_category,
        "location": location,
        "is_international": is_international.astype(int),
        "transaction_frequency_24h": transaction_frequency_24h,
        "is_fraud": is_fraud.astype(int),
    })

    return df


if __name__ == "__main__":
    df_a = generate_bank_a_dataset(
        n_rows=20000,     # ukuran besar bagus untuk FL
        fraud_ratio=0.10, # rasio ideal FL
        seed=42
    )

    print("Total rows     :", len(df_a))
    print("Fraud ratio    :", df_a["is_fraud"].mean())

    # Simpan ke folder data/
    df_a.to_csv("../data/bank_A_data.csv", index=False)

    print("Saved → ../data/bank_A_data.csv")
