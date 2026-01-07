import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_bank_c_dataset(n_rows=15000, fraud_ratio=0.07, seed=42):
    """
    Generator dataset Bank C (Ritel Konvensional).
    Karakter:
    - Amount merata (150 ribu – 3 juta)
    - Transaksi fisik di Jakarta/Bandung
    - Kategori groceries & retail dominan
    - Fraud: transaksi fisik di luar kota domisili
    """
    rng = np.random.default_rng(seed)

    # ===========================================================
    # 1. FRAUD LABEL
    # ===========================================================
    is_fraud = (rng.random(n_rows) < fraud_ratio).astype(int)

    # ===========================================================
    # 2. TRANSACTION FREQUENCY
    # ===========================================================
    freq_legit = rng.integers(1, 4, size=n_rows)  # 1-3
    freq_fraud = rng.integers(1, 3, size=n_rows)  # fraud cenderung single-shot
    transaction_frequency_24h = np.where(is_fraud == 1, freq_fraud, freq_legit)

    # ===========================================================
    # 3. AMOUNT DISTRIBUTION (merata)
    # ===========================================================
    # Legit: 150 ribu – 1.5 juta
    amount_legit = rng.integers(150_000, 1_500_000, size=n_rows)

    # Fraud: 1 juta – 3 juta (lebih tinggi & mencurigakan)
    amount_fraud = rng.integers(1_000_000, 3_000_000, size=n_rows)

    # Gabungkan
    amount = np.where(is_fraud == 1, amount_fraud, amount_legit)

    # Log transform (supaya cocok dengan FL preprocessing global)
    amount = np.log1p(amount)

    # ===========================================================
    # 4. MERCHANT CATEGORY
    # ===========================================================
    merchant_category = rng.choice(
        ["groceries", "retail", "electronics", "travel"],
        size=n_rows,
        p=[0.50, 0.35, 0.10, 0.05]  # dominan groceries & retail
    )

    # Fraud: electronics/travel lebih sering muncul
    fraud_mask = is_fraud == 1
    merchant_category[fraud_mask] = rng.choice(
        ["electronics", "travel", "retail"],
        size=fraud_mask.sum(),
        p=[0.5, 0.3, 0.2]
    )

    # ===========================================================
    # 5. LOCATION & INTERNATIONAL
    # ===========================================================
    legit_cities = ["Jakarta", "Bandung"]
    fraud_cities = ["Surabaya", "Medan", "Bali", "Semarang", "Makassar"]

    location = rng.choice(legit_cities, size=n_rows, p=[0.7, 0.3])

    # Fraud selalu dilakukan di kota luar domisili
    location[fraud_mask] = rng.choice(fraud_cities, size=fraud_mask.sum())

    is_international = np.where(location == "International", 1, 0)

    # ===========================================================
    # 6. TIMESTAMP (natural increasing)
    # ===========================================================
    start_dt = datetime(2025, 9, 19, 10, 0, 0)
    timestamps = []

    for _ in range(n_rows):
        start_dt += timedelta(minutes=int(rng.integers(1, 6)))  # 1–6 menit
        timestamps.append(start_dt.strftime("%Y-%m-%d %H:%M"))

    # ===========================================================
    # 7. TRANSACTION ID
    # ===========================================================
    width = len(str(n_rows))
    transaction_id = [f"C{str(i).zfill(width)}" for i in range(1, n_rows + 1)]

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
    df_c = generate_bank_c_dataset(
        n_rows=15000,
        fraud_ratio=0.07,  # ideal untuk FL
        seed=42
    )

    print("Total rows     :", len(df_c))
    print("Fraud ratio    :", df_c["is_fraud"].mean())

    df_c.to_csv("../data/bank_C_data.csv", index=False)

    print("Saved → ../data/bank_C_data.csv")
