import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_bank_b_dataset(n_rows=15000, fraud_ratio=0.04, seed=42):
    """
    Generator dataset Bank B (Korporat Besar).
    Karakter:
    - Amount sangat tinggi (puluhan juta – miliaran)
    - Banyak International
    - Dominan B2B & services
    - Fraud: 1 transaksi international bernilai ekstrem
    """
    rng = np.random.default_rng(seed)

    # ===========================================================
    # 1. FRAUD LABEL
    # ===========================================================
    is_fraud = (rng.random(n_rows) < fraud_ratio).astype(int)

    # ===========================================================
    # 2. TRANSACTION FREQUENCY (rendah)
    # ===========================================================
    freq_legit = rng.integers(1, 3, size=n_rows)  # 1–2
    freq_fraud = rng.integers(1, 3, size=n_rows)  # fraud juga 1–2
    transaction_frequency_24h = np.where(is_fraud == 1, freq_fraud, freq_legit)

    # ===========================================================
    # 3. AMOUNT DISTRIBUTION (extremely high)
    # ===========================================================
    # Legit large transactions: 5 juta – 500 juta
    amount_legit = rng.integers(5_000_000, 500_000_000, size=n_rows)

    # Fraud = INTERNASIONAL + jumlah ekstrem (300 juta – 5 miliar)
    amount_fraud = rng.integers(300_000_000, 5_000_000_000, size=n_rows)

    amount = np.where(is_fraud == 1, amount_fraud, amount_legit)

    # ===========================================================
    # 4. MERCHANT CATEGORY
    # ===========================================================
    merchant_category = rng.choice(
        ["B2B", "services", "retail", "travel"],
        size=n_rows,
        p=[0.55, 0.30, 0.10, 0.05]  # dominan B2B + services
    )

    # Fraud mostly B2B or services international
    fraud_mask = is_fraud == 1
    merchant_category[fraud_mask] = rng.choice(
        ["B2B", "services"],
        size=fraud_mask.sum(),
        p=[0.6, 0.4]
    )

    # ===========================================================
    # 5. LOCATION & INTERNATIONAL
    # ===========================================================
    location = np.full(n_rows, "Jakarta", dtype=object)

    # Legit international ~20%
    intl_legit_mask = (rng.random(n_rows) < 0.20) & (is_fraud == 0)
    location[intl_legit_mask] = "International"

    # Fraud = international 100%
    location[fraud_mask] = "International"

    is_international = np.where(location == "International", 1, 0)

    # ===========================================================
    # 6. TIMESTAMP (lebih natural)
    # ===========================================================
    start_dt = datetime(2025, 9, 19, 10, 0, 0)

    timestamps = []
    for i in range(n_rows):
        delta_minutes = rng.integers(1, 8)  # transaksi jarang → 1–8 menit
        start_dt += timedelta(minutes=int(delta_minutes))
        timestamps.append(start_dt.strftime("%Y-%m-%d %H:%M"))

    # ===========================================================
    # 7. TRANSACTION ID (B001 ...)
    # ===========================================================
    width = len(str(n_rows))
    transaction_id = [f"B{str(i).zfill(width)}" for i in range(1, n_rows + 1)]

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
    df_b = generate_bank_b_dataset(
        n_rows=15000,
        fraud_ratio=0.08,  # stabil & realistis untuk FL
        seed=42
    )

    print("Total rows     :", len(df_b))
    print("Fraud ratio    :", df_b['is_fraud'].mean())

    df_b.to_csv("../data/bank_B_data.csv", index=False)

    print("Saved → ../data/bank_B_data.csv")
