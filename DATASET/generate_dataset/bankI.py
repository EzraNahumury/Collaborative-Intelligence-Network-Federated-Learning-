import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_bank_i_dataset(n_rows=12000, fraud_ratio=0.03, seed=42):
    """
    Generator dataset Bank I (Bank Pembangunan Daerah).

    Karakter:
    - Sistem sederhana, tidak punya transaction_frequency_24h
    - Hanya fitur dasar: amount, merchant_category, location, is_international
    - Pola fraud "gaya lama": 1 transaksi domestik bernilai ekstrem
    - Tidak ada freq tinggi, tidak ada device score, tidak ada fitur modern
    - Fraud ratio kecil (2–3%)
    """

    rng = np.random.default_rng(seed)

    # ===========================================================
    # 1. FRAUD LABEL (old-style fraud)
    # ===========================================================
    is_fraud = (rng.random(n_rows) < fraud_ratio).astype(int)
    fraud_mask = is_fraud == 1

    # ===========================================================
    # 2. LEGIT AMOUNT (small–medium)
    # ===========================================================
    amount_legit = rng.integers(150_000, 1_800_000, size=n_rows)

    # ===========================================================
    # 3. FRAUD AMOUNT (extremely large domestic transfers)
    # ===========================================================
    amount_fraud = rng.integers(50_000_000, 120_000_000, size=n_rows)

    amount = np.where(fraud_mask, amount_fraud, amount_legit)

    # log transform agar scale match dengan bank lain di FL
    amount = np.log1p(amount)

    # ===========================================================
    # 4. MERCHANT CATEGORY
    # ===========================================================
    merchant_category = rng.choice(
        ["retail", "groceries", "travel", "electronics"],
        size=n_rows,
        p=[0.50, 0.30, 0.10, 0.10]
    )

    # fraud = electronics purchases mostly (old-style fraud)
    merchant_category[fraud_mask] = "electronics"

    # ===========================================================
    # 5. LOCATION (mostly Medan)
    # ===========================================================
    location = rng.choice(
        ["Medan", "Jakarta"],
        size=n_rows,
        p=[0.85, 0.15]
    )

    # fraud = always domestic (Medan)
    location[fraud_mask] = "Medan"

    # no internationals in BPD
    is_international = np.zeros(n_rows, dtype=int)

    # ===========================================================
    # 6. TIMESTAMP (low-traffic small bank)
    # ===========================================================
    start_dt = datetime(2025, 9, 19, 12, 2, 0)
    timestamps = []

    for i in range(n_rows):
        # transaksi jarang: 6–18 menit antar transaksi
        start_dt += timedelta(minutes=int(rng.integers(6, 19)))
        timestamps.append(start_dt.strftime("%Y-%m-%d %H:%M"))

    # ===========================================================
    # 7. TRANSACTION ID
    # ===========================================================
    width = len(str(n_rows))
    transaction_id = [f"I{str(i).zfill(width)}" for i in range(1, n_rows + 1)]

    # ===========================================================
    # 8. BUILD FINAL DATAFRAME
    #    (no transaction_frequency_24h since system is legacy)
    # ===========================================================
    df = pd.DataFrame({
        "transaction_id": transaction_id,
        "timestamp": timestamps,
        "amount": amount,
        "merchant_category": merchant_category,
        "location": location,
        "is_international": is_international,
        "is_fraud": is_fraud,
    })

    return df


if __name__ == "__main__":
    df_i = generate_bank_i_dataset(
        n_rows=12000,
        fraud_ratio=0.03,
        seed=42
    )

    print("Total rows:", len(df_i))
    print("Fraud ratio:", df_i["is_fraud"].mean())

    df_i.to_csv("../data/bank_I_data.csv", index=False)
    print("Saved → ../data/bank_I_data.csv")
