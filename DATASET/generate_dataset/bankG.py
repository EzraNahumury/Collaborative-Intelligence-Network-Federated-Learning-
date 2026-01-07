import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


def generate_bank_g_dataset(n_rows=15000, fraud_ratio=0.05, seed=42):
    """
    Generator dataset Bank G (Legacy BUMN).

    Karakter:
    - Schema berbeda: transaction_value, merchant_type
    - Banyak data kotor (dirty strings, missing location)
    - Fraud pattern: e-commerce frequency spam + electronics international
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
    freq_fraud = rng.integers(4, 8, size=n_rows)      # freq spam
    transaction_frequency_24h = np.where(fraud_mask, freq_fraud, freq_legit)

    # ===========================================================
    # 3. AMOUNT / transaction_value
    # Legacy bank: transaksi kecil–menengah
    # ===========================================================
    legit_values = rng.integers(150_000, 1_500_000, size=n_rows)
    fraud_values = rng.integers(300_000, 900_000, size=n_rows)   # small fraud
    fraud_big_values = rng.integers(3_000_000, 12_000_000, size=n_rows)  # electronics fraud

    # assign base values
    transaction_value = np.where(fraud_mask, fraud_values, legit_values)

    # inject big-value fraud (electronics → legacy fraud pattern)
    big_fraud_mask = fraud_mask & (rng.random(n_rows) < 0.20)
    transaction_value[big_fraud_mask] = fraud_big_values[big_fraud_mask]

    # -----------------------------------------------------------
    #  DIRTY DATA INJECTION (VERY IMPORTANT FOR LEGACY SYSTEM)
    # -----------------------------------------------------------
    dirty_strings = [
        "_--_", "err_", "badval_", "broken_", "__"
    ]
    dirty_mask = rng.random(n_rows) < 0.10  # 10% data kotor

    transaction_value_dirty = transaction_value.astype(str)
    for i in range(n_rows):
        if dirty_mask[i]:
            prefix = random.choice(dirty_strings)
            transaction_value_dirty[i] = prefix + str(transaction_value[i])
        else:
            transaction_value_dirty[i] = str(transaction_value[i])

    # ===========================================================
    # 4. MERCHANT TYPE
    # ===========================================================
    merchant_type = rng.choice(
        ["retail", "groceries", "e-commerce", "electronics", "travel"],
        size=n_rows,
        p=[0.40, 0.25, 0.20, 0.10, 0.05]
    )

    # fraud override
    merchant_type[fraud_mask] = "e-commerce"

    # electronics fraud legacy pattern
    merchant_type[big_fraud_mask] = "electronics"

    # ===========================================================
    # 5. LOCATION (legacy = banyak NaN)
    # ===========================================================
    cities = ["Jakarta", "Bandung", "Bekasi", "Online"]
    location = rng.choice(cities, size=n_rows)

    # inject missing values
    missing_mask = rng.random(n_rows) < 0.15  # 15% missing
    location[missing_mask] = ""

    # fraud international (legacy)
    high_risk_locs = ["International", "Singapore", "Dubai"]
    location[big_fraud_mask] = rng.choice(high_risk_locs, size=big_fraud_mask.sum())

    # is_international
    is_international = np.where(location == "International", 1, 0)

    # ===========================================================
    # 6. TIMESTAMP (legacy slow system)
    # ===========================================================
    start_dt = datetime(2025, 9, 19, 12, 0, 0)
    timestamps = []

    for i in range(n_rows):
        # fraud spam faster, legit slower
        if fraud_mask[i]:
            start_dt += timedelta(minutes=int(rng.integers(1, 4)))
        else:
            start_dt += timedelta(minutes=int(rng.integers(3, 10)))
        timestamps.append(start_dt.strftime("%Y-%m-%d %H:%M"))

    # ===========================================================
    # 7. TRANSACTION ID
    # ===========================================================
    width = len(str(n_rows))
    transaction_id = [f"G{str(i).zfill(width)}" for i in range(1, n_rows + 1)]

    # ===========================================================
    # 8. BUILD RAW DATAFRAME (dirty!)
    # ===========================================================
    df = pd.DataFrame({
        "transaction_id": transaction_id,
        "timestamp": timestamps,
        "transaction_value": transaction_value_dirty,   # intentionally dirty
        "merchant_type": merchant_type,
        "location": location,
        "is_international": is_international,
        "transaction_frequency_24h": transaction_frequency_24h,
        "is_fraud": is_fraud,
    })

    return df


if __name__ == "__main__":
    df_g = generate_bank_g_dataset(
        n_rows=15000,
        fraud_ratio=0.05,   # legacy fraud low but present
        seed=42
    )

    print("Total rows     :", len(df_g))
    print("Fraud ratio    :", df_g['is_fraud'].mean())

    df_g.to_csv("../data/bank_G_data.csv", index=False)
    print("Saved → ../data/bank_G_data.csv")
