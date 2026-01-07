import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_bank_e_dataset(n_rows=10000, fraud_ratio=0.03, seed=42):
    """
    Generator dataset Bank E (Wealth Management / Private Bank).

    Karakter:
    - Nasabah high-net-worth
    - Volume transaksi rendah, nilai per transaksi sangat tinggi
    - Kategori dominan: investment, asset_management, luxury_goods
    - Fraud: transfer internasional bernilai masif ke negara high-risk
             dengan frekuensi rendah (model tidak bisa hanya mengandalkan frequency)
    """
    rng = np.random.default_rng(seed)

    # ===========================================================
    # 1. FRAUD LABEL (jarang, tapi ada)
    # ===========================================================
    is_fraud = (rng.random(n_rows) < fraud_ratio).astype(int)
    fraud_mask = is_fraud == 1

    # ===========================================================
    # 2. TRANSACTION FREQUENCY (low for both)
    #    Legit: 1–3, Fraud: 1–2 (jadi freq tidak jadi sinyal utama)
    # ===========================================================
    freq_legit = rng.integers(1, 4, size=n_rows)
    freq_fraud = rng.integers(1, 3, size=n_rows)
    transaction_frequency_24h = np.where(fraud_mask, freq_fraud, freq_legit)

    # ===========================================================
    # 3. AMOUNT DISTRIBUTION (ekstrem tinggi)
    # ===========================================================
    # Legit high-value: 500 juta – 2.5 miliar
    amount_legit = rng.integers(500_000_000, 2_500_000_000, size=n_rows)

    # Fraud: 3 – 10 miliar
    amount_fraud = rng.integers(3_000_000_000, 10_000_000_000, size=n_rows)

    amount = np.where(fraud_mask, amount_fraud, amount_legit)

    # Log transform supaya skala selaras dengan bank lain (FL-friendly)
    amount = np.log1p(amount)

    # ===========================================================
    # 4. MERCHANT CATEGORY
    # ===========================================================
    merchant_category = rng.choice(
        ["investment", "asset_management", "luxury_goods"],
        size=n_rows,
        p=[0.5, 0.35, 0.15]
    )

    # Fraud sedikit lebih sering di investment / asset_management
    fraud_cats = ["investment", "asset_management"]
    merchant_category[fraud_mask] = rng.choice(fraud_cats, size=fraud_mask.sum(), p=[0.6, 0.4])

    # ===========================================================
    # 5. LOCATION & INTERNATIONAL
    # ===========================================================
    # Legit:
    legit_domestic_cities = ["Jakarta"]
    legit_offshore_cities = ["Singapore", "International"]

    location = np.full(n_rows, "Jakarta", dtype=object)

    # Sebagian kecil nasabah legit belanja / investasi luar negeri
    legit_offshore_mask = (rng.random(n_rows) < 0.20) & (~fraud_mask)
    location[legit_offshore_mask] = rng.choice(
        legit_offshore_cities,
        size=legit_offshore_mask.sum(),
        p=[0.7, 0.3]
    )

    # Fraud: negara high-risk / tax haven
    high_risk_locs = ["Switzerland", "Cayman Islands",
                      "British Virgin Islands", "Luxembourg", "Singapore"]
    location[fraud_mask] = rng.choice(high_risk_locs, size=fraud_mask.sum())

    is_international = np.where(location == "Jakarta", 0, 1)

    # ===========================================================
    # 6. TIMESTAMP (volume rendah, gap waktu besar)
    # ===========================================================
    start_dt = datetime(2025, 9, 19, 11, 30, 0)
    timestamps = []

    for i in range(n_rows):
        # transaksi jarang: loncatan 10–60 menit
        # fraud tidak punya pola frekuensi khusus di sini
        delta = int(rng.integers(10, 61))
        start_dt += timedelta(minutes=delta)
        timestamps.append(start_dt.strftime("%Y-%m-%d %H:%M"))

    # ===========================================================
    # 7. TRANSACTION ID (E001, E002, ...)
    # ===========================================================
    width = len(str(n_rows))
    transaction_id = [f"E{str(i).zfill(width)}" for i in range(1, n_rows + 1)]

    # ===========================================================
    # 8. BUILD FINAL DATAFRAME
    # ===========================================================
    df = pd.DataFrame({
        "transaction_id": transaction_id,
        "timestamp": timestamps,
        "amount": amount,
        "merchant_category": merchant_category,
        "location": location,
        "is_international": is_international.astype(int),
        "transaction_frequency_24h": transaction_frequency_24h,
        "is_fraud": is_fraud.astype(int),
    })

    return df


if __name__ == "__main__":
    df_e = generate_bank_e_dataset(
        n_rows=10000,
        fraud_ratio=0.03,
        seed=42
    )

    print("Total rows     :", len(df_e))
    print("Fraud ratio    :", df_e['is_fraud'].mean())

    df_e.to_csv("../data/bank_E_data.csv", index=False)
    print("Saved → ../data/bank_E_data.csv")
