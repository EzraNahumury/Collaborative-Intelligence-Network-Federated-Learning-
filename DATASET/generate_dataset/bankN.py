import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def _generate_hash_list(rng, n_hash: int):
    """
    Membuat daftar hash pseudo-hex seperti 0x2a7d...
    """
    hashes = []
    for _ in range(n_hash):
        # 4 digit hex lalu "..."
        val = rng.integers(0, 16**4)
        hashes.append(f"0x{val:04x}...")
    return hashes


def generate_bank_n_dataset(
    n_rows: int = 20000,
    fraud_ratio: float = 0.12,
    seed: int = 42,
):
    """
    Generate dataset Bank N (Data Terenkripsi / Hashed).

    Kolom:
    - transaction_id
    - timestamp               (string, akan di-drop di training client)
    - amount                  (log1p dari nominal IDR)
    - hashed_merchant_id      (string seperti 0x2a7d...)
    - hashed_customer_id      (string seperti 0x9f4b...)
    - location                (Jakarta, Bandung, Surabaya, International)
    - is_fraud                (label - KOLOM TERAKHIR)

    Fraud pattern:
    - amount besar (3–12 juta)
    - lokasi International
    - hashed_merchant_id & hashed_customer_id tertentu muncul sering saat fraud
      → "fraud hotspots"
    """

    rng = np.random.default_rng(seed)

    # ===========================================================
    # 1. Label fraud
    # ===========================================================
    is_fraud = (rng.random(n_rows) < fraud_ratio).astype(int)
    fraud_mask = is_fraud == 1
    fraud_indices = np.where(fraud_mask)[0]

    # ===========================================================
    # 2. Timestamp (tiap 1–3 menit)
    # ===========================================================
    start_dt = datetime(2025, 9, 19, 13, 0, 0)
    timestamps = []
    for i in range(n_rows):
        if i == 0:
            dt = start_dt
        else:
            dt = timestamps[-1] + timedelta(minutes=int(rng.integers(1, 4)))
        timestamps.append(dt)

    timestamp_str = [dt.strftime("%Y-%m-%d %H:%M") for dt in timestamps]

    # ===========================================================
    # 3. Amount
    # ===========================================================
    # Normal transaksi: 100k – 900k
    amount_legit = rng.integers(100_000, 900_000, size=n_rows)

    # Fraud: transaksi besar 3 juta – 12 juta
    amount_fraud = rng.integers(3_000_000, 12_000_000, size=n_rows)

    amount = np.where(fraud_mask, amount_fraud, amount_legit)

    # Log transform agar skala sejajar dengan bank lain
    amount_log = np.log1p(amount)

    # ===========================================================
    # 4. Hashed merchant & customer IDs
    # ===========================================================
    n_merchants = 30
    n_customers = 80

    merchant_hashes = _generate_hash_list(rng, n_merchants)
    customer_hashes = _generate_hash_list(rng, n_customers)

    # pilih merchant & customer acak untuk semua baris
    hashed_merchant_id = rng.choice(merchant_hashes, size=n_rows)
    hashed_customer_id = rng.choice(customer_hashes, size=n_rows)

    # Fraud hotspots: merchant & customer tertentu sering muncul saat fraud
    fraud_merchant_pool = rng.choice(merchant_hashes, size=3, replace=False)
    fraud_customer_pool = rng.choice(customer_hashes, size=4, replace=False)

    if fraud_indices.size > 0:
        hashed_merchant_id[fraud_indices] = rng.choice(
            fraud_merchant_pool,
            size=fraud_indices.size
        )
        hashed_customer_id[fraud_indices] = rng.choice(
            fraud_customer_pool,
            size=fraud_indices.size
        )

    # ===========================================================
    # 5. Location
    # ===========================================================
    locations = np.array(["Jakarta", "Bandung", "Surabaya", "International"])
    probs_legit = np.array([0.55, 0.20, 0.15, 0.10])

    location = rng.choice(locations, size=n_rows, p=probs_legit)

    # Fraud override: kebanyakan International
    if fraud_indices.size > 0:
        # sebagian kecil fraud masih domestik supaya tidak 100% obvious
        loc_fraud = rng.choice(
            np.array(["International", "Jakarta", "Bandung"]),
            size=fraud_indices.size,
            p=[0.8, 0.1, 0.1],
        )
        location[fraud_indices] = loc_fraud

    # ===========================================================
    # 6. Transaction ID
    # ===========================================================
    width = len(str(n_rows))
    transaction_id = [f"N{str(i).zfill(width)}" for i in range(1, n_rows + 1)]

    # ===========================================================
    # 7. Final DataFrame (label di kolom terakhir)
    # ===========================================================
    df = pd.DataFrame({
        "transaction_id": transaction_id,
        "timestamp": timestamp_str,
        "amount": amount_log,
        "hashed_merchant_id": hashed_merchant_id,
        "hashed_customer_id": hashed_customer_id,
        "location": location,
        "is_fraud": is_fraud.astype(int),
    })

    return df


if __name__ == "__main__":
    df_n = generate_bank_n_dataset(
        n_rows=20000,
        fraud_ratio=0.12,   # 12% fraud: cukup tinggi untuk data terenkripsi high-risk
        seed=42,
    )

    print("Total rows:", len(df_n))
    print("Fraud ratio:", df_n["is_fraud"].mean())

    # Sesuaikan dengan penamaan dataset yang kamu pakai di script FL
    df_n.to_csv("../data/bank_N_data.csv", index=False)
    print("Saved → ../data/bank_N_DATA.csv")
