import numpy as np
import pandas as pd


def generate_bank_m_dataset(
    n_rows: int = 30000,
    fraud_ratio: float = 0.08,
    seed: int = 42,
):
    """
    Generate synthetic dataset for Bank M (Global Payment Gateway).

    Fitur:
    - tx_id                : ID transaksi
    - unix_timestamp       : waktu (detik, monoton naik)
    - amount               : nilai transaksi (log1p USD-equivalent)
    - merchant_country     : negara merchant (US, JP, CN, RU, SG, KR)
    - avg_amount_7d        : rata-rata amount 7 hari terakhir (log1p)
    - time_since_last_tx_sec : selisih detik dari transaksi sebelumnya
    - is_fraud             : label (0/1) -- KOLOM TERAKHIR

    Fraud pattern:
    - negara high risk (RU, CN)
    - amount besar
    - time_since_last_tx_sec kecil (1–3 detik, burst)
    """

    rng = np.random.default_rng(seed)

    # ===========================================================
    # 1. Label Fraud
    # ===========================================================
    is_fraud = (rng.random(n_rows) < fraud_ratio).astype(int)
    fraud_mask = is_fraud == 1

    # ===========================================================
    # 2. UNIX TIMESTAMP (detik)
    # ===========================================================
    # start sekitar waktu acak (misal 2025-09-19 12:30 UTC)
    # kita cukup pakai angka unix sintetis
    base_unix = 1_758_250_260  # arbitrary, konsisten dengan contoh
    gaps = rng.integers(1, 300, size=n_rows)  # 1–300 detik antar transaksi
    unix_timestamp = np.zeros(n_rows, dtype=np.int64)
    unix_timestamp[0] = base_unix
    for i in range(1, n_rows):
        unix_timestamp[i] = unix_timestamp[i - 1] + int(gaps[i])

    # time_since_last_tx_sec = selisih antar timestamp
    time_since_last_tx_sec = np.zeros(n_rows, dtype=np.int64)
    time_since_last_tx_sec[0] = 86400  # default 1 hari untuk transaksi pertama
    time_since_last_tx_sec[1:] = np.diff(unix_timestamp)

    # Untuk fraud: paksa selisih menjadi sangat kecil (1–3 detik)
    fraud_indices = np.where(fraud_mask)[0]
    for idx in fraud_indices:
        if idx == 0:
            continue
        time_since_last_tx_sec[idx] = int(rng.integers(1, 4))
        unix_timestamp[idx] = unix_timestamp[idx - 1] + time_since_last_tx_sec[idx]

    # Pastikan setelah adjust, urutan masih non-decreasing
    for i in range(1, n_rows):
        if unix_timestamp[i] <= unix_timestamp[i - 1]:
            unix_timestamp[i] = unix_timestamp[i - 1] + max(1, int(time_since_last_tx_sec[i]))

    # Recompute time_since_last_tx_sec setelah penyesuaian
    time_since_last_tx_sec[0] = 86400
    time_since_last_tx_sec[1:] = np.diff(unix_timestamp)

    # ===========================================================
    # 3. AMOUNT
    # ===========================================================
    # Transaksi normal: 5 – 200
    amount_legit = rng.uniform(5.0, 200.0, size=n_rows)

    # Fraud: nominal jauh lebih besar
    amount_fraud = rng.uniform(150.0, 600.0, size=n_rows)

    amount = np.where(fraud_mask, amount_fraud, amount_legit)

    # ===========================================================
    # 4. MERCHANT COUNTRY
    # ===========================================================
    countries = np.array(["US", "JP", "CN", "RU", "SG", "KR"])
    probs_legit = np.array([0.45, 0.20, 0.10, 0.05, 0.10, 0.10])

    merchant_country = rng.choice(countries, size=n_rows, p=probs_legit)

    # Fraud override: negara high risk (CN, RU) lebih sering
    fraud_countries = rng.choice(
        np.array(["CN", "RU"]),
        size=fraud_indices.size,
        p=[0.4, 0.6],
    )
    merchant_country[fraud_indices] = fraud_countries

    # ===========================================================
    # 5. AVG_AMOUNT_7D (rolling window 7 hari)
    # ===========================================================
    # 7 hari = 7 * 86400 detik
    window_seconds = 7 * 86400

    # Kita hitung rata-rata amount dari transaksi dalam 7 hari sebelum setiap tx
    avg_amount_7d = np.zeros(n_rows, dtype=float)
    # sliding window index
    start_idx = 0
    for i in range(n_rows):
        current_time = unix_timestamp[i]
        # geser start_idx sampai window 7 hari
        while start_idx < i and unix_timestamp[start_idx] < current_time - window_seconds:
            start_idx += 1
        if i == 0:
            avg_amount_7d[i] = amount[i]
        else:
            # rata-rata amount dari start_idx..i-1
            if i > start_idx:
                avg_amount_7d[i] = amount[start_idx:i].mean()
            else:
                avg_amount_7d[i] = amount[i - 1]

    # ===========================================================
    # 6. Log-transform amount & avg_amount_7d (stabil)
    # ===========================================================
    amount_log = np.log1p(amount)
    avg_amount_7d_log = np.log1p(avg_amount_7d)

    # ===========================================================
    # 7. TX ID
    # ===========================================================
    width = len(str(n_rows))
    tx_id = [f"M{str(i).zfill(width)}" for i in range(1, n_rows + 1)]

    # ===========================================================
    # 8. Susun DataFrame (label di kolom terakhir)
    # ===========================================================
    df = pd.DataFrame({
        "tx_id": tx_id,
        "unix_timestamp": unix_timestamp,
        "amount": amount_log,
        "merchant_country": merchant_country,
        "avg_amount_7d": avg_amount_7d_log,
        "time_since_last_tx_sec": time_since_last_tx_sec,
        "is_fraud": is_fraud.astype(int),
    })

    return df


if __name__ == "__main__":
    df_m = generate_bank_m_dataset(
        n_rows=30000,
        fraud_ratio=0.08,  # 8% fraud untuk payment gateway global
        seed=42,
    )

    print("Total rows:", len(df_m))
    print("Fraud ratio:", df_m["is_fraud"].mean())

    # simpan ke folder data (relatif ke script)
    df_m.to_csv("../data/bank_M_data.csv", index=False)
    print("Saved → ../data/bank_M_data.csv")
