import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_bank_l_dataset(n_rows=20000, fraud_ratio=0.005, seed=42):
    """
    Bank L (Microfinance Rural)
    - Extreme class imbalance: ~0.1% - 0.5% fraud
    - Pinjaman kecil untuk UMKM
    - Fraud = default pada pinjaman besar + tenor panjang
    """

    rng = np.random.default_rng(seed)

    # ===========================================================
    # 1. Fraud Label (EXTREMELY rare)
    # ===========================================================
    is_fraud = (rng.random(n_rows) < fraud_ratio).astype(int)
    fraud_mask = is_fraud == 1

    # ===========================================================
    # 2. LOAN AMOUNT
    # ===========================================================
    # Legit loans: 500k – 7 juta
    loan_legit = rng.integers(500_000, 7_000_000, size=n_rows)

    # Fraud loans: 10 juta – 50 juta (outliers)
    loan_fraud = rng.integers(10_000_000, 50_000_000, size=n_rows)

    loan_amount = np.where(fraud_mask, loan_fraud, loan_legit)

    # Log transform → stabil di Federated Learning
    loan_amount = np.log1p(loan_amount)

    # ===========================================================
    # 3. TENOR MONTHS
    # ===========================================================
    # Legit: 6, 12, 24, 36 months
    tenor_legit = rng.choice([6, 12, 24, 36], size=n_rows, p=[0.25, 0.40, 0.25, 0.10])

    # Fraud: cenderung tenor panjang
    tenor_fraud = rng.choice([24, 36], size=n_rows)

    loan_tenor = np.where(fraud_mask, tenor_fraud, tenor_legit)

    # ===========================================================
    # 4. LOCATION BRANCH (Rural)
    # ===========================================================
    branches = ["Sleman", "Bantul", "Gunungkidul", "Kulon Progo", "Magelang"]

    # Legit
    location_branch = rng.choice(branches, size=n_rows, p=[0.40, 0.30, 0.20, 0.05, 0.05])

    # Fraud → lebih sering dari cabang remote
    fraud_branches = rng.choice(
        ["Gunungkidul", "Kulon Progo", "Magelang"],
        size=fraud_mask.sum()
    )
    location_branch[fraud_mask] = fraud_branches

    # ===========================================================
    # 5. TIMESTAMP
    # ===========================================================
    start_dt = datetime(2025, 9, 19, 12, 50, 0)
    timestamps = []

    for i in range(n_rows):
        start_dt += timedelta(minutes=int(rng.integers(3, 18)))
        timestamps.append(start_dt.strftime("%Y-%m-%d %H:%M"))

    # ===========================================================
    # 6. LOAN ID
    # ===========================================================
    width = len(str(n_rows))
    loan_id = [f"L{str(i).zfill(width)}" for i in range(1, n_rows + 1)]

    # ===========================================================
    # 7. FINAL DATAFRAME
    # ===========================================================
    df = pd.DataFrame({
        "loan_id": loan_id,
        "timestamp": timestamps,
        "loan_amount": loan_amount,
        "loan_tenor_months": loan_tenor,
        "location_branch": location_branch,
        "is_default": is_fraud,   # FRAUD LABEL
    })

    return df


# ===========================================================
#  MAIN EXECUTION
# ===========================================================
if __name__ == "__main__":
    df_l = generate_bank_l_dataset(
        n_rows=20000,       # microfinance dataset cukup besar
        fraud_ratio=0.005,  # 0.3% fraud
        seed=42
    )

    print("Total rows:", len(df_l))
    print("Fraud ratio:", df_l["is_default"].mean())

    df_l.to_csv("../data/bank_L_data.csv", index=False)
    print("Saved → ../data/bank_L_data.csv")
