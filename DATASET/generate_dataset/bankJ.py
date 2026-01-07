import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_bank_j_dataset(n_rows=15000, fraud_ratio=0.06, seed=42):
    """
    Optimized Generator for Bank J (Broker Saham & Crypto)
    Improved for Federated Learning stability:
    - Lower fraud ratio (6%)
    - Reduced synthetic tickers (80, not 200)
    - More realistic fraud patterns
    - Less overfitting to withdrawal/crypto
    """

    rng = np.random.default_rng(seed)

    # ===========================================================
    # 1. FRAUD LABEL
    # ===========================================================
    is_fraud = (rng.random(n_rows) < fraud_ratio).astype(int)
    fraud_mask = is_fraud == 1

    # ===========================================================
    # 2. HIGH-CARDINALITY ASSET TICKERS
    # ===========================================================
    stock_tickers = [
        "BBCA", "BBRI", "BBNI", "BMRI", "TLKM", "UNVR",
        "ICBP", "GOTO", "ASII", "BBTN", "SIDO", "ANTM",
        "INCO", "PGAS", "CPIN", "KLBF"
    ]

    crypto_tickers = [
        "BTC", "ETH", "USDT", "BNB", "SOL",
        "XRP", "ADA", "DOGE", "LTC", "DOT"
    ]

    synthetic_tickers = [f"TICK{i:03d}" for i in range(1, 81)]   # <= 80 only

    all_tickers = np.array(stock_tickers + crypto_tickers + synthetic_tickers)

    asset_ticker = rng.choice(all_tickers, size=n_rows)

    # ===========================================================
    # 3. TRANSACTION TYPE
    # ===========================================================
    transaction_type = rng.choice(
        ["BUY", "SELL", "DEPOSIT", "WITHDRAWAL"],
        size=n_rows,
        p=[0.50, 0.25, 0.15, 0.10]
    )

    # Fraud override (not always withdrawal)
    fraud_actions = rng.choice(
        ["WITHDRAWAL", "SELL"],
        size=fraud_mask.sum(),
        p=[0.85, 0.15]   # mostly withdrawal but some SELL fraud
    )
    transaction_type[fraud_mask] = fraud_actions

    # ===========================================================
    # 4. AMOUNT DISTRIBUTION
    # ===========================================================
    amount_legit_trade = rng.integers(3_000_000, 70_000_000, size=n_rows)
    amount_legit_deposit = rng.integers(10_000_000, 180_000_000, size=n_rows)

    # Fraud types:
    amount_fraud_small = rng.integers(20_000_000, 80_000_000, size=n_rows)     # small fraud
    amount_fraud_big = rng.integers(120_000_000, 900_000_000, size=n_rows)     # large withdrawal fraud

    amount = amount_legit_trade.copy()

    deposit_mask = (transaction_type == "DEPOSIT") & (~fraud_mask)
    amount[deposit_mask] = amount_legit_deposit[deposit_mask]

    # Assign fraud amounts
    small_f_mask = fraud_mask & (rng.random(n_rows) < 0.40)  # 40% small fraud
    big_f_mask = fraud_mask & (~small_f_mask)                # rest big fraud

    amount[small_f_mask] = amount_fraud_small[small_f_mask]
    amount[big_f_mask] = amount_fraud_big[big_f_mask]

    amount = np.log1p(amount)

    # ===========================================================
    # 5. OVERRIDE TICKERS FOR FRAUD (not always crypto)
    # ===========================================================
    crypto_pool = np.array(["BTC", "ETH", "USDT", "BNB", "SOL"])
    stock_pool_highrisk = np.array(["GOTO", "BBTN", "ASII"])

    # 80% fraud crypto, 20% fraud stock
    crypto_mask = fraud_mask & (rng.random(n_rows) < 0.80)
    stock_mask = fraud_mask & (~crypto_mask)

    asset_ticker[crypto_mask] = rng.choice(crypto_pool, size=crypto_mask.sum())
    asset_ticker[stock_mask] = rng.choice(stock_pool_highrisk, size=stock_mask.sum())

    # ===========================================================
    # 6. TIMESTAMP
    # ===========================================================
    start_dt = datetime(2025, 9, 19, 12, 30)
    timestamps = []

    for i in range(n_rows):
        start_dt += timedelta(minutes=int(rng.integers(1, 4)))
        timestamps.append(start_dt.strftime("%Y-%m-%d %H:%M"))

    # ===========================================================
    # 7. TRANSACTION ID
    # ===========================================================
    width = len(str(n_rows))
    transaction_id = [f"J{str(i).zfill(width)}" for i in range(1, n_rows + 1)]

    # ===========================================================
    # 8. BUILD DATAFRAME
    # ===========================================================
    df = pd.DataFrame({
        "transaction_id": transaction_id,
        "timestamp": timestamps,
        "amount": amount,
        "asset_ticker": asset_ticker,
        "transaction_type": transaction_type,
        "is_fraud": is_fraud,
    })

    return df


if __name__ == "__main__":
    df_j = generate_bank_j_dataset(
        n_rows=15000,
        fraud_ratio=0.06,
        seed=42
    )

    print("Total rows:", len(df_j))
    print("Fraud ratio:", df_j["is_fraud"].mean())
    df_j.to_csv("../data/bank_J_data.csv", index=False)
    print("Saved → ../data/bank_J_data.csv")
