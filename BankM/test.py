import os
import numpy as np
import joblib
import pandas as pd
from keras.layers import TFSMLayer
from colorama import Fore, Style, init
from sklearn.metrics import precision_recall_curve, roc_curve

init(autoreset=True)

# ============================================================
# ⚙️ KONFIGURASI
# ============================================================
THRESHOLD_MODE = "AUTO"   # "AUTO" atau "MANUAL"
THRESHOLD_MANUAL = 0.5    # jika MANUAL

MODEL_PATH = "models_round2/saved_bank_M_tff"
PREPROC_PATH = "models_global/fitur_global_test.pkl"

# ============================================================
# 🔎 Preprocessing Baru (Tanpa Hashing)
# ============================================================
def preprocess_transaction(data, preproc):
    """Konversi transaksi dict menjadi vektor model global (One-hot + Scaling)."""

    if not isinstance(preproc, dict):
        raise TypeError("File fitur_global.pkl tidak valid! Format harus dictionary.")

    FEATURE_COLS = preproc["FEATURE_COLS"]
    SCALER = preproc["SCALER"]

    df = pd.DataFrame([data])
    df = pd.get_dummies(df, drop_first=False)
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    df = SCALER.transform(df)

    return df.astype("float32")

# ============================================================
# ⚙️ Threshold Otomatis
# ============================================================
def auto_threshold(y_true, y_prob):
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    try:
        precision, recall, th_pr = precision_recall_curve(y_true, y_prob)
        f1s = (2 * precision * recall) / np.clip(precision + recall, 1e-9, None)
        t_pr = th_pr[np.argmax(f1s)]
    except Exception:
        t_pr = 0.5
    try:
        fpr, tpr, th_roc = roc_curve(y_true, y_prob)
        t_roc = th_roc[np.argmax(tpr - fpr)]
    except Exception:
        t_roc = 0.5
    return float(np.clip((t_pr + t_roc) / 2, 0, 1))

# ============================================================
# 🧠 Uji Model Global
# ============================================================
def test_global_model(model_path, preproc_path, cases, label):
    print(f"\n{'='*70}")
    print(f"🌍 TEST MODEL GLOBAL di Data ")
    print(f"{'='*70}")

    try:
        model = TFSMLayer(model_path, call_endpoint="serving_default")
        preproc = joblib.load(preproc_path)
        print(Fore.GREEN + " Model & fitur global berhasil dimuat!" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f" Gagal memuat: {e}" + Style.RESET_ALL)
        return 0, 0, 0.0  # correct, total, acc

    probs, labels = [], []
    for i, (data, expected) in enumerate(cases):
        try:
            X = preprocess_transaction(data, preproc)
            y_pred = list(model(X).values())[0].numpy().flatten()[0]
            probs.append(y_pred)
            labels.append(expected)
        except Exception as e:
            print(Fore.RED + f" Gagal uji kasus {i+1}: {e}" + Style.RESET_ALL)

    if not probs:
        print(Fore.YELLOW + " Tidak ada hasil prediksi valid!" + Style.RESET_ALL)
        return 0, 0, 0.0

    threshold = auto_threshold(labels, probs) if THRESHOLD_MODE == "AUTO" else THRESHOLD_MANUAL

    correct = 0
    for i, (p, y) in enumerate(zip(probs, labels), 1):
        pred = int(p >= threshold)
        icon = "✅" if pred == y else "❌"
        if pred == y:
            correct += 1
        print(f"   {icon} Case {i:02d} | Exp={y} | Pred={pred} | Prob={p:.4f}")

    total = len(labels)
    acc = correct / total * 100
    print(f"\n Akurasi Global di {label}: {Fore.CYAN}{acc:.2f}% ({correct}/{total}){Style.RESET_ALL}")
    print("-"*70)

    return correct, total, acc

# ============================================================
# 🧪 TEST CASE PER BANK
# ============================================================

bank_a_cases = [
    ({"amount": 200000, "merchant_category": "ecommerce", "location": "Jakarta", "is_international": 0, "transaction_frequency_24h": 5}, 0),
    ({"amount": 7500000, "merchant_category": "travel", "location": "Singapore", "is_international": 1, "transaction_frequency_24h": 1}, 1),
    ({"amount": 120000, "merchant_category": "subscription", "location": "Bandung", "is_international": 0, "transaction_frequency_24h": 7}, 0),
    ({"amount": 9600000, "merchant_category": "investment", "location": "Online", "is_international": 1, "transaction_frequency_24h": 1}, 1),
]

bank_b_cases = [
    ({"amount": 300000, "merchant_category": "groceries", "location": "Jakarta", "is_international": 0, "transaction_frequency_24h": 9}, 0),
    ({"amount": 9800000, "merchant_category": "electronics", "location": "Tokyo", "is_international": 1, "transaction_frequency_24h": 2}, 1),
    ({"amount": 210000, "merchant_category": "food", "location": "Bali", "is_international": 0, "transaction_frequency_24h": 6}, 0),
    ({"amount": 8500000, "merchant_category": "luxury_goods", "location": "Hong Kong", "is_international": 1, "transaction_frequency_24h": 1}, 1),
]

bank_c_cases = [
    ({"amount": 4500000, "merchant_category": "crypto", "location": "Online", "is_international": 1, "transaction_frequency_24h": 1}, 1),
    ({"amount": 600000, "merchant_category": "retail", "location": "Surabaya", "is_international": 0, "transaction_frequency_24h": 7}, 0),
    ({"amount": 7800000, "merchant_category": "loan", "location": "Singapore", "is_international": 1, "transaction_frequency_24h": 1}, 1),
    ({"amount": 250000, "merchant_category": "payment_gateway", "location": "Jakarta", "is_international": 0, "transaction_frequency_24h": 12}, 0),
]

bank_d_cases = [
    ({"amount": 1800000, "merchant_category": "transport", "location": "Jakarta", "is_international": 0, "transaction_frequency_24h": 3}, 0),
    ({"amount": 13500000, "merchant_category": "travel", "location": "Bangkok", "is_international": 1, "transaction_frequency_24h": 1}, 1),
    ({"amount": 90000, "merchant_category": "parking", "location": "Bandung", "is_international": 0, "transaction_frequency_24h": 8}, 0),
    ({"amount": 11000000, "merchant_category": "hotel", "location": "Dubai", "is_international": 1, "transaction_frequency_24h": 1}, 1),
]

bank_e_cases = [
    ({"amount": 500000, "merchant_category": "insurance", "location": "Jakarta", "is_international": 0, "transaction_frequency_24h": 2}, 0),
    ({"amount": 12000000, "merchant_category": "insurance", "location": "London", "is_international": 1, "transaction_frequency_24h": 1}, 1),
    ({"amount": 250000, "merchant_category": "utilities", "location": "Bali", "is_international": 0, "transaction_frequency_24h": 4}, 0),
    ({"amount": 10500000, "merchant_category": "medical_billing", "location": "Germany", "is_international": 1, "transaction_frequency_24h": 1}, 1),
]

bank_f_cases = [
    ({"amount": 380000, "merchant_category": "entertainment", "location": "Jakarta", "is_international": 0, "transaction_frequency_24h": 6}, 0),
    ({"amount": 9200000, "merchant_category": "donation", "location": "New York", "is_international": 1, "transaction_frequency_24h": 1}, 1),
    ({"amount": 140000, "merchant_category": "streaming", "location": "Online", "is_international": 0, "transaction_frequency_24h": 5}, 0),
    ({"amount": 17000000, "merchant_category": "gambling", "location": "Macau", "is_international": 1, "transaction_frequency_24h": 1}, 1),
]

if __name__ == "__main__":
    totals_correct = 0
    totals_total = 0
    per_bank = []

    for cases, label in [
        (bank_a_cases, "BANK A"),
        (bank_b_cases, "BANK B"),
        (bank_c_cases, "BANK C"),
        (bank_d_cases, "BANK D"),
        (bank_e_cases, "BANK E"),
        (bank_f_cases, "BANK F"),
    ]:
        c, t, acc = test_global_model(MODEL_PATH, PREPROC_PATH, cases, label)
        totals_correct += c
        totals_total += t
        per_bank.append((label, acc, c, t))

    # ✅ TOTAL keseluruhan
    if totals_total > 0:
        total_acc = totals_correct / totals_total * 100
        print(f"\n{'='*70}")
        print(f"🏁 TOTAL AKURASI (SEMUA TEST CASE)")
        print(f"{'='*70}")
        print(f" Total: {Fore.CYAN}{total_acc:.2f}% ({totals_correct}/{totals_total}){Style.RESET_ALL}")
        print("-"*70)
        
        # 💾 Simpan total accuracy ke best_accuracy.txt
        best_acc_path = os.path.join(MODEL_PATH, "best_accuracy.txt")
        with open(best_acc_path, "w") as f:
            f.write(f"{total_acc / 100:.6f}\n")
        print(f"\n💾 Total accuracy ({total_acc:.2f}%) disimpan ke {best_acc_path}")
    else:
        print(Fore.YELLOW + "\n Tidak ada test case valid untuk dihitung total accuracy!" + Style.RESET_ALL)
