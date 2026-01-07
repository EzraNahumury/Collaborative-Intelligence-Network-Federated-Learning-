# ============================================================
# 🧠 TRAIN CLIENT TFF
# ============================================================
import os, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import tensorflow_federated as tff
import time
from datetime import datetime


# ============================================================
# ⚙️ ARGUMEN
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--bank", type=str, default="C_data", help="Kode bank: A..F")
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--models_dir", type=str, default="models")
parser.add_argument("--global_dir", type=str, default="models_global")
parser.add_argument("--n_clients", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--rounds", type=int, default=10)
parser.add_argument("--lr_client", type=float, default=5e-4)
parser.add_argument("--lr_server", type=float, default=1e-3)
args = parser.parse_args()

BANK = f"bank_{args.bank.upper()}"
DATA_PATH = Path(args.data_dir) / f"{BANK}.csv"
GLOBAL_FEATS_PATH = Path(args.global_dir) / "fitur_global.pkl"
SAVE_DIR = Path(args.models_dir) / f"saved_{BANK}_tff"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f" Training Federated Client: {BANK}")
print(f"📂 Data: {DATA_PATH}")
print(f" Fitur Global: {GLOBAL_FEATS_PATH}")

# ============================================================
#  LOAD DATA
# ============================================================
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data tidak ditemukan: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
if "is_fraud" not in df.columns:
    raise ValueError("Kolom 'is_fraud' tidak ditemukan!")

y_all = df["is_fraud"].astype("float32").values
X_raw = df.drop(columns=["is_fraud"]).copy()

# hilangkan kolom ID & timestamp
for c in ["transaction_id", "timestamp"]:
    if c in X_raw.columns:
        X_raw.drop(columns=[c], inplace=True)

print(f" {len(df):,} baris, {len(X_raw.columns)} fitur awal")

# ============================================================
#  MUAT FITUR GLOBAL (dict atau list)
# ============================================================
if not GLOBAL_FEATS_PATH.exists():
    raise FileNotFoundError(f" {GLOBAL_FEATS_PATH} tidak ditemukan!")

GLOBAL_FEATURES = joblib.load(GLOBAL_FEATS_PATH)
MODE = "DICT" if isinstance(GLOBAL_FEATURES, dict) else "LIST"
print(f" Mode fitur global terdeteksi: {MODE}")

# ============================================================
#  PREPROCESS
# ============================================================
def clean_numeric(series: pd.Series):
    s = series.astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def preprocess_mode_dict(X_df: pd.DataFrame):
    NUM_COLS = GLOBAL_FEATURES["NUM_COLS"]
    CAT_COLS = GLOBAL_FEATURES["CAT_COLS"]
    HASHER_DIM = int(GLOBAL_FEATURES["HASHER_DIM"])
    SCALER = GLOBAL_FEATURES["SCALER"]

    mins = pd.Series(SCALER["data_min_"], index=NUM_COLS)
    rng = pd.Series(SCALER["data_range_"], index=NUM_COLS).replace(0, 1.0)

    for c in NUM_COLS:
        if c not in X_df.columns:
            X_df[c] = 0
        else:
            X_df[c] = clean_numeric(X_df[c])

    X_num = ((X_df[NUM_COLS] - mins) / rng).fillna(0.0).astype("float32")

    from sklearn.feature_extraction import FeatureHasher
    hasher = FeatureHasher(n_features=HASHER_DIM, input_type="string", alternate_sign=False)

    hashed_parts = []
    for col in CAT_COLS:
        vals = X_df[col].fillna("NA").astype(str).values if col in X_df.columns else np.array(["NA"] * len(X_df))
        hashed = hasher.transform([[v] for v in vals]).toarray().astype("float32")
        hdf = pd.DataFrame(hashed, columns=[f"{col}_hash{i}" for i in range(HASHER_DIM)], dtype="float32")
        hashed_parts.append(hdf)

    X_cat = pd.concat(hashed_parts, axis=1) if hashed_parts else pd.DataFrame(index=X_df.index)
    X_all = pd.concat([X_num, X_cat], axis=1)
    meta = {
        "MODE": "DICT",
        "NUM_COLS": NUM_COLS,
        "CAT_COLS": CAT_COLS,
        "HASHER_DIM": HASHER_DIM,
        "SCALER": SCALER,
        "FEATURE_DIM": X_all.shape[1],
    }
    return X_all, meta

def preprocess_mode_list(X_df: pd.DataFrame):
    FEATURE_LIST = list(GLOBAL_FEATURES)
    cat_cols_local = X_df.select_dtypes(include=["object", "bool"]).columns.tolist()
    X_enc = pd.get_dummies(X_df, columns=cat_cols_local, drop_first=False)
    for col in FEATURE_LIST:
        if col not in X_enc.columns:
            X_enc[col] = 0.0
    X_enc = X_enc[FEATURE_LIST].astype("float32")
    meta = {"MODE": "LIST", "FEATURE_LIST": FEATURE_LIST, "FEATURE_DIM": X_enc.shape[1]}
    return X_enc, meta

if MODE == "DICT":
    X_ready, META = preprocess_mode_dict(X_raw)
else:
    X_ready, META = preprocess_mode_list(X_raw)

FEATURE_DIM = META["FEATURE_DIM"]
print(f" Fitur siap: {FEATURE_DIM} kolom total")

# ============================================================
#  SPLIT CLIENTS
# ============================================================
def to_tf_dataset(X_df, y, batch):
    ds = tf.data.Dataset.from_tensor_slices(
        (X_df.values.astype("float32"), y.astype("float32").reshape(-1, 1))
    )
    return ds.shuffle(min(len(X_df), 8192)).batch(batch).prefetch(tf.data.AUTOTUNE)

def split_clients(X_df, y, n_clients, batch):
    idx = np.arange(len(X_df)); np.random.shuffle(idx)
    size = len(X_df) // n_clients
    clients = []
    for i in range(n_clients):
        s = i * size
        e = (i + 1) * size if i < n_clients - 1 else len(X_df)
        clients.append(to_tf_dataset(X_df.iloc[idx[s:e]], y[idx[s:e]], batch))
    return clients

clients = split_clients(X_ready, y_all, args.n_clients, args.batch_size)
print(f" {len(clients)} klien federated data siap digunakan.")

# ============================================================
#  MODEL
# ============================================================
def build_keras_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    return model

def model_fn():
    keras_model = build_keras_model(FEATURE_DIM)
    return tff.learning.models.from_keras_model(
        keras_model=keras_model,
        input_spec=clients[0].element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
                 tf.keras.metrics.AUC(curve="PR", name="pr_auc")]
    )

# ============================================================
#  FEDERATED TRAINING (Weighted FedAvg)
# ============================================================
process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=tff.learning.optimizers.build_adam(learning_rate=args.lr_client),
    server_optimizer_fn=tff.learning.optimizers.build_adam(learning_rate=args.lr_server),
)

state = process.initialize()
prev_acc = 0.0
history = []

print("\ Mulai Federated Training ===========================")
for r in range(1, args.rounds + 1):
    state, metrics = process.next(state, clients)
    m_cw = metrics['client_work']['train']
    acc = float(m_cw.get('binary_accuracy', 0.0))
    prauc = float(m_cw.get('pr_auc', 0.0))
    loss = float(m_cw.get('loss', 0.0))

    history.append({"round": r, "acc": acc, "pr_auc": prauc, "loss": loss})
    print(f"[{BANK}] Round {r:02d} | acc={acc:.4f} | pr_auc={prauc:.4f} | loss={loss:.4f}")

    if r > 3 and abs(acc - prev_acc) < 1e-4:
        print(f" Early stop di round {r}, metrik sudah stabil.")
        break
    prev_acc = acc

# ============================================================
# 💾 SAVE MODEL (Keras SavedModel + NPZ) + LOG AKURASI & HISTORY
# ============================================================

# Assign weights dari state ke keras_model
keras_model = build_keras_model(FEATURE_DIM)
process.get_model_weights(state).assign_weights_to(keras_model)

# 1) Simpan Keras SavedModel (folder)
keras_model.save(SAVE_DIR, include_optimizer=False)
print(f"\nModel {BANK} disimpan di {SAVE_DIR}")

# 2) Simpan bobot sebagai NPZ (timestamped) di SAVE_DIR
timestamp = time.strftime("%Y%m%d_%H%M%S")
npz_path = SAVE_DIR / f"{timestamp}.npz"

weights_list = keras_model.get_weights()  # list of numpy arrays
np.savez_compressed(npz_path, *weights_list)
print(f"Bobot model disimpan sebagai NPZ: {npz_path}")

# 3) Simpan preprocess meta & history JSON (tetap berguna)
joblib.dump(META, SAVE_DIR / f"preprocess_{BANK}.pkl")
with open(SAVE_DIR / f"history_{BANK}.json", "w") as f:
    json.dump(history, f, indent=2)

# 4) Simpan history accuracy lengkap ke file (tab-separated)
history_path = SAVE_DIR / "accuracy_history.txt"
if not history_path.exists():
    with open(history_path, "w") as hf:
        hf.write("bank\tround\tacc\tpr_auc\tloss\ttimestamp\n")

with open(history_path, "a") as hf:
    for entry in history:
        r = int(entry.get("round", -1))
        acc = float(entry.get("acc", 0.0))
        prauc = float(entry.get("pr_auc", 0.0))
        loss = float(entry.get("loss", 0.0))
        ts = datetime.utcnow().isoformat() + "Z"
        hf.write(f"{BANK}\t{r}\t{acc:.6f}\t{prauc:.6f}\t{loss:.6f}\t{ts}\n")

print(f"History akurasi ditulis/ditambah ke {history_path}")

# 5) Perbarui best_accuracy.txt (ambil akurasi tertinggi dari history)
# DISABLED: best_accuracy.txt tidak dipakai untuk sementara
# best_acc = float(np.max([h.get("acc", 0.0) for h in history])) if history else 0.0
# best_path = SAVE_DIR / "best_accuracy.txt"
# with open(best_path, "w") as bf:
#     bf.write(f"{best_acc:.6f}\n")
# print(f"Best accuracy ({best_acc:.6f}) ditulis ke {best_path}")
print(f"\nModel {BANK} dan log tersimpan di {SAVE_DIR}")