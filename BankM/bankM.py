# ============================================================
# 🧠 TRAIN CLIENT TFF (Bank G – ROUND 2) + Base Model Global R1
# ============================================================

# ========== BLOK 1: LIMIT THREADING (anti kill) ==========
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # paksa CPU

import argparse, json, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import tensorflow_federated as tff

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass


# ============================================================
# ⚙️ ARGUMEN
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--bank", type=str, default="M", help="Kode bank: G–N")

parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--data_file", type=str, default="", help="Opsional override nama csv (contoh: bank_G_data_clean.csv)")

parser.add_argument("--models_dir", type=str, default="models_round2")

parser.add_argument("--global_dir", type=str, default="models_global")
parser.add_argument("--global_feats", type=str, default="fitur_global.pkl")

parser.add_argument("--global_model_r1", type=str, default="models_global_round1/global_savedmodel")

parser.add_argument("--n_clients", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--rounds", type=int, default=10)
parser.add_argument("--lr_client", type=float, default=5e-4)
parser.add_argument("--lr_server", type=float, default=1e-3)

parser.add_argument("--resume", action="store_true", help="Lanjutkan dari checkpoint jika ada")
args = parser.parse_args()


# ============================================================
# 📦 PATHS
# ============================================================
BANK = f"bank_{args.bank.upper()}"

# default sesuai folder kamu: data/bank_G_data_clean.csv
if args.data_file.strip():
    DATA_PATH = Path(args.data_dir) / args.data_file
else:
    DATA_PATH = Path(args.data_dir) / f"{BANK}_data_clean.csv"

GLOBAL_FEATS_PATH = Path(args.global_dir) / args.global_feats
GLOBAL_MODEL_R1_PATH = Path(args.global_model_r1)

SAVE_DIR = Path(args.models_dir) / f"saved_{BANK}_tff"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CKPT_DIR = SAVE_DIR / "ckpt"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = CKPT_DIR / "server_state.npz"
HIST_PATH  = CKPT_DIR / "history.json"

print(f"🚀 Training Federated Client (R2): {BANK}")
print(f"📂 Data         : {DATA_PATH}")
print(f"🌐 Fitur Global : {GLOBAL_FEATS_PATH}")
print(f"🌍 Global R1    : {GLOBAL_MODEL_R1_PATH}")
print(f"💾 Output model : {SAVE_DIR}")


# ============================================================
# 📂 LOAD DATA
# ============================================================
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data tidak ditemukan: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
if "is_fraud" not in df.columns:
    raise ValueError("Kolom 'is_fraud' tidak ditemukan!")

y_all = df["is_fraud"].astype("float32").values
X_raw = df.drop(columns=["is_fraud"]).copy()

# buang kolom non-fitur jika ada
for c in ["transaction_id", "timestamp", "tx_id", "unix_timestamp"]:
    if c in X_raw.columns:
        X_raw.drop(columns=[c], inplace=True)

print(f"✅ {len(df):,} baris, {len(X_raw.columns)} fitur awal")


# ============================================================
# 🌍 LOAD FITUR GLOBAL (dict / list)
# ============================================================
if not GLOBAL_FEATS_PATH.exists():
    raise FileNotFoundError(f"{GLOBAL_FEATS_PATH} tidak ditemukan!")

GLOBAL_FEATURES = joblib.load(GLOBAL_FEATS_PATH)

MODE = "DICT" if isinstance(GLOBAL_FEATURES, dict) else ("LIST" if isinstance(GLOBAL_FEATURES, (list, tuple)) else "UNKNOWN")
print(f"✅ Mode fitur global terdeteksi: {MODE}")

def clean_numeric(series: pd.Series):
    s = series.astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


# ============================================================
# 🧹 PREPROCESS (MODE DICT / LIST)
# ============================================================
def preprocess_mode_dict(X_df: pd.DataFrame):
    """
    Support 2 variasi key dict:
    A) NUM_COLS / CAT_COLS / HASHER_DIM / SCALER (punyamu di round1)
    B) NUMERIC_COLS / CATEGORICAL_COLS / HASHED_SIZE / SCALER (variasi lain)
    """
    keys = set(GLOBAL_FEATURES.keys())

    # normalisasi key
    if "NUM_COLS" in keys:
        NUM_COLS = list(GLOBAL_FEATURES["NUM_COLS"])
        CAT_COLS = list(GLOBAL_FEATURES.get("CAT_COLS", []))
        HASHER_DIM = int(GLOBAL_FEATURES.get("HASHER_DIM", GLOBAL_FEATURES.get("HASHED_SIZE", 0)))
    else:
        NUM_COLS = list(GLOBAL_FEATURES.get("NUMERIC_COLS", []))
        CAT_COLS = list(GLOBAL_FEATURES.get("CATEGORICAL_COLS", []))
        HASHER_DIM = int(GLOBAL_FEATURES.get("HASHED_SIZE", GLOBAL_FEATURES.get("HASHER_DIM", 0)))

    SCALER = GLOBAL_FEATURES.get("SCALER", None)
    if SCALER is None:
        raise KeyError("SCALER tidak ditemukan di fitur global dict.")

    # scaler kamu di round1 berupa dict: {"data_min_":..., "data_range_":...}
    # tapi kalau ternyata scaler adalah object sklearn, tetap bisa dipakai
    X_df = X_df.copy()

    for c in NUM_COLS:
        if c not in X_df.columns:
            X_df[c] = 0
        else:
            X_df[c] = clean_numeric(X_df[c])

    # numeric scaled
    X_num = X_df[NUM_COLS].astype("float32")

    if isinstance(SCALER, dict) and ("data_min_" in SCALER) and ("data_range_" in SCALER):
        mins = pd.Series(SCALER["data_min_"], index=NUM_COLS)
        rng  = pd.Series(SCALER["data_range_"], index=NUM_COLS).replace(0, 1.0)
        X_num = ((X_num - mins) / rng).fillna(0.0).astype("float32")
    else:
        # fallback: scaler object sklearn
        X_num = pd.DataFrame(SCALER.transform(X_num.values), columns=NUM_COLS).astype("float32")

    # categorical hashed
    from sklearn.feature_extraction import FeatureHasher
    hashed_parts = []

    if HASHER_DIM > 0 and len(CAT_COLS) > 0:
        hasher = FeatureHasher(n_features=HASHER_DIM, input_type="string", alternate_sign=False)

        for col in CAT_COLS:
            vals = X_df[col].fillna("NA").astype(str).values if col in X_df.columns else np.array(["NA"] * len(X_df))
            hashed = hasher.transform([[v] for v in vals]).toarray().astype("float32")
            hdf = pd.DataFrame(hashed, columns=[f"{col}_hash{i}" for i in range(HASHER_DIM)], dtype="float32")
            hashed_parts.append(hdf)

    X_cat = pd.concat(hashed_parts, axis=1) if hashed_parts else pd.DataFrame(index=X_df.index)
    X_all = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)

    meta = {
        "MODE": "DICT",
        "NUM_COLS": NUM_COLS,
        "CAT_COLS": CAT_COLS,
        "HASHER_DIM": HASHER_DIM,
        "SCALER": SCALER,
        "FEATURE_DIM": int(X_all.shape[1]),
    }
    return X_all, meta


def preprocess_mode_list(X_df: pd.DataFrame):
    """
    fitur_global.pkl berisi list[str] = FEATURE_LIST final (contoh: 39 kolom),
    maka kita one-hot lokal lalu align ke FEATURE_LIST.
    """
    FEATURE_LIST = list(GLOBAL_FEATURES)

    X_df = X_df.copy()
    cat_cols_local = X_df.select_dtypes(include=["object", "bool"]).columns.tolist()
    X_enc = pd.get_dummies(X_df, columns=cat_cols_local, drop_first=False)

    # pastikan semua feature ada
    for col in FEATURE_LIST:
        if col not in X_enc.columns:
            X_enc[col] = 0.0

    # ambil urutan sesuai global
    X_enc = X_enc[FEATURE_LIST].astype("float32")

    meta = {"MODE": "LIST", "FEATURE_LIST": FEATURE_LIST, "FEATURE_DIM": int(X_enc.shape[1])}
    return X_enc, meta


if MODE == "DICT":
    X_ready, META = preprocess_mode_dict(X_raw)
elif MODE == "LIST":
    X_ready, META = preprocess_mode_list(X_raw)
else:
    raise TypeError(f"Format fitur global tidak dikenali: {type(GLOBAL_FEATURES)}")

FEATURE_DIM = META["FEATURE_DIM"]
print(f"✅ Fitur siap: {FEATURE_DIM} kolom total")


# ============================================================
# 👥 SPLIT CLIENTS
# ============================================================
def to_tf_dataset(X_df, y, batch):
    ds = tf.data.Dataset.from_tensor_slices(
        (X_df.values.astype("float32"), y.astype("float32").reshape(-1, 1))
    )
    buffer = min(len(X_df), 8192)
    if buffer > 0:
        ds = ds.shuffle(buffer)
    return ds.batch(batch, drop_remainder=False).prefetch(1)

def split_clients(X_df, y, n_clients, batch):
    n = len(X_df)
    n_clients = max(1, min(int(n_clients), n))
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)

    splits = np.array_split(idx, n_clients)
    clients = []
    for sp in splits:
        if len(sp) == 0:
            continue
        clients.append(to_tf_dataset(X_df.iloc[sp], y[sp], batch))
    return clients

clients = split_clients(X_ready, y_all, args.n_clients, args.batch_size)
print(f"✅ {len(clients)} klien federated data siap digunakan.")


# ============================================================
# 🧠 MODEL + BASE MODEL dari Global R1
# ============================================================
def build_keras_model(input_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

# default base model baru
base_model = build_keras_model(FEATURE_DIM)

# coba load global model R1 untuk base init
if GLOBAL_MODEL_R1_PATH.exists():
    try:
        loaded = tf.keras.models.load_model(str(GLOBAL_MODEL_R1_PATH))

        # validasi input dim biar gak crash
        in_dim = None
        try:
            in_dim = loaded.input_shape[-1]
        except Exception:
            pass

        if isinstance(loaded, tf.keras.Model) and (in_dim == FEATURE_DIM):
            base_model = loaded
            print(f"✅ Base model diinisialisasi dari Global R1 (input_dim={in_dim}).")
        else:
            print(f"⚠️ Global R1 dimuat tapi input_dim tidak cocok ({in_dim} != {FEATURE_DIM}) → pakai model baru.")
    except Exception as e:
        print(f"⚠️ Gagal load Global R1 ({e}) → pakai model baru.")
else:
    print("ℹ️ Global R1 tidak ditemukan → pakai model baru.")


def model_fn():
    # clone supaya aman dipakai oleh TFF dan tiap call fresh model
    cloned = tf.keras.models.clone_model(base_model)
    try:
        cloned.set_weights(base_model.get_weights())
    except Exception:
        pass

    return tff.learning.models.from_keras_model(
        keras_model=cloned,
        input_spec=clients[0].element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ]
    )


# ============================================================
# 🔁 FEDERATED TRAINING (Weighted FedAvg) + CHECKPOINT
# ============================================================
process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=tff.learning.optimizers.build_adam(learning_rate=args.lr_client),
    server_optimizer_fn=tff.learning.optimizers.build_adam(learning_rate=args.lr_server),
)

state = process.initialize()
history = []

# resume checkpoint jika ada
if args.resume and STATE_PATH.exists():
    data = np.load(STATE_PATH, allow_pickle=True)
    # pastikan urutan w_2 sebelum w_10
    w_list = [data[k] for k in sorted(data.files, key=lambda k: int(k.split("_")[1]))]

    tmp = build_keras_model(FEATURE_DIM)
    tmp.set_weights(w_list)

    model_weights = tff.learning.models.ModelWeights.from_keras_model(tmp)
    state = process.set_model_weights(state, model_weights)

    if HIST_PATH.exists():
        history = json.load(open(HIST_PATH))

    print("🔁 Resume dari checkpoint sebelumnya.")

print("\n🚀 Mulai Federated Training (ROUND 2) ===========================")
prev_acc = 0.0
start_round = len(history) + 1

for r in range(start_round, args.rounds + 1):
    state, metrics = process.next(state, clients)

    m_cw = metrics.get("client_work", {}).get("train", metrics)
    acc   = float(m_cw.get("binary_accuracy", 0.0))
    prauc = float(m_cw.get("pr_auc", 0.0))
    loss  = float(m_cw.get("loss", 0.0))

    history.append({"round": r, "acc": acc, "pr_auc": prauc, "loss": loss})
    print(f"[{BANK}] Round {r:02d} | acc={acc:.4f} | pr_auc={prauc:.4f} | loss={loss:.4f}")

    # save checkpoint tiap round
    keras_tmp = build_keras_model(FEATURE_DIM)
    process.get_model_weights(state).assign_weights_to(keras_tmp)
    np.savez_compressed(
        STATE_PATH,
        **{f"w_{i}": w for i, w in enumerate([w.numpy() for w in keras_tmp.weights])}
    )
    json.dump(history, open(HIST_PATH, "w"), indent=2)

    if r > 3 and abs(acc - prev_acc) < 1e-4:
        print(f"🛑 Early stop di round {r}, metrik stabil.")
        break
    prev_acc = acc


# ============================================================
# 💾 SAVE MODEL (Keras SavedModel + NPZ) + LOG AKURASI & HISTORY
# ============================================================
keras_model = build_keras_model(FEATURE_DIM)
process.get_model_weights(state).assign_weights_to(keras_model)

# 1) Simpan Keras SavedModel (folder)
keras_model.save(str(SAVE_DIR), include_optimizer=False)
print(f"\n✅ Model {BANK} (R2) disimpan di {SAVE_DIR}")

# 2) Simpan bobot sebagai NPZ (timestamped)
timestamp = time.strftime("%Y%m%d_%H%M%S")
npz_path = SAVE_DIR / f"{timestamp}.npz"
np.savez_compressed(npz_path, *keras_model.get_weights())
print(f"✅ Bobot model disimpan sebagai NPZ: {npz_path}")

# 3) Simpan preprocess meta & history
joblib.dump(META, SAVE_DIR / f"preprocess_{BANK}.pkl")
with open(SAVE_DIR / f"history_{BANK}.json", "w") as f:
    json.dump(history, f, indent=2)

# 4) Append history accuracy
history_path = SAVE_DIR / "accuracy_history.txt"
if not history_path.exists():
    with open(history_path, "w") as hf:
        hf.write("bank\tround\tacc\tpr_auc\tloss\ttimestamp\n")

with open(history_path, "a") as hf:
    for entry in history:
        rr = int(entry.get("round", -1))
        a  = float(entry.get("acc", 0.0))
        p  = float(entry.get("pr_auc", 0.0))
        l  = float(entry.get("loss", 0.0))
        ts = datetime.utcnow().isoformat() + "Z"
        hf.write(f"{BANK}\t{rr}\t{a:.6f}\t{p:.6f}\t{l:.6f}\t{ts}\n")

print(f"✅ History akurasi ditulis ke {history_path}")
print(f"✅ Semua artefak tersimpan di: {SAVE_DIR}")
