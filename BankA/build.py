import shutil
from pathlib import Path
import numpy as np
import joblib
import tensorflow as tf

NPZ_PATH = Path("Models/models_global_model_fedavg_20251218_195152.npz")
PREPROC_PATH = Path("models_global/fitur_global.pkl")

OUT_DIR = Path(r"C:\KP\MATERI\BANK - FIX\BankA\Models\global_savedmodel")

def build_keras_model(input_dim: int) -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

def load_npz_weights(npz_path: Path):
    npz = np.load(str(npz_path))
    keys = sorted(npz.files, key=lambda k: int(k.split("_")[1]))
    return [npz[k] for k in keys]

def infer_input_dim_from_preproc(preproc):
    # MODE DICT (hashing)
    if isinstance(preproc, dict) and "NUM_COLS" in preproc and "CAT_COLS" in preproc and "HASHER_DIM" in preproc:
        num_dim = len(preproc["NUM_COLS"])
        cat_dim = len(preproc["CAT_COLS"]) * int(preproc["HASHER_DIM"])
        return num_dim + cat_dim

    # MODE LIST (one-hot feature list)
    if isinstance(preproc, (list, tuple)):
        return len(preproc)

    raise ValueError("Format fitur_global.pkl tidak dikenali (harus dict mode hashing atau list feature cols).")

def main():
    if not PREPROC_PATH.exists():
        raise FileNotFoundError(f"Preproc tidak ditemukan: {PREPROC_PATH}")
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"NPZ tidak ditemukan: {NPZ_PATH}")

    preproc = joblib.load(str(PREPROC_PATH))
    input_dim = infer_input_dim_from_preproc(preproc)
    print(f"✅ input_dim dari fitur_global.pkl = {input_dim}")

    model = build_keras_model(input_dim)
    model(tf.zeros((1, input_dim), dtype=tf.float32))  # build

    weights = load_npz_weights(NPZ_PATH)
    model.set_weights(weights)
    print("✅ set_weights berhasil")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Keras 3 export SavedModel folder
    model.export(str(OUT_DIR))
    print(f"✅ SavedModel dibuat di: {OUT_DIR}")

if __name__ == "__main__":
    main()
