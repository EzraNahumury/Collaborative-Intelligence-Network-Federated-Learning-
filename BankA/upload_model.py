#!/usr/bin/env python3
import os
import time
import base64
import json
from pathlib import Path
import numpy as np
import requests
import tensorflow as tf

# -------------------------
# CONFIG
# -------------------------
SERVER_URL   = "https://federatedserver.up.railway.app"
CLIENT_NAME  = "BANK_A"
# Prefered model folder you'd normally save to. Script will try alternatives if not found.
MODEL_PATH   = Path("models/saved_bank_A_tff")
NPZ_PATH     = MODEL_PATH / f"{CLIENT_NAME.lower()}_weights.npz"
TIMEOUT      = 180
RETRY_LIMIT  = 3
LARGE_FILE_WARN_MB = 20  # warn if file > this size before base64-ing
HISTORY_TAIL_LINES = 200

# -------------------------
# UTIL
# -------------------------
def print_size(path: Path):
    try:
        size_mb = os.path.getsize(path) / 1024 / 1024
        return f"{size_mb:.2f} MB"
    except Exception:
        return "unknown"

def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")

def log_line(message: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open("upload_log.txt", "a") as f:
        f.write(f"[{CLIENT_NAME}] {ts} | {message}\n")

# -------------------------
# FIND model folder (robust)
# -------------------------
def find_model_folder(preferred: Path, client_name: str):
    # 1) preferred as-is
    if preferred.exists() and preferred.is_dir():
        return preferred

    # 2) try swapping lowercase/uppercase Models
    parent = preferred.parent
    if parent.exists():
        alt_parent = Path(str(parent).replace("models", "Models"))
        alt = alt_parent / preferred.name
        if alt.exists() and alt.is_dir():
            return alt

    # 3) search cwd for directories containing client_name
    name_lower = client_name.lower()
    cwd = Path.cwd()
    for p in cwd.rglob("*"):
        if p.is_dir() and name_lower in p.name.lower():
            # heuristic: contains saved model markers
            if (p / "variables").exists() or (p / "assets").exists() or any(p.glob("*.pb")) or any(p.glob("*.npz")):
                return p

    # 4) top-level Models / models search
    for top in [cwd / "Models", cwd / "models"]:
        if top.exists():
            for p in top.iterdir():
                if p.is_dir() and name_lower in p.name.lower():
                    return p

    return None

# -------------------------
# LOAD MODEL (Keras or SavedModel variables)
# -------------------------
def load_weights_from_model(model_path: Path):
    # Try to load as Keras SavedModel first
    try:
        print(f"Trying tf.keras.models.load_model('{model_path}')")
        model = tf.keras.models.load_model(model_path)
        print("Keras model loaded. Extracting weights...")
        weights = {}
        for w in model.weights:
            weights[safe_name(w.name)] = w.numpy()
        return weights
    except Exception as e:
        print("load_model failed, trying checkpoint extraction:", e)

    vars_dir = model_path / "variables"
    prefix = vars_dir / "variables"
    if not (vars_dir.exists() and (vars_dir / "variables.index").exists()):
        raise FileNotFoundError(f"No variables checkpoint found at {vars_dir}")

    ckpt = tf.train.load_checkpoint(str(prefix))
    var_map = ckpt.get_variable_to_shape_map()
    weights = {}
    for name in var_map:
        weights[safe_name(name)] = ckpt.get_tensor(name)
    return weights

def save_weights_npz_dict(weights_dict: dict, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, **weights_dict)
    print(f"Saved weights to {save_path} ({print_size(save_path)})")

# -------------------------
# READ LOCAL METRICS (best_accuracy + history)
# -------------------------
def collect_local_metrics(model_folder: Path):
    metrics = {"best_accuracy": None, "history": []}
    if model_folder is None:
        return metrics

    # common filenames to try
    candidates_best = ["best_accuracy.txt", "best_acc.txt", "BEST_ACCURACY.txt"]
    candidates_hist = ["accuracy_history.txt", "accuracy_history.log", "history.txt", "history_accuracy.txt"]

    # Try reading best_accuracy
    for fname in candidates_best:
        p = model_folder / fname
        if p.exists():
            try:
                with open(p, "r") as f:
                    content = f.read().strip()
                    if content:
                        first_line = content.splitlines()[0].strip()
                        metrics["best_accuracy"] = float(first_line)
                        break
            except Exception:
                pass

    # Try reading history (tail)
    for hname in candidates_hist:
        ph = model_folder / hname
        if ph.exists():
            try:
                with open(ph, "r") as f:
                    lines = [ln.strip() for ln in f.read().strip().splitlines() if ln.strip()]
                    if lines:
                        metrics["history"] = lines[-HISTORY_TAIL_LINES:]
                        break
            except Exception:
                pass

    return metrics

# -------------------------
# BUILD PAYLOAD (with metrics)
# -------------------------
def build_payload_with_metrics(npz_path: Path, client_name: str):
    # locate model folder to read metrics
    model_folder = find_model_folder(MODEL_PATH, client_name)
    local_metrics = collect_local_metrics(model_folder)

    with open(npz_path, "rb") as f:
        b = f.read()

    payload = {
        "client": client_name,
        "compressed_weights": base64.b64encode(b).decode("utf-8"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "best_accuracy": local_metrics.get("best_accuracy"),
            "history": local_metrics.get("history", [])
        }
    }
    return payload

# -------------------------
# UPLOAD (JSON base64)
# -------------------------
def upload_json_base64(npz_path: Path):
    file_size = npz_path.stat().st_size
    size_mb = file_size / 1024 / 1024
    if size_mb > LARGE_FILE_WARN_MB:
        print(f"WARNING: file is {size_mb:.1f} MB — base64 will expand it by ~33% and use memory.")

    payload = build_payload_with_metrics(npz_path, CLIENT_NAME)
    headers = {"Content-Type": "application/json"}
    url = f"{SERVER_URL}/upload-model"

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            print(f"[JSON] Attempt {attempt} -> {url}")
            start = time.time()
            res = requests.post(url, data=json.dumps(payload), headers=headers, timeout=TIMEOUT)
            dur = time.time() - start
            print(f"[JSON] Response {res.status_code} in {dur:.2f}s. Body: {res.text}")
            if res.status_code == 200:
                log_line(f"OK JSON {res.status_code} {res.text}")
                return True, res
            if res.status_code == 415 or (res.status_code >= 400 and "Unsupported Media Type" in res.text):
                log_line(f"JSON REJECTED {res.status_code} {res.text}")
                return False, res
            log_line(f"JSON ERROR {res.status_code} {res.text}")
        except requests.RequestException as e:
            print(f"[JSON] RequestException: {e}")
            log_line(f"JSON EXC {e}")
        time.sleep(2 ** attempt)
    return False, None

# -------------------------
# UPLOAD (multipart fallback)
# -------------------------
def upload_multipart(npz_path: Path):
    url = f"{SERVER_URL}/upload-model"

    # collect local metrics as JSON string
    model_folder = find_model_folder(MODEL_PATH, CLIENT_NAME)
    local_metrics = collect_local_metrics(model_folder)
    metrics_field = json.dumps(local_metrics)

    data = {"client": CLIENT_NAME, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "metrics": metrics_field}
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            print(f"[MULTIPART] Attempt {attempt} -> {url}")
            with open(npz_path, "rb") as fh:
                files = {"file": (npz_path.name, fh, "application/octet-stream")}
                start = time.time()
                res = requests.post(url, data=data, files=files, timeout=TIMEOUT)
                dur = time.time() - start
            print(f"[MULTIPART] Response {res.status_code} in {dur:.2f}s. Body: {res.text}")
            if res.status_code == 200:
                log_line(f"OK MULTIPART {res.status_code} {res.text}")
                return True, res
            log_line(f"MULTIPART ERROR {res.status_code} {res.text}")
        except requests.RequestException as e:
            print(f"[MULTIPART] RequestException: {e}")
            log_line(f"MULTIPART EXC {e}")
        time.sleep(2 ** attempt)
    return False, None

# -------------------------
# UPLOAD WITH FALLBACK
# -------------------------
def upload_with_fallback(npz_path: Path):
    success, res = upload_json_base64(npz_path)
    if success:
        return True
    if res is not None and (res.status_code == 415 or ("Unsupported Media Type" in res.text) or res.status_code >= 500):
        print("Server rejected JSON or returned server error; trying multipart fallback...")
        return upload_multipart(npz_path)[0]
    print("JSON didn't succeed; trying multipart fallback...")
    return upload_multipart(npz_path)[0]

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    # locate a usable model folder
    found_folder = find_model_folder(MODEL_PATH, CLIENT_NAME)
    if found_folder:
        print(f"Using model folder: {found_folder}")
        # if NPZ not found, prefer npz inside found folder
        candidate_npz = found_folder / f"{CLIENT_NAME.lower()}_weights.npz"
        if candidate_npz.exists():
            NPZ_PATH = candidate_npz
        else:
            # keep NPZ_PATH as configured, may be created below
            NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    else:
        print("Model folder not found using given MODEL_PATH; will try to extract from configured path.")

    # ensure npz exists: if not, extract from saved model checkpoint
    if not NPZ_PATH.exists():
        print(".npz not found — attempting to extract from saved model folder...")
        model_folder = found_folder or MODEL_PATH
        try:
            weights = load_weights_from_model(model_folder)
            print(f"Extracted {len(weights)} arrays. Saving to NPZ...")
            save_weights_npz_dict(weights, NPZ_PATH)
        except Exception as e:
            print(f"Failed to extract weights: {e}")
            log_line(f"EXTRACT FAIL {e}")
            raise SystemExit(1)
    else:
        print(f"Found NPZ: {NPZ_PATH} ({print_size(NPZ_PATH)})")

    ok = upload_with_fallback(NPZ_PATH)
    if ok:
        print("Upload succeeded.")
    else:
        print("Upload failed — check upload_log.txt and server endpoint.")
