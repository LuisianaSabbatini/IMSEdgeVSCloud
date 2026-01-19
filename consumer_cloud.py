"""
Cloud-side Kafka consumer simulator for predictive-maintenance experiment.

- ML: RF or XGBoost on extracted features
- DL: Keras CNN (fallback)
- Simulates realistic latency profiles
- Logs metrics: latency, cpu, mem, accuracy, timestamp
- Optional CodeCarbon tracking
"""

# python consumer_cloud.py --mode ml --scaler models/scaler.pkl --ml-model models/rf_model.pkl --out-csv cloud_results_ml_rf.csv --track-emissions --emissions-file cloud_emissions_ml_rf.csv
# python consumer_cloud.py --mode ml --scaler models/scaler.pkl --ml-model models/xgb_model.pkl --out-csv cloud_results_ml_xgb.csv --track-emissions --emissions-file cloud_emissions_ml_xgb.csv
# python consumer_cloud.py --mode dl --scaler models/raw_scaling_params.pkl --out-csv cloud_results_dl.csv --track-emissions --emissions-file cloud_emissions_dl.csv

import argparse, json, time, os
import numpy as np
import pickle
import pandas as pd
import psutil
import joblib
from featuresExtraction import *
from tensorflow.keras.models import load_model
from collections import deque

# confluent_kafka imports
try:
    from confluent_kafka import Consumer, Producer
    KAFKA_AVAILABLE = True
except Exception:
    KAFKA_AVAILABLE = False

# -------- Keras fallback --------
KERAS_AVAILABLE = False
try:
    import tensorflow as tf  # noqa: F401
    from tensorflow.keras.models import load_model
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False

# -------- CodeCarbon --------
CODECARBON_AVAILABLE = False
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except Exception as e:
    print(f"[WARN] CodeCarbon import failed: {e}")
    CODECARBON_AVAILABLE = False


# -------- Latency simulation --------
LATENCY_PROFILES = {
    "low": (0.02, 0.05),
    "medium": (0.08, 0.12),
    "high": (0.15, 0.25)
}


def simulate_latency(profile):
    low, high = LATENCY_PROFILES[profile]
    time.sleep(np.random.uniform(low, high))

# -------- ML loader --------
def load_ml_pipeline(scaler_path, model_path):
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ML model not found: {model_path}")
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    print(f"[CLOUD] ML pipeline loaded: scaler={scaler_path}, model={model_path}")
    return scaler, model

# -------- DL loader --------
def load_dl_pipeline(keras_path):
    cnn = load_model(keras_path, compile=False)
    print(f"[CLOUD] DL CNN loaded from {keras_path}")
    return cnn


def main(args):
    print(f"\n=== Avvio consumer cloud in modalità {args.mode} ===")

    if not KAFKA_AVAILABLE:
        print("[CLOUD] Warning: confluent_kafka non disponibile. Serve Kafka per lo streaming.")

    consumer, producer = None, None
    if KAFKA_AVAILABLE:
        consumer = Consumer({
            'bootstrap.servers': args.bootstrap,
            'group.id': 'cloud-group',
            'auto.offset.reset': 'latest'
        })
        consumer.subscribe([args.topic])
        producer = Producer({'bootstrap.servers': args.bootstrap})

    # Label encoder
    label_encoder = None
    if os.path.exists(args.label_encoder):
        try:
            label_encoder = joblib.load(args.label_encoder)
            print(f"[CLOUD] Loaded label encoder: {args.label_encoder}")
        except Exception as e:
            print("[CLOUD] Failed loading label encoder:", e)

    # Pipelines
    ml_scaler, ml_model, dl_model = None, None, None
    dl_scaler = None  # dict con mean/std per x e y

    if args.mode == "ml":
        ml_scaler, ml_model = load_ml_pipeline(args.scaler, args.ml_model)
    else:
        dl_model = load_dl_pipeline(keras_path = args.keras_model)
        if os.path.exists(args.scaler):
            try:
                with open(args.scaler, "rb") as f:
                    dl_scaler = pickle.load(f)
                print(f"[CLOUD] Loaded DL scaling params: {args.scaler}")
            except Exception as e:
                print("[CLOUD] Failed to load DL scaling params:", e)

    # Emissions tracker
    tracker = None
    if CODECARBON_AVAILABLE and args.track_emissions:
        try:
            tracker = EmissionsTracker(save_to_file=True, measure_power_secs=15, output_file=args.emissions_file)
            tracker.start()
            print("[CLOUD] EmissionsTracker started.")
        except Exception as e:
            print("[CLOUD] EmissionsTracker could not be started:", e)
    print("[CLOUD] Consumer ready. Mode:", args.mode)

    # --- Process object for per-process CPU/mem ---
    proc = psutil.Process(os.getpid())
    logs, total, correct = [], 0, 0

    idle_count = 0
    max_idle = 30

    try:
        while True:
            msg = consumer.poll(timeout=1.0) if KAFKA_AVAILABLE else None
            if msg is None:
                idle_count += 1
                if idle_count >= max_idle:
                    print("[CLOUD] Nessun messaggio ricevuto per troppo tempo. Chiudo consumer.")
                    break
                continue
            idle_count = 0

            if KAFKA_AVAILABLE:
                if msg.error():
                    print("[CLOUD] Kafka error:", msg.error())
                    continue

                try:
                    data = json.loads(msg.value().decode("utf-8"))
                except Exception:
                    data = msg.value()
                    if isinstance(data, bytes):
                        data = json.loads(data.decode("utf-8"))
            else:
                print("[CLOUD] No Kafka available. Exiting.")
                break

            frame_id = data.get("id", None)
            true_label = data.get("label", None)
            x = np.array(data.get("x", []))
            y = np.array(data.get("y", []))
            if x.size == 0 or y.size == 0:
                print("[CLOUD] Warning: empty signal received, skipping frame", frame_id)
                continue

            start_time = time.time()
            simulate_latency(args.latency_profile)

            # Inference
            if args.mode == "ml":
                fx = extract_all_features(x, fs=args.fs)
                fy = extract_all_features(y, fs=args.fs)
                if fx is None or fy is None or len(fx) == 0 or len(fy) == 0:
                    print(f"[CLOUD] Warning: feature extraction failed, skipping frame {frame_id}")
                    continue

                # --- Trasformazione in array ordinato ---
                fx_arr = np.array([v for k, v in (fx.items())], dtype=np.float32)
                fy_arr = np.array([v for k, v in (fy.items())], dtype=np.float32)

                feat_vec = np.concatenate([fx_arr, fy_arr]).reshape(1, -1)
                feat_scaled = ml_scaler.transform(feat_vec)
                pred_num = ml_model.predict(feat_scaled)[0]
                pred_out = (
                    label_encoder.inverse_transform([int(pred_num)])[0]
                    if label_encoder is not None else str(pred_num)
                )
            else:
                sig = np.stack([x, y], axis=-1).reshape((1, len(x), 2))
                if dl_scaler is not None:
                    sig_x = (sig[0, :, 0] - dl_scaler["global_mean_x"]) / (dl_scaler["global_std_x"] + 1e-8)
                    sig_y = (sig[0, :, 1] - dl_scaler["global_mean_y"]) / (dl_scaler["global_std_y"] + 1e-8)
                    sig = np.stack([sig_x, sig_y], axis=-1).reshape((1, len(sig_x), 2))
                else:
                    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

                out = dl_model.predict(sig, verbose=0)
                pred_num = int(np.argmax(out, axis=1)[0])
                pred_out = (
                    label_encoder.inverse_transform([pred_num])[0]
                    if label_encoder is not None else str(pred_num)
                )

            latency_ms = (time.time() - start_time) * 1000.0

            # --- CPU/memoria processo ---
            cpu = proc.cpu_percent(interval=None)
            mem = proc.memory_percent()

            total += 1
            if true_label is not None and str(pred_out) == str(true_label):
                correct += 1
            acc = correct / total if total > 0 else 0.0

            row = {
                "id": frame_id,
                "pred": pred_out,
                "true": true_label,
                "latency_ms": latency_ms,
                "cpu_percent": cpu,
                "mem_percent": mem,
                "accuracy_cumulative": acc,
                "timestamp": time.time(),
                "mode": args.mode,
                "profile": "cloud"
            }
            logs.append(row)

            if args.produce_results and KAFKA_AVAILABLE:
                try:
                    producer.produce(args.result_topic, json.dumps(row).encode("utf-8"))
                except Exception as e:
                    print("[CLOUD] Error producing result:", e)
                if total % 50 == 0 and KAFKA_AVAILABLE:
                    producer.flush()

            if len(logs) >= args.flush_every:
                df = pd.DataFrame(logs)
                header = not os.path.exists(args.out_csv)
                df.to_csv(args.out_csv, index=False, mode='a' if not header else 'w', header=header)
                logs = []

            print(f"[CLOUD] Frame {frame_id} | pred {pred_out} | true {true_label} | "
                  f"lat {latency_ms:.1f} ms | cpu {cpu}% | mem {mem}% | acc {acc:.3f}")

    except KeyboardInterrupt:
        print("[CLOUD] Interrupted by user.")
    finally:
        if logs:
            df = pd.DataFrame(logs)
            header = not os.path.exists(args.out_csv)
            df.to_csv(args.out_csv, index=False, mode='a' if not header else 'w', header=header)
        if KAFKA_AVAILABLE and consumer is not None:
            consumer.close()
        if tracker is not None:
            try:
                tracker.stop()
            except Exception:
                pass
        print("[CLOUD] Terminated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ml", "dl"], required=True)
    parser.add_argument("--latency-profile", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default="sensor-data")
    parser.add_argument("--produce-results", action="store_true")  # ADDED
    parser.add_argument("--result-topic", default="cloud-results")
    parser.add_argument("--scaler", default="models/scaler.pkl", help="Path scaler per ML o DL")  # CHANGED
    parser.add_argument("--ml-model", default="models/rf_model.pkl")
    parser.add_argument("--keras-model", dest="keras_model", default="models/cnn1d_raw.keras")  # CHANGED
    parser.add_argument("--label-encoder", dest="label_encoder", default="models/label_encoder.pkl")  # ADDED
    parser.add_argument("--fs", type=int, default=20000)
    parser.add_argument("--out-csv", dest="out_csv", default="cloud_results.csv")
    parser.add_argument("--flush-every", type=int, default=20)  # ADDED
    parser.add_argument("--track-emissions", action="store_true")
    parser.add_argument("--emissions-file", default="cloud_emissions.csv")
    args = parser.parse_args()
    main(args)
