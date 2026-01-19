import json
import time
import datetime
import argparse   # >>> AGGIUNTO
import pandas as pd
from confluent_kafka import Producer

# ASSICURARSI CHE KAFKA DOCKER SIA ATTIVO

# poi lanciare sul terminale PyCharm:
# python producer_sensor.py

# ======================
# Config
# ======================
conf = {
    "bootstrap.servers": "localhost:9092",
    "client.id": "sensor-simulator",
    "acks": "all"   # >>> AGGIUNTO: maggiore affidabilità
}

producer = Producer(conf)

# >>> AGGIUNTO: argomento per controllare velocità simulazione
parser = argparse.ArgumentParser()
parser.add_argument("--speedup", type=float, default=1.0,
                    help="Fattore di velocità simulazione (es. 10 = 10x più veloce)")
args = parser.parse_args()

# ======================
# Caricamento dataset
# ======================
df_raw = pd.read_parquet("ims_1test_raw.parquet")

# ======================
# Callback consegna
# ======================
def delivery_report(err, msg):
    if err is not None:
        print("Delivery failed: {}".format(err))
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")

# ======================
# Produzione messaggi
# ======================
for idx, row in df_raw.iterrows():
    message = {
        "bearing_id": row["bearing_id"],
        "filename": row["filepath"],
        "label": row["label"],
        "x": row["x"].tolist(),   # >>> AGGIUNTO: serializzazione robusta
        "y": row["y"].tolist()    # >>> AGGIUNTO
    }

    producer.produce(
        "sensor-data",
        key=str(row["bearing_id"]),
        value=json.dumps(message),
        callback=delivery_report
    )

    # Flush ogni 100 messaggi
    if idx % 100 == 0:
        producer.flush()

    # >>> AGGIUNTO: simulazione tempo reale (1s per riga / speedup)
    time.sleep(1.0 / args.speedup)

producer.flush()
print(f"Finished producing all sensor data: {datetime.datetime.now()}.")
