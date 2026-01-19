# ================================================
# 🚀 IMS Bearing NASA - Baseline ML & CNN 1D
# ================================================
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime

# ---------- Funzioni di utilità ----------
def parse_timestamp(ts_str: str) -> datetime:
    """Converte timestamp tipo 2003.10.22.12.06.24 in datetime"""
    return datetime.strptime(ts_str, "%Y.%m.%d.%H.%M.%S")

def assign_label(ts: datetime, intervals: list) -> str:
    """Ritorna la label corrispondente al timestamp in base agli intervalli"""
    for start, end, label in intervals:
        if start <= ts <= end:
            return label
    return "unknown"



# ---------- Funzione principale ----------
def load_ims_data(base_path, out_path, batch_size=50):
    base_path = Path(base_path)
    batch = []
    batch_count = 0
    total_files = sum(len(files) for _, _, files in os.walk(base_path))
    processed_files = 0
    print(">>> Inizio il loading...")

    # rimuovo file di output se già esiste
    if Path(out_path).exists():
        Path(out_path).unlink()

    writer = None  # writer parquet
    for root, _, files in os.walk(base_path):
        for fname in files:
            file_path = Path(root) / fname

            ts_str = file_path.name
            try:
                ts = parse_timestamp(ts_str)
            except ValueError:
                print(f"[WARNING] Skip file {file_path.name}, non è un timestamp valido")
                continue

            # Leggo il file (8 colonne, nessun header)
            data = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python").values

            # 4 cuscinetti, 2 colonne ciascuno
            for bearing_id in range(4):
                col_x = 2 * bearing_id
                col_y = col_x + 1
                x = data[:, col_x]
                y = data[:, col_y]

                bearing_key = f"b{bearing_id + 1}"
                batch.append({
                    "bearing_id": bearing_id + 1,
                    "x": x,
                    "y": y,
                    "filepath": str(file_path),
                    "label": assign_label(ts, intervals[bearing_key])
                })

            processed_files += 1

            # Scrittura batch a chunk
            if len(batch) >= batch_size:
                df_batch = pd.DataFrame(batch)
                table = pa.Table.from_pandas(df_batch)

                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")

                writer.write_table(table)
                batch_count += 1
                print(f"[INFO] Salvato batch {batch_count}, file processati: {processed_files}/{total_files}")
                batch = []

    # Scrittura finale se rimane qualcosa
    if batch:
        df_batch = pd.DataFrame(batch)

        # sanity checks
        assert df_batch['x'].apply(len).min() > 0, "Ci sono segnali x vuoti"
        assert df_batch['y'].apply(len).min() > 0, "Ci sono segnali y vuoti"
        assert (df_batch['x'].apply(len) == df_batch['y'].apply(len)).all(), "Lunghezze x/y non corrispondono"

        table = pa.Table.from_pandas(df_batch)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)
        batch_count += 1
        print(f"[INFO] Salvato batch finale {batch_count}, file processati: {processed_files}/{total_files}")

    if writer:
        writer.close()

    print("[DONE] Lettura completata.")
    return out_path


# ---------- Definizione intervalli ----------
intervals = {
    "b1": [
        (parse_timestamp("2003.10.22.12.06.24"), parse_timestamp("2003.10.23.09.14.13"), "early"),
        (parse_timestamp("2003.10.23.09.24.13"), parse_timestamp("2003.11.08.12.11.44"), "suspect"),
        (parse_timestamp("2003.11.08.12.21.44"), parse_timestamp("2003.11.19.21.06.07"), "normal"),
        (parse_timestamp("2003.11.19.21.16.07"), parse_timestamp("2003.11.24.20.47.32"), "suspect"),
        (parse_timestamp("2003.11.24.20.57.32"), parse_timestamp("2003.11.25.23.39.56"), "imminent failure"),
    ],
    "b2": [
        (parse_timestamp("2003.10.22.12.06.24"), parse_timestamp("2003.11.01.21.41.44"), "early"),
        (parse_timestamp("2003.11.01.21.51.44"), parse_timestamp("2003.11.24.01.01.24"), "normal"),
        (parse_timestamp("2003.11.24.01.11.24"), parse_timestamp("2003.11.25.10.47.32"), "suspect"),
        (parse_timestamp("2003.11.25.10.57.32"), parse_timestamp("2003.11.25.23.39.56"), "imminent failure"),
    ],
    "b3": [
        (parse_timestamp("2003.10.22.12.06.24"), parse_timestamp("2003.11.01.21.41.44"), "early"),
        (parse_timestamp("2003.11.01.21.51.44"), parse_timestamp("2003.11.22.09.16.56"), "normal"),
        (parse_timestamp("2003.11.22.09.26.56"), parse_timestamp("2003.11.25.10.47.32"), "suspect"),
        (parse_timestamp("2003.11.25.10.57.32"), parse_timestamp("2003.11.25.23.39.56"), "inner race failure"),
    ],
    "b4": [
        (parse_timestamp("2003.10.22.12.06.24"), parse_timestamp("2003.10.29.21.39.46"), "early"),
        (parse_timestamp("2003.10.29.21.49.46"), parse_timestamp("2003.11.15.05.08.46"), "normal"),
        (parse_timestamp("2003.11.15.05.18.46"), parse_timestamp("2003.11.18.19.12.30"), "suspect"),
        (parse_timestamp("2003.11.19.09.06.09"), parse_timestamp("2003.11.22.17.36.56"), "rolling element failure"),
        (parse_timestamp("2003.11.22.17.46.56"), parse_timestamp("2003.11.25.23.39.56"), "stage 2 failure"),
    ],
}

print("intervals impostati")

# ---------- Avvio ----------
if __name__ == "__main__":
    out_file = ("/Users/luisianasabbatini/Library/CloudStorage/OneDrive-UniversitàPolitecnicadelleMarche/WORK/UNIVPM/"
                "Univpm_Pubblicazioni/WIP_2025_SI_sustainability/IMSbearing_edge_cloud/pythonProject/ims_1test_raw.parquet")
    load_ims_data("/Users/luisianasabbatini/Library/CloudStorage/OneDrive-UniversitàPolitecnicadelleMarche/"
                  "WORK/UNIVPM/Univpm_Pubblicazioni/WIP_2025_SI_sustainability/IMSbearing_edge_cloud/pythonProject/"
                  "IMS/1st_test", out_file, batch_size=50)
