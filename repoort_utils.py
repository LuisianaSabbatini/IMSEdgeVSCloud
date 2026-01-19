import pandas as pd
import numpy as np

def log_metrics(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, mode="a", header=not pd.io.common.file_exists(filename), index=False)

    # Calcolo metriche aggregate
    latency_mean = df["latency_s"].mean()
    energy_mean = df["energy_mJ"].mean()
    cpu_mean = df["cpu_%"].mean()
    mem_mean = df["mem_%"].mean()

    print(f"[REPORT] Saved {len(df)} records to {filename}")
    print(f"   → Avg latency={latency_mean:.3f}s | Avg energy={energy_mean:.2f}mJ | "
          f"CPU={cpu_mean:.1f}% | MEM={mem_mean:.1f}%")
