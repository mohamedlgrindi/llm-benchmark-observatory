import pandas as pd

for filename in ["raw_hf.csv", "raw_helm.csv"]:
    print(f"\n{'='*40}")
    try:
        df = pd.read_csv(filename)
        print(f" {filename}")
        print(f"   Rows:              {len(df)}")
        print(f"   Unique models:     {df['model_name'].nunique()}")
        print(f"   Unique benchmarks: {df['benchmark_name'].nunique()}")
        print(f"   Score range:       {df['score'].min():.3f} to {df['score'].max():.3f}")
    except FileNotFoundError:
        print(f" {filename} — not found yet")