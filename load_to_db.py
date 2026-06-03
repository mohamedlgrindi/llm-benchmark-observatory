import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

# ── SECTION 1: CONNECT ────────────────────────────────────────────────────────

engine = create_engine(
    "postgresql://postgres:Mohamedkali@localhost:5432/benchmark_dw"
)
# replace YourPassword with your actual pgAdmin password
print("Connected to PostgreSQL")

# ── SECTION 2: READ BOTH CSV FILES ───────────────────────────────────────────

df_hf   = pd.read_csv("raw_hf.csv")
df_helm = pd.read_csv("raw_helm.csv")
df_all  = pd.concat([df_hf, df_helm], ignore_index=True)
print(f"Total rows loaded: {len(df_all)}")

# ── SECTION 3: LOAD dim_models ────────────────────────────────────────────────

dim_models = df_all[[
    "model_name", "organization", "architecture", "param_count_b"
]].drop_duplicates(subset=["model_name"])

#dim_models["param_count_b"] = pd.to_numeric(dim_models["param_count_b"], errors="coerce")

#dim_models.to_sql("dim_models", engine, if_exists="append", index=False, method="multi")

dim_models = dim_models.drop_duplicates(subset=["model_name"])
dim_models["param_count_b"] = pd.to_numeric(dim_models["param_count_b"], errors="coerce")

for _, row in dim_models.iterrows():
    try:
        pd.DataFrame([row]).to_sql("dim_models", engine, if_exists="append", index=False)
    except Exception:
        pass

print(f"dim_models: inserted rows")

print(f"dim_models: inserted {len(dim_models)} rows")

# ── SECTION 4: LOAD dim_dates ─────────────────────────────────────────────────

unique_dates = df_all["fetch_date"].unique()
date_rows = []

for d in unique_dates:
    dt = datetime.strptime(str(d), "%Y-%m-%d")
    date_rows.append({
        "full_date": d,
        "year":      dt.year,
        "month":     dt.month,
        "quarter":   (dt.month - 1) // 3 + 1,
        "week":      dt.isocalendar()[1]
    })

dim_dates = pd.DataFrame(date_rows)
dim_dates.to_sql("dim_dates", engine, if_exists="append", index=False)
print(f"dim_dates: inserted {len(dim_dates)} rows")

# ── SECTION 5: LOAD fact_scores ───────────────────────────────────────────────

with engine.connect() as conn:
    models_df     = pd.read_sql("SELECT model_id, model_name FROM dim_models", conn)
    benchmarks_df = pd.read_sql("SELECT benchmark_id, benchmark_name FROM dim_benchmarks", conn)
    dates_df      = pd.read_sql("SELECT date_id, CAST(full_date AS TEXT) as full_date FROM dim_dates", conn)
    sources_df    = pd.read_sql("SELECT source_id, source_name FROM dim_sources", conn)

df_facts = df_all.copy()
df_facts = df_facts.merge(models_df,     on="model_name",      how="left")
df_facts = df_facts.merge(benchmarks_df, on="benchmark_name",  how="left")
df_facts = df_facts.merge(dates_df,      left_on="fetch_date", right_on="full_date", how="left")
df_facts = df_facts.merge(sources_df,    left_on="source",     right_on="source_name", how="left")

fact_scores = df_facts[[
    "model_id", "benchmark_id", "date_id", "source_id", "score", "flagged_by_hf"
]].dropna(subset=["model_id", "benchmark_id", "score"])

fact_scores["suspicion_score"] = None

fact_scores.to_sql("fact_scores", engine, if_exists="append", index=False, method="multi")
print(f"fact_scores: inserted {len(fact_scores)} rows")
print("\nDone. All tables loaded into PostgreSQL.")