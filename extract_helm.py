import requests
import pandas as pd 
from datetime import date 
from io import StringIO

def extract_helm():

    print("Downloading LLM benchmark data from Github.")

    url = "https://raw.githubusercontent.com/salttechno/LLM-Model-Comparison-2026/main/data/llm-model-comparison-2026.csv"
    response = requests.get(url,timeout=30)

    if response.status_code != 200:
        print(f"Error: {response.statuse_code}")
        return None
    

    df_raw = pd.read_csv(StringIO(response.text))

    print(f"Downloaded. Shape: {df_raw.shape}")
    print(f"Columns: {list(df_raw.columns)}")


    SCORE_COLUMNS = {
        "mmlu_score":      "mmlu",
        "humaneval_score": "humaneval",
        "math_score":      "math",
        "mt_bench_score":  "mt_bench"
    }


    all_rows = []

    for index, row in df_raw.iterrows():

        # ── get model name ────────────────────────────────────────────────────
        model_name = str(row.get("model", "")).lower().strip()
        if not model_name or model_name == "nan":
            continue

        # ── get organization ──────────────────────────────────────────────────
        organization = str(row.get("provider", "unknown")).lower().strip()
        
        # ── get parameter count ───────────────────────────────────────────────
        param_count = row.get("parameters_billions", None)
        

        # ── get training cutoff ───────────────────────────────────────────────
        training_cutoff = str(row.get("training_cutoff", "")).strip()
        # example: "Jun 2

        for raw_col, clean_name in SCORE_COLUMNS.items():

            score = row.get(raw_col, None)

            if score is None or str(score) == "nan":
                continue
            # no score for this benchmark → skip

            try:
                score = float(score)
            except (ValueError, TypeError):
                continue

            
            if score > 1.0:
                score = score / 100

            if score < 0 or score > 1:
                continue

            score = round(score, 4)

            all_rows.append({
                "model_name":      model_name,
                # e.g. "gpt-4.1" → dim_models.model_name

                "organization":    organization,
                # e.g. "openai" → dim_models.organization

                "architecture":    "unknown",
                # not provided in this dataset

                "model_type":      "unknown",

                "param_count_b":   param_count,
                # e.g. 7.0 → dim_models.param_count_b

                "training_cutoff": training_cutoff,
                # e.g. "Jun 2024" → useful for dim_models

                "flagged_by_hf":   False,
                # not applicable for this source

                "benchmark_name":  clean_name,
                # e.g. "mmlu", "humaneval", "math", "mt_bench"
                # → dim_benchmarks.benchmark_name

                "task_type":       "unknown",
                "language":        "english",

                "score":           score,
                # e.g. 0.865 → fact_scores.score

                "fetch_date":      str(date.today()),
                "source":          "GitHub-LLM-Comparison"
                # → dim_sources.source_name
            })

    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["score"])
    df = df[(df["score"] >= 0) & (df["score"] <= 1)]
    df = df.drop_duplicates()
    df.to_csv("raw_helm.csv", index=False)

    print(f"Done. Saved {len(df)} rows to raw_helm.csv")
    print(f"Unique models:     {df['model_name'].nunique()}")
    print(f"Unique benchmarks: {df['benchmark_name'].nunique()}")
    print(f"Score range:       {df['score'].min():.3f} to {df['score'].max():.3f}")

    return df

if __name__ == "__main__":
    df = extract_helm()
    if df is not None:
        print("\nSample output:")
        print(df.head(8).to_string())