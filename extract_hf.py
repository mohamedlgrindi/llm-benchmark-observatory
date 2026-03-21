import os 
import pandas as pd
from huggingface_hub import snapshot_download
from datetime import date




BENCHMARK_RAW_COLUMNS = {
    "IFEval Raw":     "ifeval",
    "BBH Raw":        "bbh",
    "MATH Lvl 5 Raw": "math_lvl5",
    "GPQA Raw":       "gpqa",
    "MUSR Raw":       "musr",
    "MMLU-PRO Raw":   "mmlu_pro"
}


def extract_huggingface():

    print("=" * 55)
    print("HuggingFace Open LLM Leaderboard v2 — Extraction")
    print("=" * 55)


    local_path = snapshot_download(repo_id="open-llm-leaderboard/contents", repo_type="dataset", local_dir="./hf_data")

    print(f"Downloaded to: {local_path}")

    print("\n Finding Parquet files")


    parquet_files = []

    for root, dirs ,files in os.walk(local_path):

        for filename in files:
            if filename.endswith(".parquet"):
                parquet_files.append(os.path.join(root,filename))
    
    print(f"Found {len(parquet_files)} file(s)")
    
    if len(parquet_files) == 0:
        print("Error: No parquet files found.")
        return None

    print("\n Reading files...")

    chunks = []

    for filepath in parquet_files:
        try:
            chunk = pd.read_parquet(filepath)
            chunks.append(chunk)
            print(f"  Read: {os.path.basename(filepath)} — {len(chunk)} rows")
        except Exception as e :
            print(f"Skipped:{filepath}-{e}")
            continue

    if len(chunks) == 0:
        print("ERROR: Could not read any files.")
        return None

    df_raw = pd.concat(chunks,ignore_index= True)

    print(f"\nTotal rows loaded: {len(df_raw)}")
    print(f"Columns found: {list(df_raw.columns)}")

    available = {
      col : clear_name
      for col,clear_name in BENCHMARK_RAW_COLUMNS.items() 
      if col in df_raw.columns
      }
     
    missing = [col for col in BENCHMARK_RAW_COLUMNS if col not in df_raw.columns]

    print(f"  Found:   {list(available.keys())}")
    print(f"  Missing: {missing}")
    
    if len(available) == 0:
        print("Error: No benchmark columns found.")
        return None


    print(" Building output rows")

    all_rows = []

    for index,row in df_raw.iterrows():

        model_name = str(row.get("fullname","").lower().strip())
        if not model_name or model_name == "nan":
            continue
        
        if "/" in model_name:
            organization = model_name.split("/")[0]
        elif "-" in model_name:
            organization = model_name.split("-")[0]
        else: 
            organization = "unknown"

        architecture =str(row.get("Architecture","unknown").strip())

        model_type = str(row.get("Type", "unknown").strip())

        param_count = row.get("#Params (B)", None)

        flagged = bool(row.get("Flagged", False))

        for raw_col, clean_name in available.items():

            score = row.get(raw_col,None)

            if score is None or str(score)== "nan":
                continue
            
            try:
                score = float(score)
            except (ValueError, TypeError):
                continue 
            
            if score > 1.0:
                score = score / 100
            
            score = round(score, 4)

            all_rows.append({
                # ── Goes to dim_models ────────────────────────────────────────
                "model_name":     model_name,
                # example: "0-hero/matter-0.2-7b-dpo"

                "organization":   organization,
                # example: "0-hero"

                "architecture":   architecture,
                # example: "MistralForCausalLM"

                "model_type":     model_type,
                # example: "chat models (RLHF, DPO, IFT, ...)"

                "param_count_b":  param_count,
                # example: 7.242

                # ── Goes to dim_benchmarks ────────────────────────────────────
                "benchmark_name": clean_name,
                # example: "ifeval"

                "task_type":      "unknown",
                # HF does not provide this — you fill manually later

                "language":       "english",
                # all v2 benchmarks are English

                # ── Goes to fact_scores ───────────────────────────────────────
                "score":          score,
                # example: 0.3303

                "flagged_by_hf":  flagged,
                # example: False

                # ── Goes to dim_dates ─────────────────────────────────────────
                "fetch_date":     str(date.today()),
                # example: "2026-03-19"

                # ── Goes to dim_sources ───────────────────────────────────────
                "source":         "HuggingFace"
            })

    print(f"Built {len(all_rows)} rows total")        

    print("\n Cleaning...")

    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["score"])
    df = df[(df["score"] >= 0) & (df["score"] <= 1)]
    df = df.drop_duplicates()



    print("\n Saving...")

    df.to_csv("raw_hf.csv", index=False)

    print(f"Done. Saved {len(df)} rows to raw_hf.csv")
    print("This file goes to Talaxie in Step 3.")

    return df


if __name__ == "__main__":
    df = extract_huggingface()

    if df is not None:
        print("\n" + "=" * 55)
        print("SAMPLE — first 6 rows (one model, all benchmarks):")
        print("=" * 55)
        print(df.head(6).to_string())