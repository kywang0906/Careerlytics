import pandas as pd
import glob
import os

COMPANY = 'google'
ROLE = 'pm'

input_dir = f"dataset/{COMPANY}/all_raw"
file_paths = sorted(glob.glob(os.path.join(input_dir, "*.csv")))

dfs = [pd.read_csv(fp) for fp in file_paths]
merged = pd.concat(dfs, ignore_index=True)

# Remove duplcate IDs
deduped = merged.drop_duplicates(subset=["id"], keep="first")

# Saved to CSV
output_path = f"dataset/{COMPANY}/merged_{ROLE}.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
deduped.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"{len(file_paths)} files merged. Original data: {len(merged)}. After duplicated removed: {len(deduped)}")
print(f"Results saved to {output_path}")
