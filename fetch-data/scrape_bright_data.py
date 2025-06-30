import requests
import time
import json
import pandas as pd

API_KEY      = "MY_API_KEY"
DATASET_ID   = "TARGET_DATASET_ID"
TRIGGER_URL  = "https://api.brightdata.com/datasets/v3/trigger"
PROGRESS_URL = "https://api.brightdata.com/datasets/v3/progress/{snapshot_id}"
DOWNLOAD_URL = "https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}"

COMPANY = "google"
ROLE    = "pm"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

columns_to_keep = [
    'id', 'name', 'country_code', 'position', 'about', 'experience',
    'url', 'education', 'courses', 'certifications',
    'current_company_company_id', 'current_company_name',
    'publications', 'projects'
]
custom_fields = "|".join(columns_to_keep)
output_file = f"dataset/{COMPANY}/{COMPANY}_{ROLE}_profiles_filtered.csv"

# 1) Load URL list
with open(f"dataset/{COMPANY}/{COMPANY}_{ROLE}_urls.txt", encoding="utf-8") as f:
    urls = [u.strip() for u in f if u.strip()]

# 2) Set batch size and get snapshot_ids
snapshot_ids = []
batch_size = 200

for idx in range(0, len(urls), batch_size):
    batch_urls = urls[idx : idx + batch_size]
    resp = requests.post(
        TRIGGER_URL,
        headers=HEADERS,
        params={
            "dataset_id": DATASET_ID,
            "format": "json",
            "custom_output_fields": custom_fields
        },
        json=[{"url": u} for u in batch_urls]
    )
    resp.raise_for_status()
    sid = resp.json()["snapshot_id"]
    snapshot_ids.append(sid)
    print(f"Batch: {idx//batch_size+1}. snapshot_id = {sid}")
    time.sleep(1)

# 3) Get results
all_profiles = []
list_fields = ['experience','education','courses','certifications','publications','projects']

for sid in snapshot_ids:
    # Polling until ready
    while True:
        p = requests.get(PROGRESS_URL.format(snapshot_id=sid), headers=HEADERS)
        p.raise_for_status()
        status = p.json().get("status")
        print(f"Snapshot {sid} status: {status}")
        if status == "ready":
            break
        if status in ("failed", "error"):
            raise RuntimeError(f"Snapshot {sid} failed: {status}")
        time.sleep(5)

    # Download
    d = requests.get(DOWNLOAD_URL.format(snapshot_id=sid), headers=HEADERS)
    d.raise_for_status()
    data = d.json()
    print(f"Snapshot {sid} has downloaded {len(data)} data.")
    all_profiles.extend(data)
    time.sleep(1)

# 4) Flatten and filter columns
df = pd.json_normalize(all_profiles)
df = df[[c for c in columns_to_keep if c in df.columns]]

# 5) Convert list of columns into JSON 
for f in list_fields:
    if f in df.columns:
        df[f] = df[f].apply(lambda x: json.dumps(x, ensure_ascii=False))

# 6) Save results to CSV file 
df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"File saved to {output_file}")

