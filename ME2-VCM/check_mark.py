import pandas as pd, os
MARK_DIR = "/home/kent.justin.canja/sandbox/ME2_ Voice Controlled Smart Device-20260925T153305Z-1-001/ME2_ Voice Controlled Smart Device/vcm_dataset"
meta = pd.read_csv(os.path.join(MARK_DIR, "metadata.csv"))
found = 0
for i in range(min(5, len(meta))):
    fn = meta.iloc[i]["file_name"]
    p = os.path.join(MARK_DIR, "audio", fn)
    print(f"{fn} -> exists={os.path.isfile(p)}")
    if os.path.isfile(p):
        found += 1
print(f"Sample found: {found}/5")
# Count total
total_found = sum(1 for _, r in meta.iterrows() if os.path.isfile(os.path.join(MARK_DIR, "audio", r["file_name"])))
print(f"Total found: {total_found}/{len(meta)}")
