import pandas as pd, os
MARK_DIR = "/home/kent.justin.canja/sandbox/ME2_ Voice Controlled Smart Device-20260925T153305Z-1-001/ME2_ Voice Controlled Smart Device/vcm_dataset"
meta = pd.read_csv(os.path.join(MARK_DIR, "metadata.csv"))
total_found = sum(1 for _, r in meta.iterrows() if os.path.isfile(os.path.join(MARK_DIR, r["file_name"])))
print(f"Mark synthetic: {total_found}/{len(meta)}")
# Real
real_meta = pd.read_csv("/home/kent.justin.canja/sandbox/AI-222-231/ME2-VCM/real_data/real_metadata.csv")
real_found = sum(1 for _, r in real_meta.iterrows() if os.path.isfile(os.path.join("/home/kent.justin.canja/sandbox/AI-222-231/ME2-VCM/real_data", r["file_name"])))
print(f"Mark real: {real_found}/{len(real_meta)}")
print(f"Total Mark: {total_found + real_found}")
