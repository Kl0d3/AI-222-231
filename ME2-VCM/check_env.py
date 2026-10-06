import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("GPU: none (CPU only)")
HF_DIR = "/home/kent.justin.canja/sandbox/hf_ai231_me2"
MARK_DIR = ("/home/kent.justin.canja/sandbox/"
            "ME2_ Voice Controlled Smart Device-20260925T153305Z-1-001/"
            "ME2_ Voice Controlled Smart Device/vcm_dataset")
REAL_DIR = "/home/kent.justin.canja/sandbox/AI-222-231/ME2-VCM/real_data"
hf_parquets = glob.glob(os.path.join(HF_DIR, "data", "train-*.parquet")) + glob.glob(os.path.join(HF_DIR, "train-*.parquet"))
print(f"\nHF train parquets: {len(hf_parquets)}")
mark_meta = os.path.join(MARK_DIR, "metadata.csv")
print(f"Mark metadata exists: {os.path.exists(mark_meta)}")
real_meta = os.path.join(REAL_DIR, "real_metadata.csv")
print(f"Real metadata exists: {os.path.exists(real_meta)}")
if os.path.exists(real_meta):
    import pandas as pd
    df = pd.read_csv(real_meta)
    print(f"Real clips: {len(df)}")
    print(f"Real intents: {sorted(df['intent'].unique())}")
real_wavs = glob.glob(os.path.join(REAL_DIR, "*.wav"))
print(f"Real WAV files: {len(real_wavs)}")
