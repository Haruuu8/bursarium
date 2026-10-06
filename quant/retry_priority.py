"""Retry khusus untuk saham prioritas dengan mode ultra pelan."""
import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant.fundamental_detail import fetch_detail, OUTPUT_DIR

# Baca daftar prioritas
with open("retry_priority.txt", encoding="utf-8-sig") as f:
    tickers = [l.strip() for l in f if l.strip()]

print(f"Retry {len(tickers)} saham prioritas (super pelan)...")
print("Tunggu 30 detik dulu untuk reset crumb...")
time.sleep(30)

os.makedirs(OUTPUT_DIR, exist_ok=True)
berhasil, gagal = [], []

for kode in tickers:
    print(f"\n[{kode}] Fetching...")
    result = fetch_detail(kode)
    if result:
        path = os.path.join(OUTPUT_DIR, f"{kode}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        berhasil.append(kode)
        print(f"  ✅ {kode}: OK")
    else:
        gagal.append(kode)
        print(f"  ❌ {kode}: GAGAL")
    time.sleep(5)  # jeda 5 detik antar saham

print(f"\n{'='*60}")
print(f"Berhasil: {len(berhasil)} -> {berhasil}")
print(f"Gagal   : {len(gagal)} -> {gagal}")
print("=" * 60)