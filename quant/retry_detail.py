"""Retry gagal dengan pengaturan konservatif."""
import os
import sys
import json
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant.fundamental_detail import fetch_detail, OUTPUT_DIR


# KONFIGURASI KONSERVATIF
MAX_WORKERS = 2
JITTER_MIN = 0.8
JITTER_MAX = 1.5
RETRY = 3


def worker(kode):
    time.sleep(random.uniform(JITTER_MIN, JITTER_MAX))
    for attempt in range(RETRY):
        try:
            result = fetch_detail(kode)
            if result:
                return kode, result
        except Exception:
            pass
        time.sleep(3.0)
    return kode, None


def main():
    failed_path = "failed_retry.txt"
    if not os.path.exists(failed_path):
        print("Tidak ada failed_retry.txt.")
        print("Jalankan dulu: python -m quant.split_failed")
        return

    with open(failed_path) as f:
        tickers = [l.strip() for l in f if l.strip()]

    if not tickers:
        print("Tidak ada saham untuk di-retry.")
        return

    print("=" * 70)
    print(f"RETRY SLOW MODE — {len(tickers)} saham")
    print(f"Workers: {MAX_WORKERS} (konservatif)")
    print(f"Jitter : {JITTER_MIN}-{JITTER_MAX}s")
    print(f"Retry  : {RETRY}x")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = datetime.now()
    berhasil, masih_gagal = [], []
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(worker, k): k for k in tickers}

        for i, fut in enumerate(as_completed(futures), 1):
            kode, result = fut.result()

            with lock:
                if result:
                    path = os.path.join(OUTPUT_DIR, f"{kode}.json")
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2,
                                  ensure_ascii=False, default=str)
                    berhasil.append(kode)
                else:
                    masih_gagal.append(kode)

                if i % 10 == 0 or i == len(tickers):
                    elapsed = (datetime.now() - t0).total_seconds()
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(tickers) - i) / rate if rate > 0 else 0
                    print(f"  [{i}/{len(tickers)}] "
                          f"OK={len(berhasil)} err={len(masih_gagal)} "
                          f"| ETA {eta:.0f}s")

    print(f"\nSelesai dalam {(datetime.now()-t0).total_seconds():.0f}s")
    print(f"Berhasil     : {len(berhasil)}")
    print(f"Masih gagal  : {len(masih_gagal)}")

    with open("failed_retry.txt", "w") as f:
        f.write("\n".join(masih_gagal))


if __name__ == "__main__":
    main()