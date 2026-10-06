"""
Scraper OHLCV semua saham IDX. Auto-save daftar gagal ke failed_tickers.txt.
"""

import asyncio
import csv
import os
import pandas as pd
from scrapling.fetchers import AsyncStealthySession

TAHUN = 2026
MAX_PAGES = 10
OUTPUT_DIR = "data_ohlcv_idx"
FILE_GABUNGAN = "ohlcv_idx_2026_all.csv"
FILE_FAILED = "failed_tickers.txt"
CSV_DAFTAR = "daftar_saham.csv"
BASE = "https://www.idx.co.id"

FALLBACK_SAHAM = [
    "BBCA", "BBRI", "BMRI", "TLKM", "ASII", "GOTO", "AMMN", "BBNI",
    "INDF", "UNVR", "ICBP", "KLBF", "ANTM", "ADRO", "PTBA", "SMGR",
]


def get_ticker_list():
    if not os.path.exists(CSV_DAFTAR):
        print(f"  ⚠️ '{CSV_DAFTAR}' tidak ada, pakai fallback")
        return FALLBACK_SAHAM

    tickers = []
    with open(CSV_DAFTAR, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            for key in ["Kode", "Kode Saham", "Code", "KodeEmiten", "kode"]:
                if key in row and row[key]:
                    kode = row[key].strip().upper()
                    if kode and len(kode) <= 6 and kode.isalnum():
                        tickers.append(kode)
                    break

    tickers = sorted(set(tickers))
    if len(tickers) < 500:
        print(f"  ⚠️ Hanya {len(tickers)} saham dari CSV, pakai fallback")
        return FALLBACK_SAHAM

    print(f"  ✅ {len(tickers)} saham dari '{CSV_DAFTAR}'")
    return tickers


async def fetch_ohlcv(session, kode):
    url = f"{BASE}/primary/ListedCompany/GetTradingInfoSS?code={kode}&length=365"
    try:
        page = await session.fetch(url)
        if page.status != 200:
            return kode, None, f"HTTP {page.status}"

        data = page.json()
        replies = data.get("replies", [])
        if not replies:
            return kode, None, "replies kosong"

        df = pd.DataFrame(replies)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df[df["Date"].dt.year == TAHUN]

        if df.empty:
            return kode, None, "tidak ada data tahun ini"

        return kode, df, None
    except Exception as e:
        return kode, None, f"{type(e).__name__}: {e}"


def simpan_failed(failed_list):
    """Simpan daftar gagal ke file, urut alfabetis."""
    with open(FILE_FAILED, "w", encoding="utf-8") as f:
        for kode in sorted(set(failed_list)):
            f.write(f"{kode}\n")
    print(f"→ Daftar gagal disimpan di {FILE_FAILED} ({len(failed_list)} saham)")


async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Membaca daftar saham...")
    tickers = get_ticker_list()

    async with AsyncStealthySession(
        headless=True, solve_cloudflare=True,
        max_pages=MAX_PAGES, disable_resources=True,
    ) as session:

        print(f"\nMengambil {len(tickers)} saham dengan {MAX_PAGES} worker...\n")
        sem = asyncio.Semaphore(MAX_PAGES)

        async def worker(kode):
            async with sem:
                return await fetch_ohlcv(session, kode)

        tasks = [worker(k) for k in tickers]
        berhasil, gagal = [], []

        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            kode, df, err = await coro
            if err is None:
                df.to_csv(os.path.join(OUTPUT_DIR, f"{kode}.csv"), index=False)
                berhasil.append(kode)
                print(f"[{i}/{len(tickers)}] {kode}: OK ({len(df)} baris)")
            else:
                gagal.append(kode)
                print(f"[{i}/{len(tickers)}] {kode}: GAGAL - {err}")

    # Gabungkan
    print("\n" + "=" * 50)
    print("Menggabungkan semua data...")
    all_frames = []
    for kode in berhasil:
        fp = os.path.join(OUTPUT_DIR, f"{kode}.csv")
        df = pd.read_csv(fp)
        df["Kode"] = kode
        all_frames.append(df)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        cols = ["Kode", "Date"] + [c for c in combined.columns if c not in ["Kode", "Date"]]
        combined = combined[cols]
        combined.to_csv(FILE_GABUNGAN, index=False)
        print(f"✅ {FILE_GABUNGAN} ({len(combined)} baris)")

    # Ringkasan + auto-save daftar gagal
    print("\n" + "=" * 50)
    print(f"BERHASIL: {len(berhasil)} saham")
    print(f"GAGAL   : {len(gagal)} saham")

    if gagal:
        print("\nSaham gagal:")
        for k in gagal:
            print(f"  - {k}")
        simpan_failed(gagal)
        print("\n👉 Jalankan `python retry.py` untuk mencoba ulang yang gagal.")
    else:
        # Kosongkan file failed kalau tidak ada yang gagal
        if os.path.exists(FILE_FAILED):
            os.remove(FILE_FAILED)


if __name__ == "__main__":
    asyncio.run(main())