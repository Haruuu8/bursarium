"""
UPDATE HARIAN - versi CEPAT (20 worker).
Menggunakan Scrapling AsyncStealthySession (bypass Cloudflare).

Fitur:
- Skip otomatis kalau weekend (Sabtu/Minggu)
- Cek jam publish IDX (~17:00 WIB)
- Fetch hanya data baru (bukan download ulang)
- Buffer 14 hari untuk jaga-jaga
- 20 concurrent worker
"""

import asyncio
import os
import csv
import random
from datetime import datetime
import pandas as pd
from scrapling.fetchers import AsyncStealthySession


# ============================================================
# KONFIGURASI
# ============================================================
TAHUN = 2026

# ---- Performa ----
MAX_WORKERS = 20          # concurrent request
MAX_TABS = 10             # browser tabs di pool

# ---- Data ----
LOOKBACK_DAYS = 14
OUTPUT_DIR = "data_ohlcv_idx"
CSV_DAFTAR = "daftar_saham.csv"
FILE_FAILED = "failed_update.txt"
BASE = "https://www.idx.co.id"
JAM_PUBLISH_IDX = 17

# ---- Delay (jitter kecil biar tidak dicurigai) ----
JITTER_MIN = 0.02
JITTER_MAX = 0.10


# ============================================================
# BACA DAFTAR SAHAM
# ============================================================
def get_ticker_list():
    if not os.path.exists(CSV_DAFTAR):
        print(f"  [!] File '{CSV_DAFTAR}' tidak ada.")
        return []

    tickers = []
    with open(CSV_DAFTAR, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            for key in ["Kode", "Kode Saham", "Code", "KodeEmiten", "kode"]:
                if key in row and row[key]:
                    kode = row[key].strip().upper()
                    if kode and len(kode) <= 6 and kode.isalnum():
                        tickers.append(kode)
                    break

    return sorted(set(tickers))


# ============================================================
# TANGGAL TERAKHIR
# ============================================================
def tanggal_terakhir(kode):
    fp = os.path.join(OUTPUT_DIR, f"{kode}.csv")
    if not os.path.exists(fp):
        return None
    try:
        df = pd.read_csv(fp, usecols=["Date"])
        if df.empty:
            return None
        return pd.to_datetime(df["Date"]).max().normalize()
    except Exception:
        return None


# ============================================================
# FETCH
# ============================================================
async def fetch_ohlcv(session, kode, length, retry=2):
    url = (
        f"{BASE}/primary/ListedCompany/GetTradingInfoSS"
        f"?code={kode}&length={length}"
    )
    for attempt in range(retry):
        try:
            page = await session.fetch(url)
            if page.status != 200:
                if attempt < retry - 1:
                    await asyncio.sleep(0.5)
                    continue
                return kode, None, f"HTTP {page.status}"

            data = page.json()
            replies = data.get("replies", [])
            if not replies:
                return kode, None, "replies kosong"

            df = pd.DataFrame(replies)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
            return kode, df, None
        except Exception as e:
            if attempt < retry - 1:
                await asyncio.sleep(0.5)
                continue
            return kode, None, f"{type(e).__name__}: {e}"
    return kode, None, "unknown error"


# ============================================================
# MERGE
# ============================================================
def merge_dan_simpan(kode, df_baru):
    fp = os.path.join(OUTPUT_DIR, f"{kode}.csv")

    if os.path.exists(fp):
        df_lama = pd.read_csv(fp)
        df_lama["Date"] = pd.to_datetime(df_lama["Date"])
        df = pd.concat([df_lama, df_baru], ignore_index=True)
    else:
        df = df_baru.copy()

    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.sort_values("Date").reset_index(drop=True)
    df = df[df["Date"].dt.year == TAHUN]

    if df.empty:
        return 0

    df.to_csv(fp, index=False)
    return len(df)


# ============================================================
# HARI BURSA TERAKHIR (jam WIB)
# ============================================================
def get_hari_bursa_terakhir():
    now = _now_wib()
    hari_ini = pd.Timestamp(now.date())
    jam = now.hour
    dow = hari_ini.weekday()

    if dow == 5:
        return hari_ini - pd.Timedelta(days=1)
    if dow == 6:
        return hari_ini - pd.Timedelta(days=2)

    if jam >= JAM_PUBLISH_IDX:
        return hari_ini
    else:
        if dow == 0:
            return hari_ini - pd.Timedelta(days=3)
        else:
            return hari_ini - pd.Timedelta(days=1)


def _now_wib():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Jakarta"))
    except Exception:
        try:
            import pytz
            return datetime.now(pytz.timezone("Asia/Jakarta"))
        except ImportError:
            return datetime.now()


# ============================================================
# MAIN
# ============================================================
async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    now_wib = _now_wib()
    print("=" * 55)
    print(f"UPDATE HARIAN (FAST MODE) - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 55)
    print(f"Waktu WIB      : {now_wib.strftime('%Y-%m-%d %H:%M')}")
    print(f"Jam publish IDX: {JAM_PUBLISH_IDX}:00 WIB")
    print(f"Workers        : {MAX_WORKERS} concurrent")
    print(f"Browser tabs   : {MAX_TABS}")

    hari_ini = pd.Timestamp(now_wib.date())
    dow = hari_ini.weekday()
    if dow >= 5:
        print(f"\n[SKIP] Hari ini {['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu'][dow]}, "
              f"pasar tutup.")
        return

    tickers = get_ticker_list()
    if not tickers:
        print("Tidak ada saham.")
        return
    print(f"\nTotal saham dari CSV: {len(tickers)}")

    batas_update = get_hari_bursa_terakhir()
    print(f"Hari bursa terakhir: {batas_update.date()} "
          f"({['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu'][batas_update.weekday()]})")

    if batas_update.date() < hari_ini.date() and dow < 5:
        print(f"  [i] Data hari ini ({hari_ini.date()}) belum publish.")
        print(f"      Coba lagi setelah jam {JAM_PUBLISH_IDX}:00 WIB.")

    # Filter
    perlu_update = []
    sudah_uptodate = []
    for kode in tickers:
        tgl = tanggal_terakhir(kode)
        if tgl is None or tgl < batas_update:
            perlu_update.append(kode)
        else:
            sudah_uptodate.append(kode)

    print(f"\nPerlu update   : {len(perlu_update)}")
    print(f"Sudah terbaru  : {len(sudah_uptodate)}")

    if not perlu_update:
        print("\n[OK] Semua saham sudah up-to-date.")
        return

    # ---------- Fetch paralel ----------
    print(f"\nMulai fetch dengan {MAX_WORKERS} worker...\n")
    t0 = datetime.now()

    async with AsyncStealthySession(
        headless=True,
        solve_cloudflare=True,
        max_pages=MAX_TABS,
        disable_resources=True,
    ) as session:

        sem = asyncio.Semaphore(MAX_WORKERS)
        counter = {"done": 0, "ok": 0, "skip": 0, "err": 0}
        lock = asyncio.Lock()
        total = len(perlu_update)

        async def worker(kode):
            async with sem:
                # Jitter kecil biar tidak pattern-able
                await asyncio.sleep(random.uniform(JITTER_MIN, JITTER_MAX))

                length = LOOKBACK_DAYS if tanggal_terakhir(kode) else 365
                kode, df_baru, err = await fetch_ohlcv(session, kode, length)

                if df_baru is None or df_baru.empty:
                    status = "error" if err else "skip"
                    return kode, 0, err or "tidak ada data", status

                tgl_lama = tanggal_terakhir(kode)
                if tgl_lama is not None:
                    df_baru = df_baru[df_baru["Date"] > tgl_lama]

                if df_baru.empty:
                    return kode, 0, "tidak ada baris baru", "skip"

                n = merge_dan_simpan(kode, df_baru)
                return kode, n, None, "ok"

        tasks = [worker(k) for k in perlu_update]
        berhasil, gagal, tidak_ada_baru = [], [], []

        for coro in asyncio.as_completed(tasks):
            kode, nbaris, err, status = await coro

            async with lock:
                counter["done"] += 1
                if status == "ok":
                    counter["ok"] += 1
                    berhasil.append(kode)
                elif status == "skip":
                    counter["skip"] += 1
                    tidak_ada_baru.append(kode)
                else:
                    counter["err"] += 1
                    gagal.append(kode)

                # Progress tiap 50 saham
                if counter["done"] % 50 == 0 or counter["done"] == total:
                    elapsed = (datetime.now() - t0).total_seconds()
                    rate = counter["done"] / elapsed if elapsed > 0 else 0
                    eta = (total - counter["done"]) / rate if rate > 0 else 0
                    print(f"  [{counter['done']}/{total}] "
                          f"OK={counter['ok']} skip={counter['skip']} err={counter['err']} "
                          f"| {rate:.1f}/s | ETA {eta:.0f}s")

    # ---------- Simpan gagal ----------
    if gagal:
        with open(FILE_FAILED, "w", encoding="utf-8") as f:
            for k in sorted(set(gagal)):
                f.write(f"{k}\n")
        print(f"\n[!] {len(gagal)} saham gagal -> {FILE_FAILED}")
    else:
        if os.path.exists(FILE_FAILED):
            os.remove(FILE_FAILED)

    # ---------- Ringkasan ----------
    elapsed = (datetime.now() - t0).total_seconds()
    print("\n" + "=" * 55)
    print(f"Selesai dalam      : {elapsed:.1f} detik")
    print(f"Berhasil update    : {len(berhasil)}")
    print(f"Tidak ada baris    : {len(tidak_ada_baru)}")
    print(f"Sudah up-to-date   : {len(sudah_uptodate)}")
    print(f"Gagal              : {len(gagal)}")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())