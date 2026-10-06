"""
Scraper fundamental DETAIL per saham (untuk halaman detail).
Ambil dari yfinance: rasio kunci, riwayat tahunan, statistik, tren.
Multi-worker (ThreadPoolExecutor) - 15-20x lebih cepat.

Output: data_fundamental_detail/{KODE}.json
Jalankan: python -m quant.fundamental_detail
"""

import os
import json
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# KONFIGURASI
# ============================================================
OUTPUT_DIR = "data_fundamental_detail"
CSV_DAFTAR = "data_sektor.csv"
CSV_FALLBACK = "daftar_saham.csv"
TEST_MODE = False
TEST_TICKERS = ["BBCA", "BBRI", "TLKM", "ASII", "ANTM"]

MAX_WORKERS = 5           # 10-15 optimal untuk yfinance
JITTER_MIN = 0.15         # jeda kecil antar request per thread
JITTER_MAX = 0.40
RETRY = 3                 # retry kalau gagal


# ============================================================
# DAFTAR SAHAM
# ============================================================
def get_ticker_list():
    for path in [CSV_DAFTAR, CSV_FALLBACK]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            col = None
            for c in ["Kode", "kode", "Code", "KodeEmiten"]:
                if c in df.columns:
                    col = c
                    break
            if col:
                return sorted(df[col].dropna().str.upper().unique().tolist())
    return []


def _safe(v):
    if v is None:
        return None
    try:
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _series_to_dict(s, max_years=5):
    if s is None or s.empty:
        return {}
    out = {}
    try:
        for col, val in s.items():
            year = str(col)[:4]
            v = _safe(val)
            if v is not None:
                out[year] = v
    except Exception:
        pass
    return out


# ============================================================
# HITUNG AKUMULASI DARI DATA LOKAL
# ============================================================
def _hitung_akumulasi(kode, hari=20):
    path = os.path.join("data_ohlcv_idx", f"{kode}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if len(df) < hari + 5:
            return None
        df = df.tail(hari + 5).reset_index(drop=True)

        mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / \
              (df["High"] - df["Low"]).replace(0, np.nan)
        mfv = mfm * df["Volume"]
        cmf = mfv.tail(hari).sum() / df["Volume"].tail(hari).sum()
        cmf = _safe(cmf) or 0

        v_recent = df["Volume"].tail(5).mean()
        v_prev = df["Volume"].tail(hari).head(10).mean()
        vol_trend = v_recent / v_prev if v_prev > 0 else 1.0

        price_chg = (df["Close"].iloc[-1] / df["Close"].iloc[-hari] - 1) * 100

        score = 5
        if cmf > 0.15: score += 2
        elif cmf > 0.05: score += 1
        elif cmf < -0.15: score -= 2
        elif cmf < -0.05: score -= 1

        if vol_trend > 1.5 and price_chg > 0: score += 2
        elif vol_trend > 1.5 and price_chg < 0: score -= 2
        elif vol_trend > 1.2: score += 1

        if price_chg > 5: score += 1
        elif price_chg < -5: score -= 1

        score = max(0, min(10, score))

        if score >= 7:
            label, tipe = "AKUMULASI", "ACCUMULATION"
        elif score >= 5:
            label, tipe = "NETRAL", "NEUTRAL"
        else:
            label, tipe = "DISTRIBUSI", "DISTRIBUTION"

        return {
            "score": round(score, 1),
            "label": label,
            "tipe": tipe,
            "cmf": round(cmf, 3),
            "vol_trend": round(vol_trend, 2),
            "price_chg_20d": round(price_chg, 2),
        }
    except Exception:
        return None


# ============================================================
# FETCH DETAIL 1 SAHAM
# ============================================================
def fetch_detail(kode):
    try:
        t = yf.Ticker(f"{kode}.JK")
        info = t.info or {}

        if not info or info.get("regularMarketPrice") is None:
            return None

        rasio_kunci = {
            "pe": _safe(info.get("trailingPE")),
            "pb": _safe(info.get("priceToBook")),
            "ps": _safe(info.get("priceToSalesTrailing12Months")),
            "roe": _safe(info.get("returnOnEquity")),
            "roic": None,
            "de": _safe(info.get("debtToEquity")),
            "margin_net": _safe(info.get("profitMargins")),
            "current_ratio": _safe(info.get("currentRatio")),
            "dividend_yield": _safe(info.get("dividendYield")),
            "eps": _safe(info.get("trailingEps")),
            "bv": _safe(info.get("bookValue")),
        }

        statistik = {
            "market_cap": _safe(info.get("marketCap")),
            "enterprise_value": _safe(info.get("enterpriseValue")),
            "shares_outstanding": _safe(info.get("sharesOutstanding")),
            "float_shares": _safe(info.get("floatShares")),
            "free_float_pct": None,
            "avg_volume": _safe(info.get("averageVolume")),
            "beta": _safe(info.get("beta")),
            "harga": _safe(info.get("regularMarketPrice")),
        }

        if statistik["shares_outstanding"] and statistik["float_shares"]:
            statistik["free_float_pct"] = (
                statistik["float_shares"] / statistik["shares_outstanding"]
            )

        # Riwayat tahunan
        try:
            income = t.income_stmt
            eps_hist = _series_to_dict(
                income.loc["Basic EPS"] if "Basic EPS" in income.index else None
            )
            revenue_hist = _series_to_dict(
                income.loc["Total Revenue"] if "Total Revenue" in income.index else None
            )
            net_income_hist = _series_to_dict(
                income.loc["Net Income"] if "Net Income" in income.index else None
            )
        except Exception:
            eps_hist, revenue_hist, net_income_hist = {}, {}, {}

        shares = statistik["shares_outstanding"]
        if not eps_hist and net_income_hist and shares:
            eps_hist = {y: v / shares for y, v in net_income_hist.items()}

        rev_per_share = {}
        if revenue_hist and shares:
            rev_per_share = {y: v / shares for y, v in revenue_hist.items()}

        riwayat = {
            "eps": eps_hist,
            "revenue": revenue_hist,
            "revenue_per_share": rev_per_share,
            "net_income": net_income_hist,
        }

        # Tren rasio
        try:
            balance = t.balance_sheet
            equity_hist = _series_to_dict(
                balance.loc["Stockholders Equity"]
                if "Stockholders Equity" in balance.index else None
            )
            total_assets_hist = _series_to_dict(
                balance.loc["Total Assets"]
                if "Total Assets" in balance.index else None
            )
        except Exception:
            equity_hist, total_assets_hist = {}, {}

        roe_hist = {}
        for y, ni in net_income_hist.items():
            eq = equity_hist.get(y)
            if eq and eq > 0:
                roe_hist[y] = ni / eq * 100

        roa_hist = {}
        for y, ni in net_income_hist.items():
            ta = total_assets_hist.get(y)
            if ta and ta > 0:
                roa_hist[y] = ni / ta * 100

        pe_hist = {}
        if eps_hist:
            for y, eps in eps_hist.items():
                if eps and eps > 0:
                    try:
                        hist = t.history(start=f"{y}-12-01", end=f"{y}-12-31")
                        if not hist.empty:
                            price = float(hist["Close"].iloc[-1])
                            pe_hist[y] = price / eps
                    except Exception:
                        pass

        tren = {"roe": roe_hist, "roa": roa_hist, "pe": pe_hist}

        akumulasi = _hitung_akumulasi(kode)

        return {
            "kode": kode,
            "nama": info.get("longName", ""),
            "sektor": info.get("sector", ""),
            "industri": info.get("industry", ""),
            "harga": _safe(info.get("regularMarketPrice")),
            "rasio_kunci": rasio_kunci,
            "statistik": statistik,
            "riwayat": riwayat,
            "tren": tren,
            "akumulasi": akumulasi,
            "update": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    except Exception:
        return None


# ============================================================
# WORKER (dengan retry + jitter)
# ============================================================
def worker(kode):
    time.sleep(random.uniform(JITTER_MIN, JITTER_MAX))
    for attempt in range(RETRY):
        result = fetch_detail(kode)
        if result:
            return kode, result
        if attempt < RETRY - 1:
            time.sleep(1.0)
    return kode, None


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print(f"FUNDAMENTAL DETAIL (PARALLEL) — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if TEST_MODE:
        tickers = TEST_TICKERS
        print(f"\n[MODE TES] {len(tickers)} saham")
    else:
        tickers = get_ticker_list()
        if not tickers:
            print("Tidak ada daftar saham.")
            return
        print(f"\nTotal saham: {len(tickers)}")

    sudah = set(f.replace(".json", "") for f in os.listdir(OUTPUT_DIR)
                if f.endswith(".json"))
    belum = [k for k in tickers if k not in sudah]

    print(f"Sudah ada: {len(sudah)}")
    print(f"Perlu scrape: {len(belum)}")
    print(f"Workers: {MAX_WORKERS} (paralel)")

    if not belum:
        print("\n[OK] Semua selesai.")
        return

    print(f"\nMulai scrape dengan {MAX_WORKERS} worker...\n")
    t0 = datetime.now()

    berhasil, gagal = [], []
    lock = threading.Lock()
    counter = {"done": 0, "ok": 0, "err": 0}
    total = len(belum)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker, k): k for k in belum}

        for future in as_completed(futures):
            kode, result = future.result()

            with lock:
                counter["done"] += 1

                if result:
                    filepath = os.path.join(OUTPUT_DIR, f"{kode}.json")
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2,
                                  ensure_ascii=False, default=str)
                    berhasil.append(kode)
                    counter["ok"] += 1
                else:
                    gagal.append(kode)
                    counter["err"] += 1

                # Progress tiap 25 saham
                if counter["done"] % 25 == 0 or counter["done"] == total:
                    elapsed = (datetime.now() - t0).total_seconds()
                    rate = counter["done"] / elapsed if elapsed > 0 else 0
                    eta = (total - counter["done"]) / rate if rate > 0 else 0
                    print(f"  [{counter['done']}/{total}] "
                          f"OK={counter['ok']} err={counter['err']} "
                          f"| {rate:.1f}/s | ETA {eta:.0f}s")

    elapsed = (datetime.now() - t0).total_seconds()

    print("\n" + "=" * 70)
    print(f"Selesai dalam : {elapsed:.1f} detik")
    print(f"Berhasil      : {len(berhasil)}")
    print(f"Gagal         : {len(gagal)}")
    if gagal:
        with open("failed_detail.txt", "w") as f:
            for k in sorted(set(gagal)):
                f.write(f"{k}\n")
        print("-> Daftar gagal: failed_detail.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()