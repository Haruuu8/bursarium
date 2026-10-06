"""
Halaman FUNDAMENTAL SAHAM — layout ala Stockbit.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant.theme import load_theme  # ← TAMBAH 1

st.set_page_config(page_title="Fundamental", page_icon="📈", layout="wide")

load_theme()  # ← TAMBAH 2
# ============================================================
# KONFIGURASI
# ============================================================
OUTPUT_DIR = "data_fundamental"
CSV_DAFTAR = "data_sektor.csv"
CSV_FALLBACK = "daftar_saham.csv"
TEST_MODE = True
TEST_TICKERS = ["BBCA", "BBRI", "TLKM", "ASII", "ANTM"]
DELAY = 0.3


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


# ============================================================
# FETCH 1 SAHAM
# ============================================================
def fetch_fundamental(kode):
    try:
        ticker = yf.Ticker(f"{kode}.JK")
        info = ticker.info or {}

        if not info or info.get("regularMarketPrice") is None:
            return None

        metrics = {
            "per": _safe(info.get("trailingPE")),
            "pbv": _safe(info.get("priceToBook")),
            "roe": _safe(info.get("returnOnEquity")),
            "roa": _safe(info.get("returnOnAssets")),
            "der": _safe(info.get("debtToEquity")),
            "eps": _safe(info.get("trailingEps")),
            "bv": _safe(info.get("bookValue")),
            "dividend_yield": _safe(info.get("dividendYield")),
            "net_margin": _safe(info.get("profitMargins")),
            "gross_margin": _safe(info.get("grossMargins")),
            "revenue": _safe(info.get("totalRevenue")),
            "net_income": _safe(info.get("netIncomeToCommon")),
            "market_cap": _safe(info.get("marketCap")),
            "total_asset": None,
            "total_equity": None,
            "current_ratio": _safe(info.get("currentRatio")),
        }

        n_valid = sum(1 for v in metrics.values() if v is not None)
        if n_valid < 1:
            return None

        return {
            "kode": kode,
            "sumber": "yfinance",
            "metrics": metrics,
            "info_extra": {
                "nama": info.get("longName", ""),
                "sektor": info.get("sector", ""),
                "industri": info.get("industry", ""),
                "harga": info.get("regularMarketPrice"),
            },
        }
    except Exception:
        return None


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


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print(f"FUNDAMENTAL SCRAPER v3 (yfinance) — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if TEST_MODE:
        tickers = TEST_TICKERS
        print(f"\n[MODE TES] {len(tickers)} saham")
    else:
        tickers = get_ticker_list()
        if not tickers:
            print("Tidak ada daftar saham. Pastikan data_sektor.csv ada.")
            return
        print(f"\nTotal saham: {len(tickers)}")

    sudah = set(f.replace(".json", "") for f in os.listdir(OUTPUT_DIR)
                if f.endswith(".json"))
    belum = [k for k in tickers if k not in sudah]
    print(f"Sudah ada: {len(sudah)}")
    print(f"Perlu scrape: {len(belum)}\n")

    if not belum:
        print("[OK] Semua sudah selesai.")
        return

    berhasil, gagal = [], []
    for i, kode in enumerate(belum, 1):
        result = fetch_fundamental(kode)
        if result:
            filepath = os.path.join(OUTPUT_DIR, f"{kode}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            n_valid = sum(1 for v in result["metrics"].values() if v is not None)
            berhasil.append(kode)
            print(f"[{i}/{len(belum)}] {kode}: OK ({n_valid} metrik)")
        else:
            gagal.append(kode)
            print(f"[{i}/{len(belum)}] {kode}: GAGAL")
        time.sleep(DELAY)

    print("\n" + "=" * 70)
    print(f"Berhasil: {len(berhasil)}")
    print(f"Gagal   : {len(gagal)}")
    if gagal:
        with open("failed_fundamental.txt", "w") as f:
            for k in gagal:
                f.write(f"{k}\n")
        print("-> Daftar gagal: failed_fundamental.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()