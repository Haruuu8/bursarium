"""
Scraper IHSG harian dari Yahoo Finance menggunakan yfinance.
Ticker untuk IHSG di Yahoo Finance adalah '^JKSE'.
"""

import yfinance as yf
import pandas as pd
import os

OUTPUT = "quant/data/ihsg.csv"
TICKER = "^JKSE"

def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    print(f"Mengunduh data IHSG ({TICKER}) dari Yahoo Finance...")
    df = yf.download(TICKER, period="1y", interval="1d", progress=False)

    if df.empty:
        print("❌ Tidak ada data yang diterima.")
        return

    # Ratakan kolom multi-level jika ada
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Rename kolom agar konsisten
    df = df.rename(columns={
        "Open": "OpenPrice",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
    })

    df = df.reset_index()
    df = df.rename(columns={"index": "Date", "Date": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df["Kode"] = "IHSG"

    df.to_csv(OUTPUT, index=False)
    print(f"✅ Berhasil. {len(df)} baris disimpan ke {OUTPUT}")
    print(df[["Date", "Close"]].tail())

if __name__ == "__main__":
    main()