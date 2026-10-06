"""
Loader data mentah dari hasil scraper.
Menghasilkan panel data: satu baris = (Date, Kode).
"""

import pandas as pd
from pathlib import Path
from .config import (
    RAW_CSV_GABUNGAN, RAW_CSV_PER_SAHAM, DAFTAR_SAHAM_CSV, IHSG_CSV,
)


# Kolom standar yang kita butuhkan dari endpoint IDX
KOLOM_STANDAR = {
    "Date": "date",
    "StockCode": "kode",
    "OpenPrice": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Previous": "prev_close",
    "Volume": "volume",
    "Value": "value",
    "Frequency": "frequency",
    "ForeignBuy": "foreign_buy",
    "ForeignSell": "foreign_sell",
    "ListedShares": "listed_shares",
    "TradebleShares": "tradeable_shares",
}


def load_panel_from_gabungan() -> pd.DataFrame:
    """Load dari ohlcv_idx_2026_all.csv (gabungan semua saham)."""
    if not RAW_CSV_GABUNGAN.exists():
        raise FileNotFoundError(
            f"File gabungan tidak ditemukan: {RAW_CSV_GABUNGAN}\n"
            "Jalankan ticker.py atau gabung.py dulu."
        )

    df = pd.read_csv(RAW_CSV_GABUNGAN)

    # Map nama kolom
    kolom_ada = {k: v for k, v in KOLOM_STANDAR.items() if k in df.columns}
    df = df.rename(columns=kolom_ada)

    # Wajib punya 'Kode' — di gabungan namanya 'Kode'
    if "kode" not in df.columns and "Kode" in df.columns:
        df = df.rename(columns={"Kode": "kode"})

    # Konversi tipe
    df["date"] = pd.to_datetime(df["date"])
    if "kode" not in df.columns:
        raise ValueError("Kolom 'kode' tidak ditemukan di file gabungan.")

    df["kode"] = df["kode"].astype(str).str.upper().str.strip()

    # Urutkan
    df = df.sort_values(["kode", "date"]).reset_index(drop=True)

    # Buang kolom yang tidak dipakai supaya ringan
        # Buang kolom yang tidak diperlukan
    drop_cols = [c for c in df.columns
                 if c.startswith("Unnamed") or c in
                 ("IDStockSummary", "StockName", "Remarks", "No",
                  "DelistingDate", "IndexIndividual", "TradebleShares")]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Konversi kolom yang seharusnya numerik — biarkan yang gagal jadi NaN
    for col in df.columns:
        if col in ("date", "kode"):
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def load_ihsg() -> pd.DataFrame:
    """Load IHSG. Return DataFrame dengan kolom: date, ihsg_close, dst."""
    if not IHSG_CSV.exists():
        print(f"⚠️ File IHSG tidak ada: {IHSG_CSV}")
        print("   Jalankan quant/scrape_ihsg.py dulu.")
        return pd.DataFrame()

    df = pd.read_csv(IHSG_CSV)
    df = df.rename(columns=KOLOM_STANDAR)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Beri prefix 'ihsg_' pada kolom harga untuk hindari tabrakan
    rename_ihsg = {c: f"ihsg_{c}" for c in
                   ("open", "high", "low", "close", "volume", "value", "frequency")
                   if c in df.columns}
    df = df.rename(columns=rename_ihsg)

    return df[["date"] + list(rename_ihsg.values())]


def load_daftar_saham() -> list[str]:
    """Baca daftar saham dari CSV."""
    if not DAFTAR_SAHAM_CSV.exists():
        return []
    df = pd.read_csv(DAFTAR_SAHAM_CSV)
    for key in ["Kode", "Kode Saham", "Code", "kode"]:
        if key in df.columns:
            return df[key].dropna().astype(str).str.upper().str.strip().tolist()
    return []


if __name__ == "__main__":
    # Smoke test
    panel = load_panel_from_gabungan()
    print(f"Panel: {panel.shape}")
    print(f"Kolom: {list(panel.columns)}")
    print(f"Jumlah saham: {panel['kode'].nunique()}")
    print(f"Rentang: {panel['date'].min()} s/d {panel['date'].max()}")
    print("\nContoh:")
    print(panel.head())

    ihsg = load_ihsg()
    if not ihsg.empty:
        print(f"\nIHSG: {ihsg.shape}, {ihsg['date'].min()} s/d {ihsg['date'].max()}")