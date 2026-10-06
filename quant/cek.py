"""Cek data tanggal spesifik untuk saham tertentu."""
import pandas as pd


def cek(kode, tanggal):
    path = f"data_ohlcv_idx/{kode}.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    print(f"\n=== {kode} ===")
    print(f"5 baris terakhir:")
    print(df.tail(5)[["Date", "OpenPrice", "High", "Low", "Close", "Volume"]].to_string(index=False))

    # Cek tanggal spesifik
    baris = df[df["Date"] == tanggal]
    if len(baris) == 0:
        print(f"\n  [!] Tidak ada baris untuk {tanggal}")
        # Cek tanggal terdekat
        before = df[df["Date"] < tanggal]
        after = df[df["Date"] > tanggal]
        if len(before) > 0:
            print(f"  Tanggal terdekat sebelum: {before['Date'].max().date()}")
        if len(after) > 0:
            print(f"  Tanggal terdekat sesudah: {after['Date'].min().date()}")
    else:
        print(f"\n  Baris untuk {tanggal}:")
        print(baris[["Date", "OpenPrice", "High", "Low", "Close", "Volume"]].to_string(index=False))


if __name__ == "__main__":
    for kode in ["TEBE", "ASPI"]:
        cek(kode, "2026-10-01")