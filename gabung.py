"""
Gabungkan semua CSV per saham di data_ohlcv_idx/ menjadi satu file.
Output: ohlcv_idx_2026_all.csv
"""

import os
import pandas as pd

OUTPUT_DIR = "data_ohlcv_idx"
FILE_GABUNGAN = "ohlcv_idx_2026_all.csv"


def main():
    if not os.path.isdir(OUTPUT_DIR):
        print(f"❌ Folder '{OUTPUT_DIR}' tidak ditemukan.")
        return

    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv")])
    if not files:
        print(f"❌ Tidak ada file CSV di '{OUTPUT_DIR}'.")
        return

    print(f"Menggabungkan {len(files)} file dari '{OUTPUT_DIR}'...")

    all_frames = []
    for i, fname in enumerate(files, 1):
        kode = fname.replace(".csv", "")
        try:
            df = pd.read_csv(os.path.join(OUTPUT_DIR, fname))
            if df.empty:
                continue
            df["Kode"] = kode
            all_frames.append(df)
        except Exception as e:
            print(f"  ⚠️ Skip {kode}: {e}")

        if i % 100 == 0:
            print(f"  ... proses {i}/{len(files)}")

    if not all_frames:
        print("❌ Tidak ada data yang valid untuk digabungkan.")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    # Urutkan kolom: Kode, Date di depan
    kolom_awal = ["Kode", "Date"]
    kolom_sisa = [c for c in combined.columns if c not in kolom_awal]
    combined = combined[kolom_awal + kolom_sisa]

    # Sort by Kode, Date
    combined = combined.sort_values(["Kode", "Date"]).reset_index(drop=True)

    combined.to_csv(FILE_GABUNGAN, index=False)

    print("\n" + "=" * 55)
    print(f"✅ Disimpan: {FILE_GABUNGAN}")
    print(f"   Total baris  : {len(combined):,}")
    print(f"   Total saham  : {combined['Kode'].nunique()}")
    if "Date" in combined.columns:
        print(f"   Rentang      : {combined['Date'].min()} s/d {combined['Date'].max()}")
    print("=" * 55)


if __name__ == "__main__":
    main()