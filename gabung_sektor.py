"""Gabung semua file CSV sektor jadi satu data_sektor.csv."""
import os
import pandas as pd


SEKTOR_DIR = "sektor_csv"
OUTPUT = "data_sektor.csv"


def main():
    if not os.path.isdir(SEKTOR_DIR):
        print(f"Folder {SEKTOR_DIR} tidak ada.")
        return

    files = [f for f in os.listdir(SEKTOR_DIR) if f.endswith(".csv")]
    print(f"File ditemukan: {len(files)}")

    rows = []
    for fname in files:
        # Nama sektor = nama file tanpa .csv
        sektor = fname.replace(".csv", "").strip()
        filepath = os.path.join(SEKTOR_DIR, fname)

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"  Gagal baca {fname}: {e}")
            continue

        # Deteksi kolom kode
        kolom_kode = None
        for k in ["Kode", "KodeEmiten", "Code", "code", "kode", "StockCode",
                  "Kode Saham", "Symbol"]:
            if k in df.columns:
                kolom_kode = k
                break

        if kolom_kode is None:
            print(f"  Kolom kode tidak ditemukan di {fname}")
            print(f"  Kolom tersedia: {list(df.columns)}")
            continue

        kode_list = df[kolom_kode].dropna().astype(str).str.upper().str.strip()

        for k in kode_list:
            rows.append({"kode": k, "sektor": sektor})

        print(f"  {sektor}: {len(kode_list)} saham")

    df_all = pd.DataFrame(rows)
    df_all = df_all.drop_duplicates(subset=["kode"], keep="first")
    df_all = df_all.sort_values("kode").reset_index(drop=True)

    df_all.to_csv(OUTPUT, index=False)

    print(f"\n{'='*60}")
    print(f"Total saham: {len(df_all)}")
    print(f"-> {OUTPUT}")
    print()

    # Distribusi
    print("Distribusi sektor:")
    for s, n in df_all["sektor"].value_counts().items():
        print(f"  {s:<30} : {n} saham")


if __name__ == "__main__":
    main()