import pandas as pd

df = pd.read_csv("data_sektor.csv")

cek = ["BBCA", "BSDE", "JSMR", "ASSA", "SMGR", "ANTM",
       "GOTO", "TLKM", "SMRA", "PWON", "TMAS", "BIRD"]

print("=" * 60)
print("CEK SEKTOR SAHAM TERTENTU")
print("=" * 60)

for kode in cek:
    rows = df[df["kode"] == kode]
    if len(rows) > 0:
        sektor = rows["sektor"].values[0]
    else:
        sektor = "tidak ada"
    print(f"  {kode:<7} : {sektor}")

print()
print("=" * 60)
print("DAFTAR SEMUA SEKTOR UNIK")
print("=" * 60)
for s in sorted(df["sektor"].dropna().unique()):
    n = (df["sektor"] == s).sum()
    print(f"  {s:<30} : {n} saham")