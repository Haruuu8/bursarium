"""Pisahkan failed_detail.txt jadi 404 (skip) vs 401 (retry)."""
import os

if not os.path.exists("failed_detail.txt"):
    print("failed_detail.txt tidak ada.")
    exit()

with open("failed_detail.txt") as f:
    all_failed = [l.strip() for l in f if l.strip()]

# Kode yang sudah kelihatan 404 di log (skip permanen)
known_404 = [
    "CNTX", "CBMF", "BTEL", "ARMY", "COWL", "CPRI", "HDTX",
    "HKMU", "HOME", "IIKP", "JKSW", "HOTL", "KBRI", "JSKY",
    "LCGP", "KPAS", "KRAH", "KPAL", "LMAS",
]

retry_list = [k for k in all_failed if k not in known_404]
skip_list = [k for k in all_failed if k in known_404]

with open("failed_retry.txt", "w") as f:
    f.write("\n".join(retry_list))

with open("failed_404.txt", "w") as f:
    f.write("\n".join(skip_list))

print(f"Total gagal   : {len(all_failed)}")
print(f"Perlu retry   : {len(retry_list)} -> failed_retry.txt")
print(f"Skip (404)    : {len(skip_list)} -> failed_404.txt")