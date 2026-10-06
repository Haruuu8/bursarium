"""
Analisis HARI dan BULAN: apakah ada hari/bulan tertentu yang lebih prediktif?
"""

import numpy as np
import pandas as pd

from .data_loader import load_panel_from_gabungan
from .data_cleaner import clean_panel
from .feature_engineering import build_features
from .event_detector import detect_events


EVENT_TARGET = "is_touch_10pct"


def add_time_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.dayofweek  # 0=Senin, 4=Jumat
    df["dom"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["week_of_month"] = ((df["dom"] - 1) // 7) + 1

    # Kategori
    df["hari_senin"] = (df["dow"] == 0).astype(int)
    df["hari_selasa"] = (df["dow"] == 1).astype(int)
    df["hari_rabu"] = (df["dow"] == 2).astype(int)
    df["hari_kamis"] = (df["dow"] == 3).astype(int)
    df["hari_jumat"] = (df["dow"] == 4).astype(int)

    df["awal_bulan"] = (df["dom"] <= 10).astype(int)
    df["tengah_bulan"] = ((df["dom"] > 10) & (df["dom"] <= 20)).astype(int)
    df["akhir_bulan"] = (df["dom"] > 20).astype(int)

    df["event_next"] = (
        df.groupby("kode")[EVENT_TARGET]
             .shift(-1).fillna(0).astype(int)
    )
    return df


def main():
    print("=" * 100)
    print("ANALISIS HARI & BULAN")
    print("=" * 100)

    panel = load_panel_from_gabungan()
    panel = clean_panel(panel)
    panel = build_features(panel)
    panel = detect_events(panel)
    panel = add_time_features(panel)
    panel = panel[(panel["open"] > 0) & (panel["close"] > 0)].copy()

    baseline = panel["event_next"].mean()
    print(f"\nTotal: {len(panel):,}, baseline {baseline*100:.2f}%")

    # A. Per hari
    print(f"\n{'='*100}")
    print("A. PER HARI")
    print(f"{'='*100}\n")
    print(f"{'Hari':<10} {'n':>7} {'prec%':>8} {'lift':>7}")
    print("-" * 40)
    for dow, nama in enumerate(["Senin", "Selasa", "Rabu", "Kamis", "Jumat"]):
        sub = panel[panel["dow"] == dow]
        if len(sub) == 0:
            continue
        prec = sub["event_next"].mean()
        print(f"{nama:<10} {len(sub):>7} {prec*100:>7.2f}% "
              f"{prec/baseline:>6.2f}x")

    # B. Per minggu dalam bulan
    print(f"\n{'='*100}")
    print("B. PER MINGGU DALAM BULAN")
    print(f"{'='*100}\n")
    print(f"{'Minggu':<12} {'n':>7} {'prec%':>8} {'lift':>7}")
    print("-" * 45)
    for w in [1, 2, 3, 4, 5]:
        sub = panel[panel["week_of_month"] == w]
        if len(sub) < 100:
            continue
        prec = sub["event_next"].mean()
        print(f"Minggu {w:<5} {len(sub):>7} {prec*100:>7.2f}% "
              f"{prec/baseline:>6.2f}x")

    # C. Per bulan
    print(f"\n{'='*100}")
    print("C. PER BULAN")
    print(f"{'='*100}\n")
    print(f"{'Bulan':<10} {'n':>7} {'prec%':>8} {'lift':>7}")
    print("-" * 40)
    for m in sorted(panel["month"].unique()):
        sub = panel[panel["month"] == m]
        if len(sub) < 100:
            continue
        prec = sub["event_next"].mean()
        print(f"Bulan {m:<4} {len(sub):>7} {prec*100:>7.2f}% "
              f"{prec/baseline:>6.2f}x")

    # D. Kombinasi dengan rule utama
    print(f"\n{'='*100}")
    print("D. KOMBINASI DENGAN RULE UTAMA")
    print(f"{'='*100}\n")

    # Rule multiday yang kuat
    base_rule = lambda d: (d["gap"] > 0.05) & (d["ret_3d"] > 0.15)
    # ^ gap>5% + ret_3d>15%

    # Kalau belum ada ret_3d, hitung
    if "ret_3d" not in panel.columns:
        panel["ret_3d"] = panel.groupby("kode")["close"].pct_change(3)

    base_mask = base_rule(panel).fillna(False).values
    sub_base = panel[base_mask]
    print(f"Rule dasar (gap>5% + ret_3d>15%):")
    print(f"  n={len(sub_base)}, prec={sub_base['event_next'].mean()*100:.1f}%")

    print(f"\nKombinasi dengan hari:")
    print(f"{'Kondisi':<45} {'n':>6} {'prec%':>8} {'lift':>7}")
    print("-" * 70)

    untuk_dites = [
        ("hari Senin", (panel["dow"] == 0).values),
        ("hari Selasa", (panel["dow"] == 1).values),
        ("hari Rabu", (panel["dow"] == 2).values),
        ("hari Kamis", (panel["dow"] == 3).values),
        ("hari Jumat", (panel["dow"] == 4).values),
        ("awal bulan (1-10)", (panel["dom"] <= 10).values),
        ("tengah bulan (11-20)", ((panel["dom"] > 10) & (panel["dom"] <= 20)).values),
        ("akhir bulan (21+)", (panel["dom"] > 20).values),
    ]

    for label, day_mask in untuk_dites:
        m = base_mask & day_mask
        n = int(m.sum())
        if n < 10:
            continue
        prec = panel.loc[m, "event_next"].mean()
        print(f"gap>5% + ret_3d>15% + {label:<20} {n:>6} "
              f"{prec*100:>7.2f}% {prec/baseline:>6.2f}x")


if __name__ == "__main__":
    main()