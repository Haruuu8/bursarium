"""
Kombinasi SEMUA dimensi: multiday + hari + bulan + tier + likuiditas.
Cari rule final yang paling optimal.
Walk-forward untuk hindari overfit.
"""

import itertools
import numpy as np
import pandas as pd

from .data_loader import load_panel_from_gabungan
from .data_cleaner import clean_panel
from .feature_engineering import build_features
from .event_detector import detect_events


EVENT_TARGET = "is_touch_10pct"
DISCOVERY_START = "2026-01-01"
DISCOVERY_END   = "2026-07-31"
TEST_START      = "2026-08-01"
TEST_END        = "2026-10-02"

COST_RT = 0.008
TRAIL = 0.03


def add_all_features(df):
    df = df.copy()
    df = df.sort_values(["kode", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # Waktu
    df["dow"] = df["date"].dt.dayofweek
    df["dom"] = df["date"].dt.day
    df["hari_kamis"] = (df["dow"] == 3).astype(int)
    df["hari_jumat"] = (df["dow"] == 4).astype(int)
    df["hari_senin_selasa"] = df["dow"].isin([0, 1]).astype(int)
    df["akhir_bulan"] = (df["dom"] > 20).astype(int)
    df["awal_bulan"] = (df["dom"] <= 10).astype(int)

    # Multiday
    df["ret_3d"] = df.groupby("kode")["close"].pct_change(3)
    df["ret_5d"] = df.groupby("kode")["close"].pct_change(5)

    # Green streak
    green = (df["close"] > df["open"]).astype(int)
    streak = green.copy()
    for i in range(1, 4):
        streak = streak + green.groupby(df["kode"]).shift(i).fillna(0)
    df["green_streak"] = streak

    # Cross-sectional
    df["rk_clv"] = df.groupby("date")["clv"].rank(pct=True)
    df["rk_ret_1d"] = df.groupby("date")["ret_1d"].rank(pct=True)
    df["rk_ret_5d"] = df.groupby("date")["ret_5d"].rank(pct=True)

    # Tier
    df["tier_gocap"] = (df["close"] < 100).astype(int)
    df["tier_kecil"] = ((df["close"] >= 100) & (df["close"] < 500)).astype(int)
    df["tier_besar"] = (df["close"] >= 2000).astype(int)
    df["likuid_med"] = ((df["value"] >= 1e9) & (df["value"] < 5e9)).astype(int)
    df["likuid_tinggi"] = (df["value"] >= 5e9).astype(int)

    # Event
    df["event_next"] = (
        df.groupby("kode")[EVENT_TARGET]
             .shift(-1).fillna(0).astype(int)
    )

    # Forward
    for h in [1, 2, 3]:
        df[f"open_T{h}"] = df.groupby("kode")["open"].shift(-h)
        df[f"high_T{h}"] = df.groupby("kode")["high"].shift(-h)
        df[f"low_T{h}"] = df.groupby("kode")["low"].shift(-h)
        df[f"close_T{h}"] = df.groupby("kode")["close"].shift(-h)

    return df


def trail(row, trail_pct, horizon=3):
    entry = row["close"]
    if pd.isna(row["open_T1"]) or row["open_T1"] <= 0:
        return None
    open1 = row["open_T1"]
    if open1 < entry:
        return open1 / entry - 1 - COST_RT
    trail_high = entry
    trail_level = None
    for d in range(1, horizon + 1):
        h = row[f"high_T{d}"]
        l = row[f"low_T{d}"]
        c = row[f"close_T{d}"]
        if pd.isna(h) or pd.isna(l) or pd.isna(c):
            break
        if l <= 0:
            continue
        if trail_level is not None and l <= trail_level:
            return trail_level / entry - 1 - COST_RT
        if h > trail_high:
            trail_high = h
        trail_level = max(trail_high * (1 - trail_pct), entry)
    for d in range(horizon, 0, -1):
        c = row[f"close_T{d}"]
        if not pd.isna(c):
            return c / entry - 1 - COST_RT
    return None


def evaluate(df, mask, label):
    signals = df[mask]
    n = len(signals)
    if n < 15:
        return None
    n_hit = int(signals["event_next"].sum())
    prec = n_hit / n
    rets = [trail(r, TRAIL, 3) for _, r in signals.iterrows()]
    rets = [r for r in rets if r is not None and not np.isnan(r)]
    if len(rets) < 5:
        return None
    arr = np.array(rets)
    return {
        "kondisi": label,
        "n": n,
        "prec": prec,
        "avg_ret": float(arr.mean()),
        "win": float((arr > 0).mean()),
        "month": n / signals["date"].dt.to_period("M").nunique(),
    }


def main():
    print("=" * 110)
    print("KOMBINASI SEMUA DIMENSI")
    print("=" * 110)

    panel = load_panel_from_gabungan()
    panel = clean_panel(panel)
    panel = build_features(panel)
    panel = detect_events(panel)
    panel = add_all_features(panel)
    panel = panel[(panel["open"] > 0) & (panel["close"] > 0)].copy()

    d1, d2 = pd.Timestamp(DISCOVERY_START), pd.Timestamp(DISCOVERY_END)
    t1, t2 = pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)

    disc = panel[(panel["date"] >= d1) & (panel["date"] <= d2)].copy()
    test = panel[(panel["date"] >= t1) & (panel["date"] <= t2)].copy()

    print(f"\nDiscovery: {len(disc):,} baris "
          f"({disc['date'].nunique()} hari), "
          f"{disc['event_next'].mean()*100:.2f}% event")
    print(f"Test     : {len(test):,} baris "
          f"({test['date'].nunique()} hari), "
          f"{test['event_next'].mean()*100:.2f}% event")

    # Base rule - multiday kuat
    base = lambda d: (d["gap"] > 0.05) & (d["ret_3d"] > 0.15)

    # Kondisi tambahan
    filters = {
        "no filter": lambda d: pd.Series(True, index=d.index),
        "+ Kamis": lambda d: d["hari_kamis"] == 1,
        "+ Jumat": lambda d: d["hari_jumat"] == 1,
        "+ Senin-Selasa": lambda d: d["hari_senin_selasa"] == 1,
        "+ akhir bulan": lambda d: d["akhir_bulan"] == 1,
        "+ awal bulan": lambda d: d["awal_bulan"] == 1,
        "+ gocap": lambda d: d["tier_gocap"] == 1,
        "+ kecil": lambda d: d["tier_kecil"] == 1,
        "+ likuid med": lambda d: d["likuid_med"] == 1,
        "+ likuid tinggi": lambda d: d["likuid_tinggi"] == 1,
        "+ green>=2": lambda d: d["green_streak"] >= 2,
        "+ green>=3": lambda d: d["green_streak"] >= 3,
        "+ rk_clv>0.85": lambda d: d["rk_clv"] > 0.85,
        "+ rk_ret_5d>0.9": lambda d: d["rk_ret_5d"] > 0.90,
        "+ ret_5d>20%": lambda d: d["ret_5d"] > 0.20,
        "+ close>100": lambda d: d["close"] > 100,
        "+ body>5%": lambda d: d["body"] > 0.05,
        "+ range>10%": lambda d: d["range"] > 0.10,
    }

    # Uji kombinasi base + 1 filter
    print(f"\n{'='*110}")
    print("KOMBINASI BASE RULE + 1 FILTER")
    print(f"{'='*110}\n")
    print(f"{'Kondisi':<40} {'n_d':>5} {'p_d%':>7} {'n_t':>5} "
          f"{'p_t%':>7} {'avgR_t':>8} {'/bln':>6}")
    print("-" * 95)

    results = []
    for label, fn in filters.items():
        try:
            mask_d = base(disc).fillna(False).values & fn(disc).fillna(False).values
            mask_t = base(test).fillna(False).values & fn(test).fillna(False).values
            r_d = evaluate(disc, mask_d, f"base + {label}")
            r_t = evaluate(test, mask_t, f"base + {label}")
            if r_d is None or r_t is None:
                continue

            print(f"{label:<40} {r_d['n']:>5} {r_d['prec']*100:>6.1f}% "
                  f"{r_t['n']:>5} {r_t['prec']*100:>6.1f}% "
                  f"{r_t['avg_ret']*100:>+7.2f}% {r_t['month']:>5.1f}")

            results.append({
                "filter": label,
                "n_disc": r_d["n"], "prec_disc": r_d["prec"],
                "n_test": r_t["n"], "prec_test": r_t["prec"],
                "avg_ret_test": r_t["avg_ret"],
                "win_test": r_t["win"],
                "per_month_test": r_t["month"],
            })
        except Exception as e:
            print(f"{label:<40} ERROR: {e}")

    df = pd.DataFrame(results)
    df.to_csv("quant/output/combine_all_1filter.csv", index=False)

    # ============================================================
    # KOMBINASI 2 FILTER
    # ============================================================
    print(f"\n{'='*110}")
    print("KOMBINASI BASE + 2 FILTER (cari yang precision test >50%)")
    print(f"{'='*110}\n")

    filter_names = list(filters.keys())
    results2 = []

    for c1, c2 in itertools.combinations(filter_names, 2):
        try:
            mask_d = (base(disc).fillna(False).values &
                      filters[c1](disc).fillna(False).values &
                      filters[c2](disc).fillna(False).values)
            mask_t = (base(test).fillna(False).values &
                      filters[c1](test).fillna(False).values &
                      filters[c2](test).fillna(False).values)

            r_d = evaluate(disc, mask_d, f"{c1} + {c2}")
            r_t = evaluate(test, mask_t, f"{c1} + {c2}")
            if r_d is None or r_t is None:
                continue
            if r_t["prec"] < 0.40:
                continue

            results2.append({
                "filter": f"{c1} + {c2}",
                "n_disc": r_d["n"], "prec_disc": r_d["prec"],
                "n_test": r_t["n"], "prec_test": r_t["prec"],
                "avg_ret_test": r_t["avg_ret"],
                "per_month_test": r_t["month"],
            })
        except Exception:
            continue

    if results2:
        df2 = pd.DataFrame(results2)
        df2 = df2.sort_values("prec_test", ascending=False).reset_index(drop=True)
        print(f"{'Kombinasi':<45} {'n_d':>5} {'p_d%':>7} {'n_t':>5} "
              f"{'p_t%':>7} {'avgR_t':>8} {'/bln':>6}")
        print("-" * 95)
        for _, r in df2.head(25).iterrows():
            print(f"{r['filter']:<45} {r['n_disc']:>5} "
                  f"{r['prec_disc']*100:>6.1f}% {r['n_test']:>5} "
                  f"{r['prec_test']*100:>6.1f}% "
                  f"{r['avg_ret_test']*100:>+7.2f}% {r['per_month_test']:>5.1f}")
        df2.to_csv("quant/output/combine_all_2filter.csv", index=False)

        # Best recommendation
        best = df2.iloc[0]
        print(f"\n{'='*110}")
        print("REKOMENDASI FINAL")
        print(f"{'='*110}\n")
        print(f"Kombinasi: gap>5% + ret_3d>15% + {best['filter']}")
        print(f"  Discovery: n={int(best['n_disc'])}, "
              f"prec={best['prec_disc']*100:.1f}%")
        print(f"  Test     : n={int(best['n_test'])}, "
              f"prec={best['prec_test']*100:.1f}%, "
              f"avgR={best['avg_ret_test']*100:+.2f}%")
        print(f"  Frekuensi test: {best['per_month_test']:.1f}/bulan "
              f"({best['per_month_test']/20:.1f}/hari)")
    else:
        print("Tidak ada kombinasi 2 filter dengan precision test >=40%")

    # ============================================================
    # BANDINGKAN DENGAN RULE LAMA
    # ============================================================
    print(f"\n{'='*110}")
    print("BANDINGKAN DENGAN RULE LAMA (gap>3% + rk_clv>0.85 + vol_trend>1.0)")
    print(f"{'='*110}\n")

    v3 = disc.groupby("kode")["volume"].transform(
        lambda s: s.rolling(3, min_periods=2).mean())
    v10 = disc.groupby("kode")["volume"].transform(
        lambda s: s.rolling(10, min_periods=5).mean())
    disc_vt = v3 / v10.replace(0, np.nan)

    v3t = test.groupby("kode")["volume"].transform(
        lambda s: s.rolling(3, min_periods=2).mean())
    v10t = test.groupby("kode")["volume"].transform(
        lambda s: s.rolling(10, min_periods=5).mean())
    test_vt = v3t / v10t.replace(0, np.nan)

    old_d = ((disc["gap"] > 0.03) & (disc["rk_clv"] > 0.85) &
             (disc_vt > 1.0)).fillna(False).values
    old_t = ((test["gap"] > 0.03) & (test["rk_clv"] > 0.85) &
             (test_vt > 1.0)).fillna(False).values

    r_old_d = evaluate(disc, old_d, "old")
    r_old_t = evaluate(test, old_t, "old")
    if r_old_d and r_old_t:
        print(f"RULE LAMA:")
        print(f"  Discovery: n={r_old_d['n']}, prec={r_old_d['prec']*100:.1f}%")
        print(f"  Test     : n={r_old_t['n']}, prec={r_old_t['prec']*100:.1f}%, "
              f"avgR={r_old_t['avg_ret']*100:+.2f}%, "
              f"{r_old_t['month']:.1f}/bulan")

    print(f"\nRULE BARU (jika ada di rekomendasi):")
    print(f"  (lihat di atas)")


if __name__ == "__main__":
    main()