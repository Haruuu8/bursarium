"""
Analisis MULTI-HARI: pola H-1 + H-2 + H-3, bukan hanya H-1.
"""

import numpy as np
import pandas as pd

from .data_loader import load_panel_from_gabungan
from .data_cleaner import clean_panel
from .feature_engineering import build_features
from .event_detector import detect_events


EVENT_TARGET = "is_touch_10pct"


def add_multiday_features(df):
    df = df.copy()
    df = df.sort_values(["kode", "date"]).reset_index(drop=True)

    # Volume trend 3 hari
    v1 = df["volume"]
    v2 = df.groupby("kode")["volume"].shift(1)
    v3 = df.groupby("kode")["volume"].shift(2)
    df["vol_up_3d"] = ((v1 > v2) & (v2 > v3)).astype(int)
    df["vol_down_3d"] = ((v1 < v2) & (v2 < v3)).astype(int)

    # Close trend 3 hari
    c1 = df["close"]
    c2 = df.groupby("kode")["close"].shift(1)
    c3 = df.groupby("kode")["close"].shift(2)
    df["close_up_3d"] = ((c1 > c2) & (c2 > c3)).astype(int)
    df["close_down_3d"] = ((c1 < c2) & (c2 < c3)).astype(int)

    # Green candle 3 hari
    g1 = (df["close"] > df["open"]).astype(int)
    g2 = g1.groupby(df["kode"]).shift(1).fillna(0)
    g3 = g1.groupby(df["kode"]).shift(2).fillna(0)
    df["all_green_3d"] = ((g1 == 1) & (g2 == 1) & (g3 == 1)).astype(int)

    # Volume 3 hari semua di atas MA20
    vol_ma20 = df.groupby("kode")["volume"].transform(
        lambda s: s.rolling(20, min_periods=5).mean())
    v1_above = (df["volume"] > vol_ma20).astype(int)
    v2_above = v1_above.groupby(df["kode"]).shift(1).fillna(0)
    v3_above = v1_above.groupby(df["kode"]).shift(2).fillna(0)
    df["vol_high_3d"] = ((v1_above == 1) & (v2_above == 1) & (v3_above == 1)).astype(int)

    # Ret 3 hari berturut positif
    r1 = (df["ret_1d"] > 0).astype(int)
    r2 = r1.groupby(df["kode"]).shift(1).fillna(0)
    r3 = r1.groupby(df["kode"]).shift(2).fillna(0)
    df["ret_pos_3d"] = ((r1 == 1) & (r2 == 1) & (r3 == 1)).astype(int)

    # Green streak count
    streak_g = g1.copy()
    for i in range(1, 4):
        streak_g = streak_g + g1.groupby(df["kode"]).shift(i).fillna(0)
    df["green_streak_4"] = streak_g

    # Volume acceleration
    df["vol_accel"] = (v1 - v2) / v2.replace(0, np.nan)

    # Close vs 3 hari lalu
    df["ret_3d_total"] = df["close"] / c3 - 1

    # Event
    df["event_next"] = (
        df.groupby("kode")[EVENT_TARGET]
             .shift(-1).fillna(0).astype(int)
    )

    return df


def evaluate(df, mask, label):
    signals = df[mask]
    n = len(signals)
    if n < 20:
        return None
    n_hit = int(signals["event_next"].sum())
    prec = n_hit / n
    signals_copy = signals.copy()
    signals_copy["month"] = pd.to_datetime(signals_copy["date"]).dt.to_period("M")
    per_month = n / signals_copy["month"].nunique()
    return {
        "kondisi": label, "n": n, "per_month": per_month,
        "prec": prec,
        "lift": prec / df["event_next"].mean() if df["event_next"].mean() > 0 else 0,
    }


def main():
    print("=" * 100)
    print("ANALISIS MULTI-HARI (H-1, H-2, H-3)")
    print("=" * 100)

    panel = load_panel_from_gabungan()
    panel = clean_panel(panel)
    panel = build_features(panel)
    panel = detect_events(panel)
    panel = add_multiday_features(panel)
    panel = panel[(panel["open"] > 0) & (panel["close"] > 0)].copy()

    baseline = panel["event_next"].mean()
    print(f"\nTotal: {len(panel):,}, baseline {baseline*100:.2f}%")

    conditions = {
        # Trend 3 hari
        "vol_up_3d":            lambda d: d["vol_up_3d"] == 1,
        "close_up_3d":          lambda d: d["close_up_3d"] == 1,
        "all_green_3d":         lambda d: d["all_green_3d"] == 1,
        "vol_high_3d":          lambda d: d["vol_high_3d"] == 1,
        "ret_pos_3d":           lambda d: d["ret_pos_3d"] == 1,

        # Green streak
        "green_streak_4 >= 2":  lambda d: d["green_streak_4"] >= 2,
        "green_streak_4 >= 3":  lambda d: d["green_streak_4"] >= 3,
        "green_streak_4 >= 4":  lambda d: d["green_streak_4"] >= 4,

        # Kombinasi H-1 dengan multiday
        "gap>5% + vol_up_3d":   lambda d: (d["gap"] > 0.05) & (d["vol_up_3d"] == 1),
        "gap>5% + close_up_3d": lambda d: (d["gap"] > 0.05) & (d["close_up_3d"] == 1),
        "gap>5% + all_green_3d": lambda d: (d["gap"] > 0.05) & (d["all_green_3d"] == 1),
        "gap>5% + green_4>=3":  lambda d: (d["gap"] > 0.05) & (d["green_streak_4"] >= 3),
        "clv>0.85 + vol_up_3d": lambda d: (d["clv"] > 0.85) & (d["vol_up_3d"] == 1),
        "clv>0.85 + green_4>=3": lambda d: (d["clv"] > 0.85) & (d["green_streak_4"] >= 3),

        # Gap + momentum
        "gap>5% + ret_3d>10%":  lambda d: (d["gap"] > 0.05) & (d["ret_3d_total"] > 0.10),
        "gap>5% + ret_3d>15%":  lambda d: (d["gap"] > 0.05) & (d["ret_3d_total"] > 0.15),

        # Volume acceleration
        "vol_accel > 0.5":      lambda d: d["vol_accel"] > 0.5,
        "vol_accel > 1.0":      lambda d: d["vol_accel"] > 1.0,
        "vol_accel > 2.0":      lambda d: d["vol_accel"] > 2.0,

        # Kombinasi gap + volume accel
        "gap>5% + vol_accel>1": lambda d: (d["gap"] > 0.05) & (d["vol_accel"] > 1.0),
    }

    print(f"\n{'Kondisi':<32} {'n':>6} {'/bln':>7} {'prec%':>8} {'lift':>7}")
    print("-" * 65)

    results = []
    for label, fn in conditions.items():
        try:
            mask = fn(panel).fillna(False).values
            r = evaluate(panel, mask, label)
            if r is None:
                continue
            results.append(r)
            print(f"{r['kondisi']:<32} {r['n']:>6} {r['per_month']:>6.1f} "
                  f"{r['prec']*100:>7.2f}% {r['lift']:>6.2f}x")
        except Exception as e:
            print(f"{label:<32} ERROR: {e}")

    df_res = pd.DataFrame(results).sort_values("prec", ascending=False)
    df_res.to_csv("quant/output/method_multiday.csv", index=False)
    print(f"\n-> Disimpan: quant/output/method_multiday.csv")


if __name__ == "__main__":
    main()