"""
Analisis FOREIGN FLOW: apakah foreign net buy di H-1 prediktif?
Data foreign_buy & foreign_sell sudah tersedia dari scrape IDX.
"""

import numpy as np
import pandas as pd

from .data_loader import load_panel_from_gabungan
from .data_cleaner import clean_panel
from .feature_engineering import build_features
from .event_detector import detect_events


EVENT_TARGET = "is_touch_10pct"


def add_foreign_features(df):
    df = df.copy()
    df = df.sort_values(["kode", "date"]).reset_index(drop=True)

    # Foreign net (rupiah)
    df["foreign_net"] = df["foreign_buy"] - df["foreign_sell"]
    df["foreign_net_pct"] = df["foreign_net"] / df["value"].replace(0, np.nan)

    # Rank cross-sectional
    df["rk_foreign_net"] = df.groupby("date")["foreign_net"].rank(pct=True)
    df["rk_foreign_net_pct"] = df.groupby("date")["foreign_net_pct"].rank(pct=True)

    # Streak: berapa hari berturut foreign net buy
    fn_pos = (df["foreign_net"] > 0).astype(int)
    streak = fn_pos.copy()
    for i in range(1, 3):
        streak = streak + fn_pos.groupby(df["kode"]).shift(i).fillna(0)
    df["foreign_buy_streak"] = streak

    # Foreign net total 3 hari
    df["foreign_net_3d"] = df.groupby("kode")["foreign_net"].transform(
        lambda s: s.rolling(3, min_periods=2).sum())
    df["foreign_net_5d"] = df.groupby("kode")["foreign_net"].transform(
        lambda s: s.rolling(5, min_periods=3).sum())

    # Foreign buy/sell ratio
    df["foreign_buy_ratio"] = df["foreign_buy"] / (
        df["foreign_buy"] + df["foreign_sell"]).replace(0, np.nan)

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

    # Frekuensi per bulan
    signals_copy = signals.copy()
    signals_copy["month"] = pd.to_datetime(signals_copy["date"]).dt.to_period("M")
    per_month = n / signals_copy["month"].nunique()

    return {
        "kondisi": label,
        "n": n,
        "per_month": per_month,
        "prec": prec,
        "lift": prec / df["event_next"].mean() if df["event_next"].mean() > 0 else 0,
    }


def main():
    print("=" * 100)
    print("ANALISIS FOREIGN FLOW")
    print("=" * 100)

    panel = load_panel_from_gabungan()
    panel = clean_panel(panel)
    panel = build_features(panel)
    panel = detect_events(panel)
    panel = add_foreign_features(panel)

    # Filter data error
    panel = panel[(panel["open"] > 0) & (panel["close"] > 0)].copy()

    n_days = panel["date"].nunique()
    baseline = panel["event_next"].mean()
    print(f"\nTotal: {len(panel):,} baris, {n_days} hari")
    print(f"Baseline event: {baseline*100:.2f}%")

    conditions = {
        # Foreign net absolut
        "foreign_net > 0":              lambda d: d["foreign_net"] > 0,
        "foreign_net > 1M":             lambda d: d["foreign_net"] > 1e9,
        "foreign_net > 5M":             lambda d: d["foreign_net"] > 5e9,
        "foreign_net > 10M":            lambda d: d["foreign_net"] > 1e10,
        "foreign_net < 0":              lambda d: d["foreign_net"] < 0,

        # Foreign net %
        "foreign_net_pct > 5%":         lambda d: d["foreign_net_pct"] > 0.05,
        "foreign_net_pct > 10%":        lambda d: d["foreign_net_pct"] > 0.10,
        "foreign_net_pct > 20%":        lambda d: d["foreign_net_pct"] > 0.20,
        "foreign_net_pct < -5%":        lambda d: d["foreign_net_pct"] < -0.05,

        # Cross-sectional rank
        "rk_foreign_net > 0.90":        lambda d: d["rk_foreign_net"] > 0.90,
        "rk_foreign_net > 0.95":        lambda d: d["rk_foreign_net"] > 0.95,
        "rk_foreign_net_pct > 0.90":    lambda d: d["rk_foreign_net_pct"] > 0.90,
        "rk_foreign_net_pct > 0.95":    lambda d: d["rk_foreign_net_pct"] > 0.95,

        # Streak
        "foreign_buy_streak >= 2":      lambda d: d["foreign_buy_streak"] >= 2,
        "foreign_buy_streak >= 3":      lambda d: d["foreign_buy_streak"] >= 3,

        # Multi-day foreign
        "foreign_net_3d > 5M":          lambda d: d["foreign_net_3d"] > 5e9,
        "foreign_net_3d > 10M":         lambda d: d["foreign_net_3d"] > 1e10,
        "foreign_net_5d > 10M":         lambda d: d["foreign_net_5d"] > 1e10,

        # Kombinasi foreign + price
        "foreign_net > 5M + gap > 3%":  lambda d: (d["foreign_net"] > 5e9) & (d["gap"] > 0.03),
        "foreign_net > 5M + clv > 0.7": lambda d: (d["foreign_net"] > 5e9) & (d["clv"] > 0.7),
        "rk_foreign > 0.90 + ret > 5%": lambda d: (d["rk_foreign_net"] > 0.90) & (d["ret_1d"] > 0.05),

        # Foreign SELLING (untuk cek apakah reversal)
        "foreign_net < -5M":            lambda d: d["foreign_net"] < -5e9,
        "rk_foreign_net < 0.10":        lambda d: d["rk_foreign_net"] < 0.10,
    }

    print(f"\n{'Kondisi':<40} {'n':>6} {'/bln':>7} {'prec%':>8} {'lift':>7}")
    print("-" * 75)

    results = []
    for label, fn in conditions.items():
        try:
            mask = fn(panel).fillna(False).values
            r = evaluate(panel, mask, label)
            if r is None:
                continue
            results.append(r)
            print(f"{r['kondisi']:<40} {r['n']:>6} {r['per_month']:>6.1f} "
                  f"{r['prec']*100:>7.2f}% {r['lift']:>6.2f}x")
        except Exception as e:
            print(f"{label:<40} ERROR: {e}")

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("prec", ascending=False).reset_index(drop=True)
    df_res.to_csv("quant/output/method_foreign.csv", index=False)
    print(f"\n-> Disimpan: quant/output/method_foreign.csv")

    # Best
    if not df_res.empty:
        best = df_res.iloc[0]
        print(f"\nFOREIGN FLOW TERBAIK:")
        print(f"  Kondisi: {best['kondisi']}")
        print(f"  Precision: {best['prec']*100:.1f}%")
        print(f"  Frekuensi: {best['per_month']:.1f}/bulan")


if __name__ == "__main__":
    main()