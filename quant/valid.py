"""
Validasi rule baru dari dimensi turnover, fibonacci, keltner, elder ray.
Walk-forward 5 periode.
"""

import numpy as np
import pandas as pd

from .data_loader import load_panel_from_gabungan
from .data_cleaner import clean_panel
from .feature_engineering import build_features
from .event_detector import detect_events


COST_RT = 0.008


def add_features(df):
    df = df.copy()
    df = df.sort_values(["kode", "date"]).reset_index(drop=True)

    # Turnover
    if "listed_shares" in df.columns:
        df["turnover"] = df["volume"] / df["listed_shares"].replace(0, np.nan)
        df["rk_turnover"] = df.groupby("date")["turnover"].rank(pct=True)
    else:
        df["rk_turnover"] = 0

    # Fibonacci position
    df["swing_high_20"] = df.groupby("kode")["high"].transform(
        lambda s: s.rolling(20, min_periods=5).max())
    df["swing_low_20"] = df.groupby("kode")["low"].transform(
        lambda s: s.rolling(20, min_periods=5).min())
    sw_range = (df["swing_high_20"] - df["swing_low_20"]).replace(0, np.nan)
    df["fib_pos"] = (df["close"] - df["swing_low_20"]) / sw_range

    # Elder Ray
    ema13 = df.groupby("kode")["close"].transform(
        lambda s: s.ewm(span=13, adjust=False).mean())
    df["bull_power"] = df["high"] - ema13

    # Keltner
    ema20 = df.groupby("kode")["close"].transform(
        lambda s: s.ewm(span=20, adjust=False).mean())
    atr10 = df.groupby("kode")["tr"].transform(
        lambda s: s.rolling(10, min_periods=3).mean())
    df["keltner_pos"] = (df["close"] - ema20) / (2 * atr10).replace(0, np.nan)

    # Vol regime
    df["vol_regime_high"] = (df["atr_pct_rank_60"] > 0.8).astype(int)

    # Forward
    df["open_fwd"] = df.groupby("kode")["open"].shift(-1)
    df["high_fwd"] = df.groupby("kode")["high"].shift(-1)

    return df


# Kandidat rule untuk validasi
RULES = {
    "LAMA: ret_1d>15 + ret_5d>15 + px>20%": lambda d: (
        (d["ret_1d"] > 0.15) & (d["ret_5d"] > 0.15) & (d["px_vs_sma20"] > 0.20)
    ),
    "ret_1d>15 + px>20% + turnover>0.90": lambda d: (
        (d["ret_1d"] > 0.15) & (d["px_vs_sma20"] > 0.20) &
        (d["rk_turnover"] > 0.90)
    ),
    "ret_1d>15 + px>20%": lambda d: (
        (d["ret_1d"] > 0.15) & (d["px_vs_sma20"] > 0.20)
    ),
    "ret_1d>15 + turnover>0.90 + fib>0.8": lambda d: (
        (d["ret_1d"] > 0.15) & (d["rk_turnover"] > 0.90) &
        (d["fib_pos"] > 0.8)
    ),
    "ret_5d>15 + turnover>0.90 + vol_high": lambda d: (
        (d["ret_5d"] > 0.15) & (d["rk_turnover"] > 0.90) &
        (d["vol_regime_high"] == 1)
    ),
    "ret_1d>15 + fib>0.8 + bull_power>0.05": lambda d: (
        (d["ret_1d"] > 0.15) & (d["fib_pos"] > 0.8) &
        (d["bull_power"] > 0.05 * d["close"])
    ),
}


def evaluate(df, mask):
    sig = df[mask]
    if len(sig) < 5:
        return None

    close_arr = sig["close"].values
    open_arr = sig["open_fwd"].values
    high_arr = sig["high_fwd"].values

    valid = ~(np.isnan(open_arr) | np.isnan(high_arr))
    valid &= open_arr > 0
    valid &= open_arr >= close_arr

    if valid.sum() < 5:
        return None

    ret = high_arr[valid] / open_arr[valid] - 1 - COST_RT
    n_days = df["date"].nunique()

    return {
        "n_exec": int(valid.sum()),
        "per_day": valid.sum() / n_days,
        "avg_ret": float(ret.mean()),
        "win": float((ret > 0).mean()),
        "best": float(ret.max()),
        "worst": float(ret.min()),
    }


def main():
    print("=" * 110)
    print("VALIDASI 5 PERIODE - RULE DIMENSI BARU")
    print("=" * 110)

    panel = load_panel_from_gabungan()
    panel = clean_panel(panel)
    panel = build_features(panel)
    panel = detect_events(panel)
    panel = add_features(panel)
    panel = panel[(panel["open"] > 0) & (panel["close"] > 0)].copy()

    periods = [
        ("Jan-Feb", "2026-01-01", "2026-02-28"),
        ("Mar-Apr", "2026-03-01", "2026-04-30"),
        ("Mei-Jun", "2026-05-01", "2026-06-30"),
        ("Jul-Ags", "2026-07-01", "2026-08-31"),
        ("Sep-Okt", "2026-09-01", "2026-10-02"),
    ]

    for rule_name, rule_fn in RULES.items():
        print(f"\n{'='*110}")
        print(f"{rule_name}")
        print(f"{'='*110}\n")
        print(f"{'Periode':<12} {'n':>5} {'/hari':>7} "
              f"{'avgR%':>8} {'win%':>7} {'best%':>8} {'worst%':>8}")
        print("-" * 70)

        all_avg = []
        all_win = []
        all_pd = []

        for label, start, end in periods:
            sub = panel[(panel["date"] >= pd.Timestamp(start)) &
                        (panel["date"] <= pd.Timestamp(end))].copy()
            mask = rule_fn(sub).fillna(False).values
            r = evaluate(sub, mask)
            if r is None:
                print(f"{label:<12} (sample kecil)")
                continue

            all_avg.append(r["avg_ret"])
            all_win.append(r["win"])
            all_pd.append(r["per_day"])

            print(f"{label:<12} {r['n_exec']:>5} {r['per_day']:>7.1f} "
                  f"{r['avg_ret']*100:>+7.2f}% {r['win']*100:>6.1f}% "
                  f"{r['best']*100:>+7.2f}% {r['worst']*100:>+7.2f}%")

        if all_avg:
            print(f"\n  Rata-rata : avg {np.mean(all_avg)*100:+.2f}%, "
                  f"win {np.mean(all_win)*100:.1f}%, "
                  f"frekuensi {np.mean(all_pd):.1f}/hari")
            print(f"  Minimum   : avg {np.min(all_avg)*100:+.2f}%, "
                  f"win {np.min(all_win)*100:.1f}%")

            if np.min(all_win) >= 0.70 and np.min(all_avg) > 0.03:
                print(f"  [OK] VALID - konsisten di semua periode")
            elif np.min(all_win) >= 0.60 and np.min(all_avg) > 0.01:
                print(f"  [!] CUKUP - agak variatif")
            else:
                print(f"  [X] TIDAK VALID - drop di beberapa periode")


if __name__ == "__main__":
    main()