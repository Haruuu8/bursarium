"""
Pattern mining: cari kombinasi kondisi yang paling membedakan top gainer
dari saham biasa.

Pendekatan:
1. Definisikan pool kondisi (single feature threshold)
2. Generate semua kombinasi 1-3 kondisi
3. Untuk setiap kombinasi, hitung:
   - Sample size (n)
   - Hit rate in-sample & out-of-sample
   - Lift vs baseline
   - Bootstrap CI
   - Average forward return
4. Rank berdasarkan performa out-of-sample, bukan in-sample
"""

import itertools
import numpy as np
import pandas as pd

from .data_loader import load_panel_from_gabungan
from .data_cleaner import clean_panel
from .feature_engineering import build_features
from .event_detector import detect_events


# ============================================================
# KONFIGURASI
# ============================================================
TARGET = "is_up_10pct"       # ubah sesuai target: is_up_10pct, is_up_20pct, target_B, dll.
MIN_SAMPLE = 50              # minimal baris match
MIN_LIFT = 1.5               # minimal lift di train
TRAIN_FRAC = 0.6             # split train/test by date
N_BOOTSTRAP = 200
MAX_CONDITIONS = 3


# ============================================================
# POOL KONDISI
# ============================================================
CONDITIONS = {
    # Volume
    "vol_ratio_5>1.5":    lambda d: d["vol_ratio_5"] > 1.5,
    "vol_ratio_5>2.0":    lambda d: d["vol_ratio_5"] > 2.0,
    "vol_ratio_5>3.0":    lambda d: d["vol_ratio_5"] > 3.0,
    "vol_ratio_20>1.5":   lambda d: d["vol_ratio_20"] > 1.5,
    "vol_ratio_20>2.0":   lambda d: d["vol_ratio_20"] > 2.0,
    "vol_ratio_20>3.0":   lambda d: d["vol_ratio_20"] > 3.0,
    "vol_z_20>1":         lambda d: d["vol_z_20"] > 1,
    "vol_z_20>2":         lambda d: d["vol_z_20"] > 2,

    # RSI
    "rsi_14<30":          lambda d: d["rsi_14"] < 30,
    "rsi_14<40":          lambda d: d["rsi_14"] < 40,
    "rsi_14_40_60":       lambda d: d["rsi_14"].between(40, 60),
    "rsi_14>60":          lambda d: d["rsi_14"] > 60,
    "rsi_14>70":          lambda d: d["rsi_14"] > 70,

    # Price vs MA
    "px>sma20":           lambda d: d["px_vs_sma20"] > 0,
    "px<sma20":           lambda d: d["px_vs_sma20"] < 0,
    "px_near_sma20":      lambda d: d["px_vs_sma20"].abs() < 0.02,
    "px>2pct_sma20":      lambda d: d["px_vs_sma20"] > 0.02,
    "ma20>ma50":          lambda d: d["ma20_gt_ma50"] == 1,

    # Volatility
    "atr_low":            lambda d: d["atr_pct_rank_60"] < 0.3,
    "atr_very_low":       lambda d: d["atr_pct_rank_60"] < 0.2,
    "atr_high":           lambda d: d["atr_pct_rank_60"] > 0.7,
    "squeeze":            lambda d: d["bb_width_pct_rank_60"] < 0.2,

    # Price action
    "green_candle":       lambda d: d["close"] > d["open"],
    "red_candle":         lambda d: d["close"] < d["open"],
    "clv>0.7":            lambda d: d["clv"] > 0.7,
    "clv<0.3":            lambda d: d["clv"] < 0.3,
    "nr7":                lambda d: d["nr7"] == 1,
    "higher_low_3":       lambda d: d["higher_low_3"] == 1,

    # Momentum
    "ret_1d>0":           lambda d: d["ret_1d"] > 0,
    "ret_1d>2pct":        lambda d: d["ret_1d"] > 0.02,
    "ret_5d>0":           lambda d: d["ret_5d"] > 0,
    "ret_5d<0":           lambda d: d["ret_5d"] < 0,
    "ret_5d_neg_small":   lambda d: d["ret_5d"].between(-0.03, 0),
    "ret_20d<0":          lambda d: d["ret_20d"] < 0,

    # Distance from high/low
    "dist_high_20_neg10": lambda d: d["dist_high_20"] < -0.10,
    "dist_low_20_pos10":  lambda d: d["dist_low_20"] > 0.10,
    "near_high_20":       lambda d: d["dist_high_20"] > -0.03,

    # Accumulation
    "vol_high_price_flat": lambda d: d["vol_high_price_flat"] == 1,
    "vol_up_price_down":   lambda d: d["vol_up_price_down"] == 1,
    "obv_above_ma10":      lambda d: d["obv_above_ma10"] == 1,
    "cmf_pos":             lambda d: d["cmf_20"] > 0,
    "cmf_pos_flat":        lambda d: d["cmf_pos_while_flat"] == 1,
}


# ============================================================
# UTIL
# ============================================================
def bootstrap_ci(y, n_iter=N_BOOTSTRAP, alpha=0.05):
    n = len(y)
    if n == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    means = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        means[i] = y[idx].mean()
    return float(np.percentile(means, 100 * alpha / 2)), \
           float(np.percentile(means, 100 * (1 - alpha / 2)))


def evaluate_rule(mask, target, forward_ret, baseline_rate,
                  train_mask, test_mask):
    n_total = int(mask.sum())
    if n_total < MIN_SAMPLE:
        return None

    m_train = mask & train_mask
    n_train = int(m_train.sum())
    if n_train < MIN_SAMPLE:
        return None

    hit_train = float(target[m_train].mean())
    lift_train = hit_train / baseline_rate if baseline_rate > 0 else np.nan

    m_test = mask & test_mask
    n_test = int(m_test.sum())
    if n_test < 20:
        hit_test = np.nan
        lift_test = np.nan
    else:
        hit_test = float(target[m_test].mean())
        lift_test = hit_test / baseline_rate if baseline_rate > 0 else np.nan

    fwd_ret_match = forward_ret[mask]
    avg_fwd_ret = float(np.nanmean(fwd_ret_match))
    median_fwd_ret = float(np.nanmedian(fwd_ret_match))

    ci_lo, ci_hi = bootstrap_ci(target[mask].astype(float))

    return {
        "n_total": n_total,
        "n_train": n_train,
        "n_test": n_test,
        "hit_train": hit_train,
        "hit_test": hit_test,
        "lift_train": lift_train,
        "lift_test": lift_test,
        "avg_fwd_ret": avg_fwd_ret,
        "median_fwd_ret": median_fwd_ret,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "baseline": baseline_rate,
    }


# ============================================================
# PROGRAM UTAMA
# ============================================================
def main():
    print("Memuat data...")
    panel = load_panel_from_gabungan()
    panel = clean_panel(panel)
    panel = build_features(panel)
    panel = detect_events(panel)

    # Sort by date (wajib untuk split time-series)
    panel = panel.sort_values("date").reset_index(drop=True)

    # Split train/test by date
    dates = np.sort(panel["date"].unique())
    cutoff_train = dates[int(len(dates) * TRAIN_FRAC)]
    train_mask = (panel["date"] < cutoff_train).values
    test_mask = ~train_mask

    print(f"\nSplit by date:")
    print(f"  Train: {panel.loc[train_mask, 'date'].min()} "
          f"s/d {panel.loc[train_mask, 'date'].max()} "
          f"({train_mask.sum():,} baris)")
    print(f"  Test : {panel.loc[test_mask, 'date'].min()} "
          f"s/d {panel.loc[test_mask, 'date'].max()} "
          f"({test_mask.sum():,} baris)")

        target = (
        panel.groupby("kode")[TARGET]
             .shift(-1)
             .fillna(0)
             .values
             .astype(float)
    )
    baseline_rate = float(target.mean())
    print(f"\nTarget: {TARGET}")
    print(f"  Baseline rate: {baseline_rate*100:.3f}% "
          f"({int(target.sum()):,} event)")

    if "fwd_ret_5d" in panel.columns:
        forward_ret = panel["fwd_ret_5d"].values
    else:
        forward_ret = panel["fwd_ret_3d"].values

    # Pre-compute kondisi
    print(f"\nPre-compute {len(CONDITIONS)} kondisi...")
    cond_arrays = {}
    for name, fn in CONDITIONS.items():
        try:
            arr = fn(panel).fillna(False).values
            cond_arrays[name] = arr.astype(bool)
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    print(f"  ✅ {len(cond_arrays)} kondisi siap")

    # Hitung total kombinasi
    cond_names = list(cond_arrays.keys())
    total_combos = sum(
        len(list(itertools.combinations(cond_names, k)))
        for k in range(1, MAX_CONDITIONS + 1)
    )
    print(f"\nMenguji {total_combos:,} kombinasi...")

    results = []
    counter = 0
    for k in range(1, MAX_CONDITIONS + 1):
        for combo in itertools.combinations(cond_names, k):
            counter += 1

            mask = cond_arrays[combo[0]].copy()
            for cname in combo[1:]:
                mask &= cond_arrays[cname]

            n_match = int(mask.sum())
            if n_match < MIN_SAMPLE:
                continue

            res = evaluate_rule(
                mask, target, forward_ret, baseline_rate,
                train_mask, test_mask,
            )
            if res is None:
                continue

            res["rule"] = " AND ".join(combo)
            res["n_conditions"] = k
            results.append(res)

            if counter % 2000 == 0:
                print(f"  ... {counter:,}/{total_combos:,}")

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("\n❌ Tidak ada kombinasi yang lolos filter.")
        return

    # Filter & rank by lift_test
    df_res = df_res[df_res["lift_train"] >= MIN_LIFT].copy()
    df_res = df_res.sort_values(
        ["lift_test", "lift_train"], ascending=[False, False]
    ).reset_index(drop=True)

    print(f"\n✅ {len(df_res):,} kombinasi lolos filter "
          f"(lift_train >= {MIN_LIFT})")

    out_path = f"quant/output/pattern_mining_{TARGET}.csv"
    df_res.to_csv(out_path, index=False)
    print(f"→ Disimpan: {out_path}")

    # Tampilkan top 30
    show_cols = ["rule", "n_total", "n_test", "hit_train", "hit_test",
                 "lift_train", "lift_test", "avg_fwd_ret"]

    def fmt(df):
        d = df[show_cols].copy()
        d["hit_train"] = (d["hit_train"] * 100).round(3)
        d["hit_test"] = (d["hit_test"] * 100).round(3)
        d["lift_train"] = d["lift_train"].round(2)
        d["lift_test"] = d["lift_test"].round(2)
        d["avg_fwd_ret"] = (d["avg_fwd_ret"] * 100).round(3)
        return d

    print(f"\n{'='*80}")
    print(f"TOP 30 RULE (ranked by lift_test)")
    print(f"{'='*80}")
    with pd.option_context("display.max_colwidth", 70,
                            "display.width", 220):
        print(fmt(df_res).head(30).to_string(index=False))

    print(f"\n{'='*80}")
    print("TOP 10 KONDISI TUNGGAL")
    print(f"{'='*80}")
    single = df_res[df_res["n_conditions"] == 1]
    if not single.empty:
        with pd.option_context("display.max_colwidth", 40,
                                "display.width", 220):
            print(fmt(single).head(10).to_string(index=False))


if __name__ == "__main__":
    main()