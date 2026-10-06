"""
SCREENER MAX WR - cari rule dengan win rate setinggi mungkin.
Fokus: WR tinggi, sinyal boleh sedikit (min 10 test signal).

Simulasi REALISTIS:
  - target_3   : jual di +3% kalau kena, else close T+1
  - target_5   : jual di +5% kalau kena, else close T+1
  - trail_2    : trailing 2% selama 3 hari
  - trail_3    : trailing 3% selama 3 hari

Split:
  Discovery : 2026-01-01 s/d 2026-07-31
  Test      : 2026-08-01 s/d 2026-10-02

Jalankan: python -m quant.screener_max_wr
"""

import os
import sys
import json
import itertools
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant.data_loader import load_panel_from_gabungan
from quant.data_cleaner import clean_panel
from quant.feature_engineering import build_features
from quant.event_detector import detect_events


# ============================================================
# KONFIGURASI
# ============================================================
COST_RT = 0.008

DISCOVERY_START = "2026-01-01"
DISCOVERY_END   = "2026-07-31"
TEST_START      = "2026-08-01"
TEST_END        = "2026-10-02"

MIN_TRAIN_N = 20
MIN_TEST_N  = 10
MIN_TRAIN_WR = 0.60
MIN_TEST_WR  = 0.60
MIN_AVG_PL   = 0.010

MAX_COND = 4
OUTPUT_DIR = "quant/output"

# Mode simulasi
MODES = ["target_3", "target_5", "trail_2", "trail_3"]


# ============================================================
# KONDISI POOL (fitur yang sudah ada di feature_engineering)
# ============================================================
CONDITIONS = {
    # Momentum
    "ret_1d>2%":        lambda d: d["ret_1d"] > 0.02,
    "ret_1d>5%":        lambda d: d["ret_1d"] > 0.05,
    "ret_1d>10%":       lambda d: d["ret_1d"] > 0.10,
    "ret_5d>5%":        lambda d: d["ret_5d"] > 0.05,
    "ret_5d>10%":       lambda d: d["ret_5d"] > 0.10,
    "ret_5d>20%":       lambda d: d["ret_5d"] > 0.20,
    "ret_20d>10%":      lambda d: d["ret_20d"] > 0.10,
    
    # Gap
    "gap>2%":           lambda d: d["gap"] > 0.02,
    "gap>5%":           lambda d: d["gap"] > 0.05,
    "gap>8%":           lambda d: d["gap"] > 0.08,
    
    # Volume
    "vol_z>1.5":        lambda d: d["vol_z_20"] > 1.5,
    "vol_z>2.5":        lambda d: d["vol_z_20"] > 2.5,
    "vol_r20>2":        lambda d: d["vol_ratio_20"] > 2.0,
    "vol_r20>3":        lambda d: d["vol_ratio_20"] > 3.0,
    "vol_r5>2":         lambda d: d["vol_ratio_5"] > 2.0,
    
    # Price action
    "clv>0.7":          lambda d: d["clv"] > 0.7,
    "clv>0.85":         lambda d: d["clv"] > 0.85,
    "body>3%":          lambda d: d["body"] > 0.03,
    "body>5%":          lambda d: d["body"] > 0.05,
    "range>8%":         lambda d: d["range"] > 0.08,
    "range>12%":        lambda d: d["range"] > 0.12,
    "green_candle":     lambda d: d["close"] > d["open"],
    
    # Trend
    "px>sma20":         lambda d: d["px_vs_sma20"] > 0,
    "px>sma20+5%":      lambda d: d["px_vs_sma20"] > 0.05,
    "px>sma20+10%":     lambda d: d["px_vs_sma20"] > 0.10,
    "ma20>ma50":        lambda d: d["ma20_gt_ma50"] == 1,
    
    # RSI
    "rsi>50":           lambda d: d["rsi_14"] > 50,
    "rsi>60":           lambda d: d["rsi_14"] > 60,
    "rsi>70":           lambda d: d["rsi_14"] > 70,
    "rsi<40":           lambda d: d["rsi_14"] < 40,
    
    # Position
    "near_high20":      lambda d: d["dist_high_20"] > -0.05,
    "break_high20":     lambda d: d["dist_high_20"] >= -0.005,
    "from_low20>15%":   lambda d: d["dist_low_20"] > 0.15,
    
    # Volatilitas
    "atr_low":          lambda d: d["atr_pct_rank_60"] < 0.3,
    "atr_high":         lambda d: d["atr_pct_rank_60"] > 0.6,
    "bb_squeeze":       lambda d: d["bb_width_pct_rank_60"] < 0.25,
    
    # Akumulasi
    "cmf_pos":          lambda d: d["cmf_20"] > 0.05,
    "cmf_strong":       lambda d: d["cmf_20"] > 0.15,
    "obv_up":           lambda d: d["obv_above_ma10"] == 1,
    "vol_price_up":     lambda d: d["vol_high_close_high"] == 1,
    
    # Filter kualitas
    "close>100":        lambda d: d["close"] > 100,
    "close>500":        lambda d: d["close"] > 500,
    "value>1B":         lambda d: d["value"] > 1e9,
    "value>5B":         lambda d: d["value"] > 5e9,
}


# ============================================================
# SIMULASI REALISTIS
# ============================================================
def simulate_all_modes(panel):
    """Hitung return per baris untuk 4 mode realistis."""
    entry = panel["open_T1"].values.astype(float)
    close_t = panel["close"].values.astype(float)
    high_t1 = panel["high_T1"].values.astype(float)
    low_t1 = panel["low_T1"].values.astype(float)
    close_t1 = panel["close_T1"].values.astype(float)
    high_t2 = panel["high_T2"].values.astype(float)
    low_t2 = panel["low_T2"].values.astype(float)
    close_t2 = panel["close_T2"].values.astype(float)
    high_t3 = panel["high_T3"].values.astype(float)
    low_t3 = panel["low_T3"].values.astype(float)
    close_t3 = panel["close_T3"].values.astype(float)

    valid = (entry > 0) & (~np.isnan(entry))
    skip = valid & (entry < close_t)

    results = {}

    # target_3
    ret = np.full(len(panel), np.nan)
    target = entry * 1.03
    hit = valid & (~skip) & (~np.isnan(high_t1)) & (high_t1 >= target)
    miss = valid & (~skip) & (~hit) & (~np.isnan(close_t1))
    ret[hit] = 0.03 - COST_RT
    ret[miss] = close_t1[miss] / entry[miss] - 1 - COST_RT
    results["target_3"] = ret

    # target_5
    ret = np.full(len(panel), np.nan)
    target = entry * 1.05
    hit = valid & (~skip) & (~np.isnan(high_t1)) & (high_t1 >= target)
    miss = valid & (~skip) & (~hit) & (~np.isnan(close_t1))
    ret[hit] = 0.05 - COST_RT
    ret[miss] = close_t1[miss] / entry[miss] - 1 - COST_RT
    results["target_5"] = ret

    # trail_2
    ret = np.full(len(panel), np.nan)
    for i in range(len(panel)):
        if not valid[i] or skip[i]:
            continue
        e = entry[i]
        trail_level = None
        trail_high = e
        exit_price = None
        for h, l, c in [(high_t1[i], low_t1[i], close_t1[i]),
                        (high_t2[i], low_t2[i], close_t2[i]),
                        (high_t3[i], low_t3[i], close_t3[i])]:
            if np.isnan(h) or np.isnan(l) or np.isnan(c):
                break
            if trail_level is not None and l <= trail_level:
                exit_price = trail_level
                break
            if h > trail_high:
                trail_high = h
            trail_level = max(trail_high * 0.98, e)
        if exit_price is None:
            for c in [close_t3[i], close_t2[i], close_t1[i]]:
                if not np.isnan(c):
                    exit_price = c
                    break
        if exit_price is not None and exit_price > 0:
            ret[i] = exit_price / e - 1 - COST_RT
    results["trail_2"] = ret

    # trail_3
    ret = np.full(len(panel), np.nan)
    for i in range(len(panel)):
        if not valid[i] or skip[i]:
            continue
        e = entry[i]
        trail_level = None
        trail_high = e
        exit_price = None
        for h, l, c in [(high_t1[i], low_t1[i], close_t1[i]),
                        (high_t2[i], low_t2[i], close_t2[i]),
                        (high_t3[i], low_t3[i], close_t3[i])]:
            if np.isnan(h) or np.isnan(l) or np.isnan(c):
                break
            if trail_level is not None and l <= trail_level:
                exit_price = trail_level
                break
            if h > trail_high:
                trail_high = h
            trail_level = max(trail_high * 0.97, e)
        if exit_price is None:
            for c in [close_t3[i], close_t2[i], close_t1[i]]:
                if not np.isnan(c):
                    exit_price = c
                    break
        if exit_price is not None and exit_price > 0:
            ret[i] = exit_price / e - 1 - COST_RT
    results["trail_3"] = ret

    return results


# ============================================================
# EVALUASI RULE
# ============================================================
def eval_mask(mask, targets, min_n):
    """Return dict metrics untuk mask."""
    n = int(mask.sum())
    if n < min_n:
        return None
    out = {"n": n}
    for mode, rets in targets.items():
        sub = rets[mask]
        sub = sub[~np.isnan(sub)]
        if len(sub) < 5:
            out[f"{mode}_wr"] = np.nan
            out[f"{mode}_avg"] = np.nan
            out[f"{mode}_n"] = 0
            continue
        out[f"{mode}_wr"] = float((sub > 0).mean())
        out[f"{mode}_avg"] = float(sub.mean())
        out[f"{mode}_n"] = len(sub)
    return out


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 110)
    print(f"SCREENER MAX WR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 110)

    # ---- Load ----
    print("\nLoad data...")
    panel = load_panel_from_gabungan()
    panel = clean_panel(panel)
    panel = build_features(panel)
    panel = detect_events(panel)
    panel = panel.sort_values(["kode", "date"]).reset_index(drop=True)

    # Forward OHLC
    for h in [1, 2, 3]:
        panel[f"open_T{h}"] = panel.groupby("kode")["open"].shift(-h)
        panel[f"high_T{h}"] = panel.groupby("kode")["high"].shift(-h)
        panel[f"low_T{h}"] = panel.groupby("kode")["low"].shift(-h)
        panel[f"close_T{h}"] = panel.groupby("kode")["close"].shift(-h)

    panel = panel[(panel["open"] > 0) & (panel["close"] > 0)].copy()
    panel = panel.dropna(subset=["ret_1d", "ret_5d", "ret_20d", "clv", "gap",
                                  "vol_z_20", "body", "range", "px_vs_sma20",
                                  "rsi_14", "dist_high_20", "dist_low_20"])

    print(f"Total baris: {len(panel):,}")
    print(f"Total saham: {panel['kode'].nunique()}")

    # ---- Simulate ----
    print("\nSimulasi 4 mode realistis...")
    targets = simulate_all_modes(panel)

    # ---- Split ----
    d1 = pd.Timestamp(DISCOVERY_START)
    d2 = pd.Timestamp(DISCOVERY_END)
    t1 = pd.Timestamp(TEST_START)
    t2 = pd.Timestamp(TEST_END)

    disc_mask = ((panel["date"] >= d1) & (panel["date"] <= d2)).values
    test_mask = ((panel["date"] >= t1) & (panel["date"] <= t2)).values

    print(f"\nDiscovery: {disc_mask.sum():,} baris "
          f"({panel.loc[disc_mask, 'date'].nunique()} hari)")
    print(f"Test     : {test_mask.sum():,} baris "
          f"({panel.loc[test_mask, 'date'].nunique()} hari)")

    # Baseline
    print(f"\nBaseline di test:")
    print(f"{'Mode':<12} {'N':>7} {'WR':>7} {'Avg':>8}")
    print("-" * 40)
    for mode, rets in targets.items():
        sub = rets[test_mask]
        sub = sub[~np.isnan(sub)]
        if len(sub) > 0:
            print(f"{mode:<12} {len(sub):>7} "
                  f"{(sub > 0).mean()*100:>6.1f}% "
                  f"{sub.mean()*100:>+7.2f}%")

    # ---- Precompute kondisi ----
    print(f"\nPrecompute {len(CONDITIONS)} kondisi...")
    cond_arrays = {}
    for name, fn in CONDITIONS.items():
        try:
            cond_arrays[name] = fn(panel).fillna(False).values.astype(bool)
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    print(f"  ✅ {len(cond_arrays)} kondisi siap")

    cond_names = list(cond_arrays.keys())

    # ---- Generate & evaluate ----
    total = sum(
        len(list(itertools.combinations(cond_names, k)))
        for k in range(1, MAX_COND + 1)
    )
    print(f"\nMenguji {total:,} kombinasi (max {MAX_COND} kondisi)...")

    results = []
    counter = 0
    t0 = datetime.now()

    for k in range(1, MAX_COND + 1):
        for combo in itertools.combinations(cond_names, k):
            counter += 1

            mask = cond_arrays[combo[0]].copy()
            for c in combo[1:]:
                mask &= cond_arrays[c]

            # Train
            m_train = mask & disc_mask
            if m_train.sum() < MIN_TRAIN_N:
                continue

            # Test
            m_test = mask & test_mask
            if m_test.sum() < MIN_TEST_N:
                continue

            r_train = eval_mask(m_train, targets, MIN_TRAIN_N)
            r_test = eval_mask(m_test, targets, MIN_TEST_N)
            if r_train is None or r_test is None:
                continue

            row = {"rule": " AND ".join(combo), "k": k}
            for mode in MODES:
                row[f"train_{mode}_wr"] = r_train.get(f"{mode}_wr")
                row[f"train_{mode}_avg"] = r_train.get(f"{mode}_avg")
                row[f"train_{mode}_n"] = r_train.get(f"{mode}_n", 0)
                row[f"test_{mode}_wr"] = r_test.get(f"{mode}_wr")
                row[f"test_{mode}_avg"] = r_test.get(f"{mode}_avg")
                row[f"test_{mode}_n"] = r_test.get(f"{mode}_n", 0)

            results.append(row)

            if counter % 5000 == 0:
                elapsed = (datetime.now() - t0).total_seconds()
                print(f"  ... {counter:,}/{total:,} ({elapsed:.0f}s)")

    df = pd.DataFrame(results)
    print(f"\nTotal rules lolos filter: {len(df):,}")

    if df.empty:
        print("Tidak ada rule yang lolos.")
        return

    # ---- SAVE ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_path = os.path.join(OUTPUT_DIR, "max_wr_full.csv")
    df.to_csv(full_path, index=False)
    print(f"-> Disimpan: {full_path}")

    # ---- TAMPILKAN TOP PER MODE ----
    for mode in MODES:
        col_wr_tr = f"train_{mode}_wr"
        col_wr_te = f"test_{mode}_wr"
        col_avg_tr = f"train_{mode}_avg"
        col_avg_te = f"test_{mode}_avg"
        col_n_tr = f"train_{mode}_n"
        col_n_te = f"test_{mode}_n"

        if col_wr_te not in df.columns:
            continue

        dm = df.copy()
        dm = dm.dropna(subset=[col_wr_tr, col_wr_te])
        dm = dm[(dm[col_wr_tr] >= MIN_TRAIN_WR) &
                (dm[col_wr_te] >= MIN_TEST_WR) &
                (dm[col_avg_te] >= MIN_AVG_PL)]
        dm = dm.sort_values([col_wr_te, col_avg_te], ascending=[False, False])

        print(f"\n{'='*110}")
        print(f"MODE: {mode.upper()} | filter: train_WR>={MIN_TRAIN_WR*100:.0f}% "
              f"& test_WR>={MIN_TEST_WR*100:.0f}% & test_avg>={MIN_AVG_PL*100:.1f}%")
        print(f"{'='*110}")

        if dm.empty:
            print("  Tidak ada rule yang lolos filter.")
            continue

        print(f"\n{'Rule':<55} {'n_tr':>5} {'WR_tr':>7} {'avg_tr':>8} "
              f"{'n_te':>5} {'WR_te':>7} {'avg_te':>8}")
        print("-" * 110)
        for _, r in dm.head(25).iterrows():
            print(f"{r['rule'][:53]:<55} "
                  f"{int(r[col_n_tr]):>5} {r[col_wr_tr]*100:>6.1f}% "
                  f"{r[col_avg_tr]*100:>+7.2f}% "
                  f"{int(r[col_n_te]):>5} {r[col_wr_te]*100:>6.1f}% "
                  f"{r[col_avg_te]*100:>+7.2f}%")

    # ---- FOKUS: mode target_3 & trail_3 ----
    print(f"\n{'='*110}")
    print("🏆 TOP 15 RULE UNTUK LIVE TRADING (mode target_3 / trail_3)")
    print(f"{'='*110}")

    for mode in ["target_3", "trail_3"]:
        col_wr_tr = f"train_{mode}_wr"
        col_wr_te = f"test_{mode}_wr"
        col_avg_te = f"test_{mode}_avg"
        col_n_te = f"test_{mode}_n"
        col_n_tr = f"train_{mode}_n"
        col_avg_tr = f"train_{mode}_avg"

        dm = df.dropna(subset=[col_wr_tr, col_wr_te]).copy()
        dm = dm[(dm[col_wr_tr] >= 0.60) &
                (dm[col_wr_te] >= 0.60) &
                (dm[col_avg_te] >= 0.01)]
        dm = dm.sort_values([col_wr_te, col_avg_te], ascending=[False, False])

        print(f"\n--- Mode {mode} ---")
        if dm.empty:
            print("  Tidak ada rule yang lolos (train_WR>=60% & test_WR>=60%).")
            continue

        print(f"\n{'Rule':<55} {'n_te':>5} {'WR_te':>7} {'avg_te':>8} {'n/bln':>6}")
        print("-" * 95)
        for _, r in dm.head(15).iterrows():
            n_per_month = r[col_n_te] / 3   # test = ~3 bulan
            print(f"{r['rule'][:53]:<55} {int(r[col_n_te]):>5} "
                  f"{r[col_wr_te]*100:>6.1f}% {r[col_avg_te]*100:>+7.2f}% "
                  f"{n_per_month:>5.1f}")

    # ---- ENSEMBLE VOTING ----
    print(f"\n{'='*110}")
    print("🎯 ENSEMBLE VOTING (gabungkan top 3 rule, cari WR lebih tinggi lagi)")
    print(f"{'='*110}")

    mode = "target_3"
    col_wr_te = f"test_{mode}_wr"
    col_avg_te = f"test_{mode}_avg"

    top = df[(df[f"train_{mode}_wr"] >= 0.60) &
             (df[col_wr_te] >= 0.60) &
             (df[col_avg_te] >= 0.01)].copy()
    top = top.sort_values([col_wr_te, col_avg_te], ascending=[False, False])

    if len(top) >= 3:
        top3 = top.head(3)
        print("\nMenggabungkan top 3 rule:")
        for _, r in top3.iterrows():
            print(f"  • {r['rule']}")

        # Hitung vote
        votes = np.zeros(len(panel))
        for rule_str in top3["rule"]:
            combo = rule_str.split(" AND ")
            m = cond_arrays[combo[0]].copy()
            for c in combo[1:]:
                m &= cond_arrays[c]
            votes += m.astype(int)

        # Voting threshold: minimal 2 dari 3 rule setuju
        for thresh in [1, 2, 3]:
            vote_mask = votes >= thresh
            m_tr = vote_mask & disc_mask
            m_te = vote_mask & test_mask
            if m_te.sum() < 5:
                continue
            rets = targets[mode]
            sub_tr = rets[m_tr]
            sub_te = rets[m_te]
            sub_tr = sub_tr[~np.isnan(sub_tr)]
            sub_te = sub_te[~np.isnan(sub_te)]
            print(f"\n  Vote >= {thresh}:")
            print(f"    Train: n={len(sub_tr)}, WR={(sub_tr>0).mean()*100:.1f}%, "
                  f"avg={sub_tr.mean()*100:+.2f}%")
            print(f"    Test : n={len(sub_te)}, WR={(sub_te>0).mean()*100:.1f}%, "
                  f"avg={sub_te.mean()*100:+.2f}%")

    print(f"\n{'='*110}")
    print(f"SELESAI - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*110}")


if __name__ == "__main__":
    main()