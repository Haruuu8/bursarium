"""
Pipeline discovery + backtest yang benar:
1. Rank conditions by ACTUAL PRECISION
2. Depth 3-4 combinations
3. Test on 4 months out-of-sample
"""

import itertools
import numpy as np
import pandas as pd

from .data_loader import load_panel_from_gabungan
from .data_cleaner import clean_panel
from .feature_engineering import build_features
from .event_detector import detect_events


EVENT_TARGET = "is_touch_10pct"
DISCOVERY_MONTH = "2026-09"
TEST_MONTHS = ["2026-08", "2026-07", "2026-06", "2026-05"]

MIN_PRECISION_DISC = 0.10   # minimal 10% di discovery
MIN_N_MATCH = 30            # minimal 30 sinyal
MAX_DEPTH = 4


CONDITIONS = {
    # VOLUME
    "vol_ratio_5>1.5":      lambda d: d["vol_ratio_5"] > 1.5,
    "vol_ratio_5>2.0":      lambda d: d["vol_ratio_5"] > 2.0,
    "vol_ratio_5>3.0":      lambda d: d["vol_ratio_5"] > 3.0,
    "vol_ratio_5>5.0":      lambda d: d["vol_ratio_5"] > 5.0,
    "vol_ratio_20>1.5":     lambda d: d["vol_ratio_20"] > 1.5,
    "vol_ratio_20>2.0":     lambda d: d["vol_ratio_20"] > 2.0,
    "vol_ratio_20>3.0":     lambda d: d["vol_ratio_20"] > 3.0,
    "vol_z_20>2":           lambda d: d["vol_z_20"] > 2,
    # GAP
    "gap_up>1%":            lambda d: d["gap"] > 0.01,
    "gap_up>2%":            lambda d: d["gap"] > 0.02,
    "gap_up>3%":            lambda d: d["gap"] > 0.03,
    "gap_up>5%":            lambda d: d["gap"] > 0.05,
    # PRICE ACTION
    "close>open":           lambda d: d["close"] > d["open"],
    "clv>0.5":              lambda d: d["clv"] > 0.5,
    "clv>0.7":              lambda d: d["clv"] > 0.7,
    "clv>0.85":             lambda d: d["clv"] > 0.85,
    # MOMENTUM
    "ret_1d>0":             lambda d: d["ret_1d"] > 0,
    "ret_1d>2%":            lambda d: d["ret_1d"] > 0.02,
    "ret_1d>5%":            lambda d: d["ret_1d"] > 0.05,
    "ret_1d_neg":           lambda d: d["ret_1d"] < 0,
    "ret_5d>0":             lambda d: d["ret_5d"] > 0,
    "ret_5d>3%":            lambda d: d["ret_5d"] > 0.03,
    "ret_5d>5%":            lambda d: d["ret_5d"] > 0.05,
    "ret_5d>10%":           lambda d: d["ret_5d"] > 0.10,
    "ret_20d>0":            lambda d: d["ret_20d"] > 0,
    "ret_20d>10%":          lambda d: d["ret_20d"] > 0.10,
    # TREND
    "px>sma20":             lambda d: d["px_vs_sma20"] > 0,
    "px>2pct_above_sma20":  lambda d: d["px_vs_sma20"] > 0.02,
    "px>5pct_above_sma20":  lambda d: d["px_vs_sma20"] > 0.05,
    "ma20>ma50":            lambda d: d["ma20_gt_ma50"] == 1,
    "ma50>ma200":           lambda d: d["ma50_gt_ma200"] == 1,
    "dekat_high_20d":       lambda d: d["dist_high_20"] > -0.03,
    "dekat_high_20d_5pct":  lambda d: d["dist_high_20"] > -0.05,
    # RSI
    "rsi_14>50":            lambda d: d["rsi_14"] > 50,
    "rsi_14>60":            lambda d: d["rsi_14"] > 60,
    "rsi_14>70":            lambda d: d["rsi_14"] > 70,
    "rsi_14>75":            lambda d: d["rsi_14"] > 75,
    "rsi_14<40":            lambda d: d["rsi_14"] < 40,
    "rsi_14<30":            lambda d: d["rsi_14"] < 30,
    # VOLATILITAS
    "atr_low":              lambda d: d["atr_pct_rank_60"] < 0.3,
    "atr_high":             lambda d: d["atr_pct_rank_60"] > 0.7,
    "bb_squeeze":           lambda d: d["bb_width_pct_rank_60"] < 0.2,
    "nr7":                  lambda d: d["nr7"] == 1,
    # AKUMULASI
    "vol_high_close_high":  lambda d: d["vol_high_close_high"] == 1,
    "obv_above_ma10":       lambda d: d["obv_above_ma10"] == 1,
    "cmf_pos":              lambda d: d["cmf_20"] > 0,
    "mfi_overbought":       lambda d: d["mfi_14"] > 70,
    # LIKUIDITAS
    "value>5M":             lambda d: d["value"] > 5_000_000_000,
    "value>10M":            lambda d: d["value"] > 10_000_000_000,
}


def month_mask(dates, m):
    return np.asarray(pd.to_datetime(dates).to_period("M") == m)


def discovery(df, event_col, cond_arrays):
    """Rank conditions by ACTUAL PRECISION."""
    ev = df[event_col].values.astype(int)
    n_ev = int(ev.sum())
    n_tot = len(ev)

    hasil = []
    for name, full_cond in cond_arrays.items():
        c = full_cond.loc[df.index].values.astype(bool)
        n_match = int(c.sum())
        if n_match < MIN_N_MATCH:
            continue
        n_hit = int((c & (ev == 1)).sum())
        precision = n_hit / n_match
        if precision < MIN_PRECISION_DISC:
            continue
        coverage = n_hit / n_ev if n_ev > 0 else 0
        hasil.append({
            "kondisi": name,
            "n_match": n_match,
            "n_hit": n_hit,
            "precision": precision,
            "coverage": coverage,
        })
    df_r = pd.DataFrame(hasil)
    if not df_r.empty:
        df_r = df_r.sort_values("precision", ascending=False).reset_index(drop=True)
    return df_r, n_ev


def test_combo(df_disc, df_tests, event_col, cond_arrays, combo):
    """Test satu kombinasi di discovery + semua test months."""
    # Discovery
    md = cond_arrays[combo[0]].loc[df_disc.index].values.astype(bool)
    for c in combo[1:]:
        md &= cond_arrays[c].loc[df_disc.index].values.astype(bool)
    nd = int(md.sum())
    if nd < MIN_N_MATCH:
        return None
    ed = df_disc[event_col].values.astype(int)
    pd_ = int((md & (ed == 1)).sum()) / nd

    # Test months
    precs = []
    ns = []
    for name, df_t in df_tests.items():
        mt = cond_arrays[combo[0]].loc[df_t.index].values.astype(bool)
        for c in combo[1:]:
            mt &= cond_arrays[c].loc[df_t.index].values.astype(bool)
        nt = int(mt.sum())
        et = df_t[event_col].values.astype(int)
        pt = int((mt & (et == 1)).sum()) / nt if nt > 0 else 0
        precs.append(pt)
        ns.append(nt)

    avg_test = np.mean(precs) if precs else 0
    min_test = np.min(precs) if precs else 0

    return {
        "rule": " AND ".join(combo),
        "n_cond": len(combo),
        "n_disc": nd,
        "prec_disc": pd_,
        "n_test_avg": int(np.mean(ns)),
        "prec_test_avg": avg_test,
        "prec_test_min": min_test,
        "prec_per_month": "/".join(f"{p*100:.1f}" for p in precs),
    }


def main():
    print("=" * 80)
    print("PIPELINE v2 - ranking by PRECISION, multi-month test")
    print("=" * 80)

    print("\n[1/5] Load & prepare...")
    panel = load_panel_from_gabungan()
    panel = clean_panel(panel)
    panel = build_features(panel)
    panel = detect_events(panel)
    panel = panel.sort_values(["kode", "date"]).reset_index(drop=True)

    print("\n[2/5] Shift event T+1...")
    panel["event_next"] = (
        panel.groupby("kode")[EVENT_TARGET]
             .shift(-1).fillna(0).astype(int)
    )
    print(f"  Total event T+1: {panel['event_next'].sum():,}")

    print(f"\n[3/5] Pre-compute {len(CONDITIONS)} kondisi...")
    cond_arrays = {}
    for name, fn in CONDITIONS.items():
        try:
            cond_arrays[name] = fn(panel).fillna(False)
        except Exception as e:
            print(f"  [X] {name}: {e}")
    print(f"  {len(cond_arrays)} kondisi siap")

    dates = panel["date"].values
    df_disc = panel[month_mask(dates, DISCOVERY_MONTH)].copy()
    df_tests = {m: panel[month_mask(dates, m)].copy() for m in TEST_MONTHS}

    print(f"\n[4/5] Data:")
    print(f"  Discovery {DISCOVERY_MONTH}: {len(df_disc):,} baris, "
          f"{df_disc['event_next'].sum()} events ({df_disc['event_next'].mean()*100:.2f}%)")
    for m, df_t in df_tests.items():
        print(f"  Test {m}: {len(df_t):,} baris, "
              f"{df_t['event_next'].sum()} events ({df_t['event_next'].mean()*100:.2f}%)")

    # ===== FASE 1: DISCOVERY =====
    print(f"\n{'='*80}")
    print(f"FASE 1: KONDISI RANKED BY PRECISION ({DISCOVERY_MONTH})")
    print(f"{'='*80}")

    prof, n_ev = discovery(df_disc, "event_next", cond_arrays)
    if prof.empty:
        print("[X] Tidak ada kondisi dengan precision >= "
              f"{MIN_PRECISION_DISC*100:.0f}% dan n >= {MIN_N_MATCH}")
        return

    print(f"\nTop gainer: {n_ev} events, baseline {df_disc['event_next'].mean()*100:.2f}%")
    print(f"\n{'Kondisi':<28} {'n':>6} {'n_hit':>6} {'prec%':>7} {'cov%':>6}")
    print("-" * 60)
    for _, r in prof.head(25).iterrows():
        print(f"{r['kondisi']:<28} {int(r['n_match']):>6} {int(r['n_hit']):>6} "
              f"{r['precision']*100:>6.2f}% {r['coverage']*100:>5.1f}%")

    # ===== FASE 2: KOMBINASI =====
    print(f"\n{'='*80}")
    print(f"FASE 2: KOMBINASI DEPTH 1-{MAX_DEPTH}")
    print(f"{'='*80}")

    top_kondisi = prof.head(15)["kondisi"].tolist()
    print(f"\nPool: {len(top_kondisi)} kondisi teratas")

    results = []
    total = 0
    for k in range(1, MAX_DEPTH + 1):
        n_k = 0
        for combo in itertools.combinations(top_kondisi, k):
            total += 1
            n_k += 1
            r = test_combo(df_disc, df_tests, "event_next", cond_arrays, combo)
            if r is not None:
                results.append(r)
        print(f"  Depth {k}: {n_k:,} kombinasi diuji, "
              f"lolos: {len([x for x in results if x['n_cond'] == k])}")

    print(f"\nTotal diuji: {total:,}")

    if not results:
        print("[X] Tidak ada kombinasi valid.")
        return

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("prec_test_avg", ascending=False).reset_index(drop=True)

    # ===== HASIL =====
    print(f"\n{'='*80}")
    print(f"TOP 30 KOMBINASI (ranked by precision_test_avg)")
    print(f"{'='*80}")
    print(f"\n{'Rule':<55} {'k':>2} {'n_d':>5} {'p_d%':>6} {'p_avg%':>7} {'p_min%':>7} {'per_month'}")
    print("-" * 130)
    for _, r in df_res.head(30).iterrows():
        rule_short = r["rule"][:53]
        print(f"{rule_short:<55} {r['n_cond']:>2} {int(r['n_disc']):>5} "
              f"{r['prec_disc']*100:>5.1f} {r['prec_test_avg']*100:>6.2f} "
              f"{r['prec_test_min']*100:>6.2f}  {r['prec_per_month']}")

    # ===== VERDICT =====
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")

    best = df_res.iloc[0]
    print(f"\nBest rule: {best['rule']}")
    print(f"  Precision discovery : {best['prec_disc']*100:.2f}%")
    print(f"  Precision avg test  : {best['prec_test_avg']*100:.2f}%")
    print(f"  Precision min test  : {best['prec_test_min']*100:.2f}%")
    print(f"  Per bulan: {best['prec_per_month']}")
    print(f"  n discovery: {int(best['n_disc'])}")

    # Berapa banyak rule dengan precision test avg > berbagai threshold?
    print(f"\nDistribusi precision_test_avg:")
    for t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        n = (df_res["prec_test_avg"] >= t).sum()
        print(f"  >= {t*100:.0f}%: {n} rule")

    df_res.to_csv("quant/output/pipeline_v2.csv", index=False)
    print(f"\n-> quant/output/pipeline_v2.csv")


if __name__ == "__main__":
    main()