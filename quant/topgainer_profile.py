"""
Profile top gainer: cari karakteristik apa yang KHAS dari saham
sebelum naik ekstrem.

Pendekatan:
1. Ambil SEMUA event top gainer (misal: semua baris is_up_20pct=1)
2. Ambil snapshot fitur pada T-1, T-2, T-3, T-5, T-10, T-20
3. Untuk setiap fitur, hitung:
   - Coverage: berapa % top gainer punya karakteristik ini?
   - Baseline: berapa % non-top-gainer punya karakteristik ini?
   - Lift: coverage / baseline (seberapa KHAS karakteristik ini untuk top gainer)
4. Urutkan berdasarkan lift — fitur paling khas di atas

PENTING: tidak menghitung win rate. Fokus pada "apa yang sama dari top gainer".
"""

import numpy as np
import pandas as pd

from .data_loader import load_panel_from_gabungan
from .data_cleaner import clean_panel
from .feature_engineering import build_features
from .event_detector import detect_events


# ============================================================
# KONFIGURASI
# ============================================================
# Definisi top gainer yang ingin diprofilkan
GAINER_DEFINITIONS = {
    # Definisi UTAMA: menyentuh +10% (close-based OR intraday high)
    "touch_10pct":   "is_touch_10pct",
    # Pembanding
    "up_10pct":      "is_up_10pct",
    "touch_20pct":   "is_touch_20pct",
    "top_1pct":      "is_top_1pct",
}

MIN_COVERAGE = 0.05    # hanya tampilkan kondisi yang muncul di >=5% top gainer
MIN_LIFT = 1.3         # hanya tampilkan kondisi dengan lift >= 1.3


# ============================================================
# POOL KONDISI (karakteristik yang diuji)
# ============================================================
# Setiap kondisi adalah "karakteristik" yang mungkin dimiliki top gainer.
# Kita uji satu per satu, tidak untuk prediksi, tapi untuk profil.
CONDITIONS = {
    # === VOLUME ===
    "vol_ratio_5 > 1.5":       lambda d: d["vol_ratio_5"] > 1.5,
    "vol_ratio_5 > 2.0":       lambda d: d["vol_ratio_5"] > 2.0,
    "vol_ratio_5 > 3.0":       lambda d: d["vol_ratio_5"] > 3.0,
    "vol_ratio_20 > 1.5":      lambda d: d["vol_ratio_20"] > 1.5,
    "vol_ratio_20 > 2.0":      lambda d: d["vol_ratio_20"] > 2.0,
    "vol_ratio_20 > 3.0":      lambda d: d["vol_ratio_20"] > 3.0,
    "vol_z_20 > 1":            lambda d: d["vol_z_20"] > 1,
    "vol_z_20 > 2":            lambda d: d["vol_z_20"] > 2,

    # === PRICE ACTION ===
    "green candle (close>open)": lambda d: d["close"] > d["open"],
    "red candle (close<open)":   lambda d: d["close"] < d["open"],
    "clv > 0.7 (close near high)": lambda d: d["clv"] > 0.7,
    "clv < 0.3 (close near low)":  lambda d: d["clv"] < 0.3,
    "gap up > 1%":             lambda d: d["gap"] > 0.01,
    "gap down < -1%":          lambda d: d["gap"] < -0.01,

    # === RETURNS ===
    "ret_1d > 0":              lambda d: d["ret_1d"] > 0,
    "ret_1d > 2%":             lambda d: d["ret_1d"] > 0.02,
    "ret_1d < 0":              lambda d: d["ret_1d"] < 0,
    "ret_5d > 0":              lambda d: d["ret_5d"] > 0,
    "ret_5d > 5%":             lambda d: d["ret_5d"] > 0.05,
    "ret_5d < 0":              lambda d: d["ret_5d"] < 0,
    "ret_5d antara -3% dan 0%": lambda d: d["ret_5d"].between(-0.03, 0),
    "ret_20d > 0":             lambda d: d["ret_20d"] > 0,
    "ret_20d < 0":             lambda d: d["ret_20d"] < 0,
    "ret_20d < -10%":          lambda d: d["ret_20d"] < -0.10,

    # === TREND & MA ===
    "px > sma20":              lambda d: d["px_vs_sma20"] > 0,
    "px < sma20":              lambda d: d["px_vs_sma20"] < 0,
    "px dekat sma20 (±2%)":    lambda d: d["px_vs_sma20"].abs() < 0.02,
    "px > 2% di atas sma20":   lambda d: d["px_vs_sma20"] > 0.02,
    "ma20 > ma50":             lambda d: d["ma20_gt_ma50"] == 1,
    "ma20 < ma50":             lambda d: d["ma20_gt_ma50"] == 0,
    "ma50 > ma200":            lambda d: d["ma50_gt_ma200"] == 1,

    # === RSI ===
    "rsi_14 < 30 (oversold)":  lambda d: d["rsi_14"] < 30,
    "rsi_14 < 40":             lambda d: d["rsi_14"] < 40,
    "rsi_14 40-60 (netral)":   lambda d: d["rsi_14"].between(40, 60),
    "rsi_14 > 60":             lambda d: d["rsi_14"] > 60,
    "rsi_14 > 70 (overbought)": lambda d: d["rsi_14"] > 70,

    # === VOLATILITAS ===
    "atr_pct di 30% terendah":  lambda d: d["atr_pct_rank_60"] < 0.3,
    "atr_pct di 20% terendah":  lambda d: d["atr_pct_rank_60"] < 0.2,
    "atr_pct di 30% tertinggi": lambda d: d["atr_pct_rank_60"] > 0.7,
    "bb squeeze (<20% rank)":   lambda d: d["bb_width_pct_rank_60"] < 0.2,
    "NR7 (range tersempit 7 hari)": lambda d: d["nr7"] == 1,
    "NR4 (range tersempit 4 hari)": lambda d: d["nr4"] == 1,

    # === POSISI HARGA ===
    "dekat high 20d (<3%)":    lambda d: d["dist_high_20"] > -0.03,
    "jauh dari high 20d (>10%)": lambda d: d["dist_high_20"] < -0.10,
    "dekat low 20d (<5%)":     lambda d: d["dist_low_20"] < 0.05,
    "naik >10% dari low 20d":  lambda d: d["dist_low_20"] > 0.10,

    # === POLA CANDLE ===
    "higher low 3 hari":       lambda d: d["higher_low_3"] == 1,
    "lower high 3 hari":       lambda d: d["lower_high_3"] == 1,
    "consecutive green >= 2":  lambda d: d["consec_green"] >= 2,
    "consecutive red >= 2":    lambda d: d["consec_red"] >= 2,

    # === AKUMULASI ===
    "vol tinggi harga flat":   lambda d: d["vol_high_price_flat"] == 1,
    "vol naik harga turun":    lambda d: d["vol_up_price_down"] == 1,
    "vol tinggi close tinggi": lambda d: d["vol_high_close_high"] == 1,
    "obv di atas ma10":        lambda d: d["obv_above_ma10"] == 1,
    "cmf positif":             lambda d: d["cmf_20"] > 0,
    "cmf positif saat flat":   lambda d: d["cmf_pos_while_flat"] == 1,
    "mfi oversold (<30)":      lambda d: d["mfi_14"] < 30,
    "mfi overbought (>70)":    lambda d: d["mfi_14"] > 70,
}


# ============================================================
# FUNGSI UTAMA: PROFIL TOP GAINER
# ============================================================
def profile_gainer(df, event_col, conditions, time_offset=1):
    """
    Untuk setiap kejadian top gainer pada hari T, ambil snapshot fitur
    pada hari T-time_offset.

    time_offset=1 → T-1 (sehari sebelum kenaikan)
    time_offset=2 → T-2
    dst.
    """
    df = df.sort_values(["kode", "date"]).reset_index(drop=True)

    # Shift SEMUA kolom numerik fitur ke depan (per saham) sebanyak time_offset.
    # Hasilnya: baris (K, T) akan berisi fitur dari (K, T - time_offset),
    # tapi kolom date & kode tetap menunjukkan (K, T).
    df_shifted = df.copy()
    feature_cols = [
        c for c in df.columns
        if c not in ("date", "kode")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    for c in feature_cols:
        df_shifted[c] = df.groupby("kode")[c].shift(time_offset)

    # Event mask tetap diambil dari df asli (event didefinisikan di hari T)
    event_mask = (df[event_col] == 1).values
    n_event = int(event_mask.sum())

    baseline_mask = ~event_mask
    n_baseline = int(baseline_mask.sum())

    hasil = []
    for name, fn in conditions.items():
        try:
            # Evaluasi kondisi pada df_shifted (fitur T-time_offset)
            cond = fn(df_shifted).fillna(False).values.astype(bool)

            n_match_event = int((cond & event_mask).sum())
            n_match_base = int((cond & baseline_mask).sum())

            coverage = n_match_event / max(n_event, 1)
            baseline_rate = n_match_base / max(n_baseline, 1)
            lift = coverage / baseline_rate if baseline_rate > 0 else np.nan

            hasil.append({
                "condition": name,
                "coverage": coverage,
                "baseline": baseline_rate,
                "lift": lift,
                "n_match_event": n_match_event,
            })
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    df_res = pd.DataFrame(hasil)
    df_res = df_res[df_res["coverage"] >= MIN_COVERAGE].copy()
    df_res = df_res[df_res["lift"] >= MIN_LIFT].copy()
    df_res = df_res.sort_values("lift", ascending=False).reset_index(drop=True)
    return df_res, n_event


# ============================================================
# FUNGSI: CARI KOMBINASI KARAKTERISTIK
# ============================================================
def cari_kombinasi(df, event_col, top_conditions, time_offset=1,
                   max_kombinasi=3, min_coverage=0.10):
    """
    Cari kombinasi top-N kondisi yang paling sering muncul di top gainer
    pada T-time_offset. Skip kombinasi redundan (nested conditions).
    """
    import itertools

    df = df.sort_values(["kode", "date"]).reset_index(drop=True)

    # Shift fitur
    df_shifted = df.copy()
    feature_cols = [
        c for c in df.columns
        if c not in ("date", "kode")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    for c in feature_cols:
        df_shifted[c] = df.groupby("kode")[c].shift(time_offset)

    event_mask = (df[event_col] == 1).values
    n_event = int(event_mask.sum())

    # Pre-compute kondisi
    cond_arrays = {}
    for name in top_conditions:
        try:
            cond_arrays[name] = CONDITIONS[name](df_shifted).fillna(False).values.astype(bool)
        except Exception:
            continue

    n_baseline = int((~event_mask).sum())

    hasil = []
    for k in range(2, max_kombinasi + 1):
        for combo in itertools.combinations(cond_arrays.keys(), k):
            mask = cond_arrays[combo[0]].copy()
            for c in combo[1:]:
                mask &= cond_arrays[c]

            # Skip kalau redundan: kombinasi tidak mengurangi match size
            # dibanding salah satu kondisi tunggalnya
            redundan = False
            for c in combo:
                if (mask == cond_arrays[c]).all():
                    redundan = True
                    break
            if redundan:
                continue

            n_match_event = int((mask & event_mask).sum())
            coverage = n_match_event / max(n_event, 1)
            if coverage < min_coverage:
                continue

            n_match_base = int((mask & ~event_mask).sum())
            baseline = n_match_base / max(n_baseline, 1)
            lift = coverage / baseline if baseline > 0 else np.nan

            hasil.append({
                "kombinasi": " AND ".join(combo),
                "n_conditions": k,
                "coverage": coverage,
                "baseline": baseline,
                "lift": lift,
                "n_match_event": n_match_event,
            })

    if not hasil:
        return pd.DataFrame()
    df_res = pd.DataFrame(hasil)
    df_res = df_res.sort_values("lift", ascending=False).reset_index(drop=True)
    return df_res


# ============================================================
# PROGRAM UTAMA
# ============================================================
def main():
    print("Memuat data...")
    panel = load_panel_from_gabungan()
    panel = clean_panel(panel)
    panel = build_features(panel)
    panel = detect_events(panel)

    for gainer_name, event_col in GAINER_DEFINITIONS.items():
        print(f"\n{'='*70}")
        print(f"PROFIL TOP GAINER: {gainer_name} ({event_col})")
        print(f"{'='*70}")

        for lag in [1, 2, 3, 5, 10]:
            print(f"\n--- Karakteristik pada T-{lag} ---")
            df_res, n_event = profile_gainer(
                panel, event_col, CONDITIONS, time_offset=lag
            )
            if df_res.empty:
                print(f"  Tidak ada kondisi yang lolos filter pada T-{lag}.")
                continue

            show = df_res.head(15).copy()
            show["coverage"] = (show["coverage"] * 100).round(1)
            show["baseline"] = (show["baseline"] * 100).round(2)
            show["lift"] = show["lift"].round(2)
            show = show.rename(columns={
                "coverage": "coverage_%",
                "baseline": "baseline_%",
            })
            print(show.to_string(index=False))

            out = f"quant/output/profile_{gainer_name}_T{lag}.csv"
            df_res.to_csv(out, index=False)
            print(f"  → Disimpan: {out}")


if __name__ == "__main__":
    main()