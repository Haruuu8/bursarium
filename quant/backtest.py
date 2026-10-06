"""
Cari kesamaan saham di HARI SEBELUM mereka jadi top gainer.

Alur:
1. Ambil semua event top gainer (hari T)
2. Untuk setiap event, lihat fitur di T-1 (hari sebelumnya)
3. Hitung: fitur apa yang paling sering muncul di T-1?
4. Tampilkan contoh konkret: 10 top gainer, fitur T-1 mereka
"""

import numpy as np
import pandas as pd

from .data_loader import load_panel_from_gabungan
from .data_cleaner import clean_panel
from .feature_engineering import build_features
from .event_detector import detect_events


EVENT_TARGET = "is_touch_10pct"   # top gainer = menyentuh +10%
TARGET_DATE = None                 # None = semua tanggal. Isi '2026-09-29' untuk tanggal tertentu.


# Semua fitur yang mau dilihat di T-1
FITUR_DISPLAY = [
    # MOMENTUM
    "ret_1d", "ret_5d", "ret_20d",
    # VOLUME
    "vol_ratio_5", "vol_ratio_20", "vol_z_20",
    # TREND
    "px_vs_sma20", "dist_high_20", "dist_low_20",
    # RSI / MOMENTUM
    "rsi_14", "mfi_14",
    # PRICE ACTION
    "clv", "gap", "body", "range",
    # VOLATILITAS
    "atr_pct_rank_60", "bb_width_pct_rank_60",
    # KONTEKS
    "close", "value",
]

# Kondisi threshold untuk hitung coverage
CONDITIONS = {
    "ret_1d > 0":             lambda d: d["ret_1d"] > 0,
    "ret_1d > 2%":            lambda d: d["ret_1d"] > 0.02,
    "ret_1d > 5%":            lambda d: d["ret_1d"] > 0.05,
    "ret_5d > 0":             lambda d: d["ret_5d"] > 0,
    "ret_5d > 5%":            lambda d: d["ret_5d"] > 0.05,
    "ret_5d > 10%":           lambda d: d["ret_5d"] > 0.10,
    "ret_20d > 0":            lambda d: d["ret_20d"] > 0,
    "ret_20d < -10%":         lambda d: d["ret_20d"] < -0.10,
    "vol_ratio_5 > 1.5":      lambda d: d["vol_ratio_5"] > 1.5,
    "vol_ratio_5 > 2.0":      lambda d: d["vol_ratio_5"] > 2.0,
    "vol_ratio_5 > 3.0":      lambda d: d["vol_ratio_5"] > 3.0,
    "vol_ratio_20 > 1.5":     lambda d: d["vol_ratio_20"] > 1.5,
    "vol_ratio_20 > 2.0":     lambda d: d["vol_ratio_20"] > 2.0,
    "vol_ratio_20 > 3.0":     lambda d: d["vol_ratio_20"] > 3.0,
    "vol_z_20 > 2":           lambda d: d["vol_z_20"] > 2,
    "px > sma20":             lambda d: d["px_vs_sma20"] > 0,
    "px > 2% above sma20":    lambda d: d["px_vs_sma20"] > 0.02,
    "px < sma20":             lambda d: d["px_vs_sma20"] < 0,
    "rsi_14 > 60":            lambda d: d["rsi_14"] > 60,
    "rsi_14 > 70":            lambda d: d["rsi_14"] > 70,
    "rsi_14 < 40":            lambda d: d["rsi_14"] < 40,
    "rsi_14 < 30":            lambda d: d["rsi_14"] < 30,
    "close > open":           lambda d: d["close"] > d["open"],
    "clv > 0.6":              lambda d: d["clv"] > 0.6,
    "clv > 0.7":              lambda d: d["clv"] > 0.7,
    "gap up > 1%":            lambda d: d["gap"] > 0.01,
    "gap down < -1%":         lambda d: d["gap"] < -0.01,
    "dekat high 20d":         lambda d: d["dist_high_20"] > -0.03,
    "jauh dari high 20d":     lambda d: d["dist_high_20"] < -0.10,
    "naik >10% dari low 20d": lambda d: d["dist_low_20"] > 0.10,
    "atr low (<30%)":         lambda d: d["atr_pct_rank_60"] < 0.3,
    "atr high (>70%)":        lambda d: d["atr_pct_rank_60"] > 0.7,
    "bb squeeze":             lambda d: d["bb_width_pct_rank_60"] < 0.2,
    "vol tinggi close tinggi":lambda d: d["vol_high_close_high"] == 1,
}


def main():
    print("=" * 100)
    print(f"PRE-GAINER ANALYSIS - apa kesamaan saham di T-1 sebelum jadi top gainer?")
    print(f"Target event: {EVENT_TARGET} (menyentuh +10% di hari T)")
    print("=" * 100)

    # Load
    print("\nLoad data...")
    panel = load_panel_from_gabungan()
    panel = clean_panel(panel)
    panel = build_features(panel)
    panel = detect_events(panel)
    panel = panel.sort_values(["kode", "date"]).reset_index(drop=True)

    # ============================================================
    # Bangun dataset "T-1 features dari event di T"
    # ============================================================
    # Setiap baris punya: fitur hari T + event_next (event T+1)
    # Jadi event_next=1 di baris T berarti "T+1 top gainer".
    # Fitur di baris T = fitur T-1 dari event di T+1.
    panel["event_next"] = (
        panel.groupby("kode")[EVENT_TARGET]
             .shift(-1).fillna(0).astype(int)
    )

    if TARGET_DATE is not None:
        mask_date = panel["date"] == pd.Timestamp(TARGET_DATE)
        sub = panel[mask_date]
        print(f"\nFilter tanggal: {TARGET_DATE} ({len(sub)} saham)")
        events = sub[sub["event_next"] == 1].copy()
    else:
        events = panel[panel["event_next"] == 1].copy()

    n_events = len(events)
    print(f"\nTotal event top gainer: {n_events:,}")

    if n_events == 0:
        print("Tidak ada event.")
        return

    # ============================================================
    # SECTION 1: Contoh 10 konkret
    # ============================================================
    print(f"\n{'='*100}")
    print("CONTOH 10 TOP GAINER - fitur mereka di T-1 (hari sebelum naik)")
    print(f"{'='*100}\n")

    contoh = events.sort_values("date", ascending=False).head(10).copy()
    for _, row in contoh.iterrows():
        tgl_event = row["date"] + pd.Timedelta(days=1) if TARGET_DATE is None else row["date"]
        print(f"[{row['date'].date()}] {row['kode']:6s}  -> besok top gainer")
        print(f"  Close T-1    : Rp {row['close']:.0f}")
        print(f"  Ret 1d/5d/20d: {row['ret_1d']*100:+.1f}% / {row['ret_5d']*100:+.1f}% / {row['ret_20d']*100:+.1f}%")
        print(f"  Vol ratio 5/20: {row['vol_ratio_5']:.2f}x / {row['vol_ratio_20']:.2f}x")
        print(f"  RSI(14)      : {row['rsi_14']:.1f}")
        print(f"  Px vs SMA20  : {row['px_vs_sma20']*100:+.1f}%")
        print(f"  Dist from high20: {row['dist_high_20']*100:+.1f}%")
        print(f"  CLV / Gap    : {row['clv']:.2f} / {row['gap']*100:+.1f}%")
        print()

    # ============================================================
    # SECTION 2: Kesamaan (coverage semua kondisi di T-1)
    # ============================================================
    print(f"{'='*100}")
    print("KESAMAAN - berapa % top gainer yang punya kondisi ini di T-1?")
    print(f"{'='*100}\n")

    # Baseline: coverage di semua saham (bukan event)
    non_events = panel[panel["event_next"] == 0]

    hasil = []
    for name, fn in CONDITIONS.items():
        # Coverage di events
        try:
            cond_ev = fn(events).fillna(False)
            cov = cond_ev.mean()
            n_match = int(cond_ev.sum())

            # Baseline di non-events
            cond_nonev = fn(non_events).fillna(False)
            base = cond_nonev.mean()

            lift = cov / base if base > 0 else 0
        except Exception as e:
            continue

        hasil.append({
            "kondisi": name,
            "coverage_%": cov * 100,
            "baseline_%": base * 100,
            "lift": lift,
            "n_match": n_match,
        })

    df_res = pd.DataFrame(hasil)
    df_res = df_res.sort_values("lift", ascending=False).reset_index(drop=True)

    # Tampilkan
    print(f"{'Kondisi':<25} {'Coverage%':>10} {'Baseline%':>10} {'Lift':>7} {'n_match':>8}")
    print("-" * 65)
    for _, r in df_res.iterrows():
        if r["coverage_%"] >= 5:  # hanya tampilkan yang coverage >= 5%
            print(f"{r['kondisi']:<25} {r['coverage_%']:>9.1f}% "
                  f"{r['baseline_%']:>9.2f}% {r['lift']:>6.2f}x {r['n_match']:>8}")

    # ============================================================
    # SECTION 3: Cari kondisi yang coverage TINGGI
    # ============================================================
    print(f"\n{'='*100}")
    print("KONDISI PALING UMUM (coverage >= 30% dari top gainer)")
    print(f"{'='*100}\n")

    umum = df_res[df_res["coverage_%"] >= 30].copy()
    umum = umum.sort_values("coverage_%", ascending=False)
    if umum.empty:
        print("Tidak ada kondisi dengan coverage >= 30%.")
    else:
        print(f"{'Kondisi':<25} {'Coverage%':>10} {'Baseline%':>10} {'Lift':>7}")
        print("-" * 55)
        for _, r in umum.iterrows():
            print(f"{r['kondisi']:<25} {r['coverage_%']:>9.1f}% "
                  f"{r['baseline_%']:>9.2f}% {r['lift']:>6.2f}x")

    # ============================================================
    # SECTION 4: Statistik agregat
    # ============================================================
    print(f"\n{'='*100}")
    print("STATISTIK AGREGAT top gainer di T-1 (median)")
    print(f"{'='*100}\n")

    for col in FITUR_DISPLAY:
        if col not in events.columns:
            continue
        ev_med = events[col].median()
        ne_med = non_events[col].median()
        print(f"  {col:<25}: event={ev_med:>10.4f}   non-event={ne_med:>10.4f}   "
              f"ratio={ev_med/ne_med if ne_med != 0 else 0:.2f}")

    # Simpan
    df_res.to_csv("quant/output/pre_gainer_kesamaan.csv", index=False)
    events[["date", "kode"] + FITUR_DISPLAY].to_csv(
        "quant/output/pre_gainer_events.csv", index=False)
    print(f"\n-> Detail disimpan: quant/output/pre_gainer_kesamaan.csv")
    print(f"                     quant/output/pre_gainer_events.csv")


if __name__ == "__main__":
    main()