"""
DETEKSI NEGO ANOMALI - Cari nego yang tidak wajar.
Nego anomali sering muncul SEBELUM news.

Konsep:
- Nego normal saham itu = bandingkan dengan history sendiri
- Nego anomali = 3x, 5x, 10x lebih besar dari biasanya
- Kategorikan berdasarkan tipe event yang mungkin
"""

import os
import numpy as np
import pandas as pd

from .data_loader import load_panel_from_gabungan
from .data_cleaner import clean_panel
from .feature_engineering import build_features
from .event_detector import detect_events


def add_features(df):
    df = df.copy()
    df = df.sort_values(["kode", "date"]).reset_index(drop=True)

    df["nego_value_B"] = df["NonRegularValue"] / 1e9
    df["nego_avg_price"] = df["NonRegularValue"] / df["NonRegularVolume"].replace(0, np.nan)
    df["nego_premium"] = df["nego_avg_price"] / df["close"] - 1

    # ============================================================
    # NORMALISASI PER SAHAM (bandingkan dengan history sendiri)
    # ============================================================
    # Rata-rata nego 60 hari (exclude hari ini)
    df["nego_avg_60d"] = df.groupby("kode")["NonRegularValue"].transform(
        lambda s: s.shift(1).rolling(60, min_periods=10).mean())

    # Rata-rata nego 60 hari (hanya hari yang ada nego)
    df["nego_avg_active_60d"] = df.groupby("kode")["NonRegularValue"].transform(
        lambda s: s.shift(1).rolling(60, min_periods=5).apply(
            lambda x: x[x > 0].mean() if (x > 0).any() else 0, raw=True))

    # Max nego 60 hari (exclude hari ini)
    df["nego_max_60d"] = df.groupby("kode")["NonRegularValue"].transform(
        lambda s: s.shift(1).rolling(60, min_periods=5).max())

    # Ratio nego hari ini vs rata-rata 60d
    df["nego_ratio_vs_avg"] = df["NonRegularValue"] / df["nego_avg_active_60d"].replace(0, np.nan)

    # Ratio vs max historis
    df["nego_ratio_vs_max"] = df["NonRegularValue"] / df["nego_max_60d"].replace(0, np.nan)

    # ============================================================
    # KONTEKS
    # ============================================================
    df["vol_ma20"] = df.groupby("kode")["volume"].transform(
        lambda s: s.rolling(20, min_periods=5).mean())
    df["vol_ratio"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)

    # Harga vs range
    df["high_60d"] = df.groupby("kode")["high"].transform(
        lambda s: s.rolling(60, min_periods=20).max())
    df["low_60d"] = df.groupby("kode")["low"].transform(
        lambda s: s.rolling(60, min_periods=20).min())
    df["pos_in_60d"] = (df["close"] - df["low_60d"]) / (
        df["high_60d"] - df["low_60d"]).replace(0, np.nan)

    # Forward
    for h in [5, 10, 20, 60]:
        df[f"close_fwd_{h}d"] = df.groupby("kode")["close"].shift(-h)
        df[f"high_fwd_{h}d"] = df.groupby("kode")["high"].shift(-h)

    return df


def klasifikasi_anomali(row):
    """
    Klasifikasi tipe anomali nego.
    Return: dict dengan tipe, confidence, alasan.
    """
    kode = row["kode"]
    nego_val = row["NonRegularValue"]
    ratio_avg = row.get("nego_ratio_vs_avg", np.nan)
    ratio_max = row.get("nego_ratio_vs_max", np.nan)
    premium = row.get("nego_premium", np.nan)
    pos_60d = row.get("pos_in_60d", np.nan)
    vol_ratio = row.get("vol_ratio", np.nan)
    close = row["close"]
    value = row["value"]

    alasan = []
    confidence = 0
    tipe = "NORMAL"

    # ============================================================
    # 1. ANOMALI SIZE
    # ============================================================
    if not pd.isna(ratio_max) and ratio_max > 3:
        alasan.append(f"Nego {ratio_max:.1f}x lebih besar dari MAX 60 hari")
        confidence += 40
    elif not pd.isna(ratio_max) and ratio_max > 1.5:
        alasan.append(f"Nego {ratio_max:.1f}x lebih besar dari max 60 hari")
        confidence += 25

    if not pd.isna(ratio_avg) and ratio_avg > 10:
        alasan.append(f"Nego {ratio_avg:.1f}x rata-rata historis")
        confidence += 30
    elif not pd.isna(ratio_avg) and ratio_avg > 5:
        alasan.append(f"Nego {ratio_avg:.1f}x rata-rata historis")
        confidence += 20
    elif not pd.isna(ratio_avg) and ratio_avg > 3:
        alasan.append(f"Nego {ratio_avg:.1f}x rata-rata historis")
        confidence += 10

    # ============================================================
    # 2. ANOMALI PREMIUM
    # ============================================================
    if not pd.isna(premium):
        if premium > 0.30:
            alasan.append(f"Nego di PREMIUM +{premium*100:.1f}% (SANGAT AGGRESSIVE)")
            confidence += 35
        elif premium > 0.10:
            alasan.append(f"Nego di premium +{premium*100:.1f}% (aggressive)")
            confidence += 20
        elif premium > 0.05:
            alasan.append(f"Nego di premium +{premium*100:.1f}%")
            confidence += 10
        elif premium < -0.30:
            alasan.append(f"Nego di DISCOUNT {premium*100:.1f}% (terpaksa)")
            confidence += 5  # discount bukan sinyal positif
        elif premium < -0.10:
            alasan.append(f"Nego di discount {premium*100:.1f}%")
            confidence += 2

    # ============================================================
    # 3. KONTEKS HARGA
    # ============================================================
    if not pd.isna(pos_60d):
        if pos_60d < 0.3:
            alasan.append(f"Harga di bawah (posisi {pos_60d*100:.0f}% range 60d) - nego dari bawah")
            confidence += 10
        elif pos_60d > 0.8:
            alasan.append(f"Harga di atas (posisi {pos_60d*100:.0f}% range 60d)")
            confidence -= 5  # risiko distribusi

    # ============================================================
    # 4. VOLUME NEGO VS REGULER
    # ============================================================
    total_value = value + nego_val
    if total_value > 0:
        nego_share = nego_val / total_value
        if nego_share > 0.8:
            alasan.append(f"Nego {nego_share*100:.0f}% dari total value (dominan)")
            confidence += 15
        elif nego_share > 0.5:
            alasan.append(f"Nego {nego_share*100:.0f}% dari total value")
            confidence += 8

    # ============================================================
    # 5. KLASIFIKASI TIPE
    # ============================================================
    if confidence >= 70:
        tipe = "🔴 SANGAT ANOMALI - Kemungkinan besar ada event besar"
    elif confidence >= 50:
        tipe = "🟠 ANOMALI - Kemungkinan ada news/corporate action"
    elif confidence >= 30:
        tipe = "🟡 ADA SINYAL - Perlu dipantau"
    else:
        tipe = "⚪ NORMAL"

    return {
        "tipe": tipe,
        "confidence": confidence,
        "alasan": alasan,
    }


def main():
    print("=" * 110)
    print("DETEKSI NEGO ANOMALI - Cari jejak news sebelum news")
    print("=" * 110)

    panel = load_panel_from_gabungan()
    panel = clean_panel(panel)
    panel = build_features(panel)
    panel = detect_events(panel)
    panel = add_features(panel)

    # Ambil data terbaru
    last_date = panel["date"].max()
    today = panel[panel["date"] == last_date].copy()

    # Filter hanya yang ada nego
    today_nego = today[today["NonRegularValue"] > 0].copy()

    print(f"\nTanggal: {pd.Timestamp(last_date).date()}")
    print(f"Total saham: {len(today)}")
    print(f"Ada nego: {len(today_nego)}")

    if len(today_nego) == 0:
        print("\nTidak ada nego hari ini.")
        return

    # Klasifikasi anomali
    results = []
    for _, row in today_nego.iterrows():
        try:
            k = klasifikasi_anomali(row)
            results.append({
                "kode": row["kode"],
                "close": row["close"],
                "nego_value_B": row["NonRegularValue"] / 1e9,
                "nego_premium": row["nego_premium"] * 100 if not pd.isna(row["nego_premium"]) else 0,
                "ratio_avg": row["nego_ratio_vs_avg"] if not pd.isna(row["nego_ratio_vs_avg"]) else 0,
                "tipe": k["tipe"],
                "confidence": k["confidence"],
                "alasan": k["alasan"],
            })
        except Exception:
            pass

    df = pd.DataFrame(results).sort_values("confidence", ascending=False)

    # ============================================================
    # TOP ANOMALI HARI INI
    # ============================================================
    print(f"\n{'='*110}")
    print("🚨 NEGO ANOMALI HARI INI (Top 10)")
    print(f"{'='*110}\n")

    top = df[df["confidence"] >= 30].head(10)

    if len(top) == 0:
        print("  Tidak ada nego anomali signifikan hari ini.")
    else:
        for _, r in top.iterrows():
            print(f"\n{'━'*95}")
            print(f"{r['kode']}  |  Close: Rp {r['close']:.0f}  |  "
                  f"Confidence: {r['confidence']}/100")
            print(f"{r['tipe']}")
            print(f"{'━'*95}")
            print(f"  Nego hari ini: Rp {r['nego_value_B']:.1f}M "
                  f"({r['ratio_avg']:.1f}x rata-rata historis)")

            for a in r["alasan"]:
                print(f"  • {a}")

    # ============================================================
    # HISTORI: CONTOH ANOMALI YANG BENAR ADA NEWS
    # ============================================================
    print(f"\n{'='*110}")
    print("📊 HISTORI: ANOMALI NEGO + HASILNYA (semua data)")
    print(f"{'='*110}\n")

    # Cari semua nego anomali di data
    panel_nego = panel[panel["NonRegularValue"] > 0].copy()

    # Hitung ratio untuk semua
    panel_nego = panel_nego[
        panel_nego["nego_ratio_vs_max"].notna() &
        (panel_nego["nego_ratio_vs_max"] > 2)
    ].copy()

    if len(panel_nego) > 0:
        valid = panel_nego[panel_nego["close_fwd_20d"].notna()].copy()
        if len(valid) >= 5:
            valid["ret_20d"] = valid["close_fwd_20d"] / valid["close"] - 1
            win_20 = (valid["ret_20d"] > 0).mean() * 100
            med_ret = valid["ret_20d"].median() * 100

            print(f"  Nego anomali (ratio vs max > 2x): {len(valid)} kejadian")
            print(f"  Win rate 20d  : {win_20:.1f}%")
            print(f"  Median return : {med_ret:+.2f}%")
            print()

            # Contoh top anomali
            print(f"  Contoh 10 anomali nego terbesar:")
            print(f"  {'Tanggal':<12} {'Kode':<7} {'Nego':>10} {'Ratio':>7} "
                  f"{'Premium':>9} {'Ret 20d':>9}")
            print("  " + "-" * 70)

            top_hist = valid.nlargest(10, "nego_ratio_vs_max")
            for _, r in top_hist.iterrows():
                ret_20 = r["ret_20d"] * 100
                print(f"  {str(pd.Timestamp(r['date']).date()):<12} "
                      f"{r['kode']:<7} Rp {r['NonRegularValue']/1e9:>6.1f}M "
                      f"{r['nego_ratio_vs_max']:>6.1f}x "
                      f"{r['nego_premium']*100:>+7.2f}% "
                      f"{ret_20:>+8.2f}%")

    # ============================================================
    # KESIMPULAN
    # ============================================================
    print(f"\n{'='*110}")
    print("KESIMPULAN")
    print(f"{'='*110}\n")

    print("""  CARA KERJA DETEKSI:
    1. Nego ANOMALI = nego 3x+ lebih besar dari history saham itu sendiri
    2. Nego PREMIUM = institusi bayar lebih mahal dari pasar (agresif)
    3. KONTEKS = harga di bawah range (akumulasi dari bawah)
    4. KOMBINASI = sinyal kuat

    NEGO ANOMALI SERING MUNCUL SEBELUM NEWS:
    • Akuisisi / merger
    • Private placement / rights issue
    • Strategic investor masuk
    • Buyback
    • Perubahan pemegang saham besar

    CARA PAKAI:
    • Jalankan setiap malam
    • Lihat saham dengan confidence >= 50
    • Cross-check dengan berita (Google, Stockbit)
    • Kalau ada konfirmasi berita → entry
    • Kalau tidak ada berita → tunggu 1-3 hari
    """)

    # Simpan
    os.makedirs("quant/output", exist_ok=True)
    out = f"quant/output/anomali_nego_{pd.Timestamp(last_date).date()}.csv"

    # Detail untuk semua
    df_detail = df[["kode", "close", "nego_value_B", "nego_premium",
                     "ratio_avg", "confidence", "tipe"]].copy()
    df_detail.columns = ["Kode", "Close", "Nego (M)", "Premium %",
                          "Ratio vs Avg", "Confidence", "Tipe"]
    df_detail.to_csv(out, index=False)
    print(f"-> Disimpan: {out}")


if __name__ == "__main__":
    main()