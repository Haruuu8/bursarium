"""
KOMPOSIT SINYAL PREDIKSI IHSG 1 HARI
- 6 sinyal teori-driven
- Digabung jadi 1 skor
- Tes sebagai 1 hipotesis (bebas FDR)
- Walk-forward untuk validasi
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from datetime import datetime


# ============================================================
# KONFIGURASI
# ============================================================
TICKER_IHSG = "^JKSE"
TICKER_FACTORS = {
    "DowJones": "^DJI", "SP500": "^GSPC", "Nasdaq": "^IXIC",
    "Nikkei": "^N225", "HangSeng": "^HSI", "USD_IDR": "IDR=X",
    "DXY": "DX-Y.NYB", "VIX": "^VIX",
}
LOOKBACK_DAYS = 2500
N_WALK = 5


# ============================================================
# FETCH
# ============================================================
def fetch_data():
    print("Fetching data...")
    frames = {}

    h = yf.Ticker(TICKER_IHSG).history(period=f"{LOOKBACK_DAYS}d")
    if h.empty:
        raise RuntimeError("IHSG kosong")
    h = h[["Close", "Open", "High", "Low"]].rename(columns={
        "Close": "IHSG", "Open": "IHSG_Open",
        "High": "IHSG_High", "Low": "IHSG_Low"})
    frames["IHSG"] = h
    print(f"  IHSG: {len(h)} hari")

    for nama, ticker in TICKER_FACTORS.items():
        try:
            d = yf.Ticker(ticker).history(period=f"{LOOKBACK_DAYS}d")
            if not d.empty:
                frames[nama] = d[["Close"]].rename(columns={"Close": nama})
        except Exception:
            pass

    def norm(idx):
        idx = pd.to_datetime(idx)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        return pd.DatetimeIndex(idx.normalize()).as_unit("ns")

    for k in frames:
        frames[k].index = norm(frames[k].index)
        frames[k] = frames[k][~frames[k].index.duplicated(keep="last")]

    df = pd.concat(frames.values(), axis=1).sort_index()
    for c in df.columns:
        if not c.startswith("IHSG"):
            df[c] = df[c].ffill()
    df = df[df["IHSG"].notna()].copy()
    return df


# ============================================================
# FITUR
# ============================================================
def build_features(df):
    df = df.copy()

    # Target
    df["ret_1d"] = df["IHSG"].pct_change()
    df["ret_tomorrow"] = df["ret_1d"].shift(-1)
    df["target_up"] = (df["ret_tomorrow"] > 0).astype(int)

    # RSI 14
    delta = df["IHSG"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)

    # MA
    df["ma20"] = df["IHSG"].rolling(20).mean()
    df["ma50"] = df["IHSG"].rolling(50).mean()
    df["above_ma50"] = (df["IHSG"] > df["ma50"]).astype(int)

    # Vol
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    df["vol_rank"] = df["vol_20d"].rolling(252).rank(pct=True)

    # Lead global
    for col in TICKER_FACTORS.keys():
        if col in df.columns:
            df[f"{col}_lead1"] = df[col].pct_change().shift(1)

    # US composite
    us_cols = [c for c in ["SP500_lead1", "Nasdaq_lead1"] if c in df.columns]
    if us_cols:
        df["us_lead"] = df[us_cols].mean(axis=1)

    # Asia composite
    asia_cols = [c for c in ["Nikkei_lead1", "HangSeng_lead1"] if c in df.columns]
    if asia_cols:
        df["asia_lead"] = df[asia_cols].mean(axis=1)

    return df


# ============================================================
# 6 SINYAL BERBASIS TEORI
# ============================================================
def compute_signals(df):
    sig = pd.DataFrame(index=df.index)

    # 1. US overnight
    if "us_lead" in df.columns:
        sig["s1_us"] = np.where(df["us_lead"] > 0.005, 1,
                       np.where(df["us_lead"] < -0.005, -1, 0))
    else:
        sig["s1_us"] = 0

    # 2. Asia overnight
    if "asia_lead" in df.columns:
        sig["s2_asia"] = np.where(df["asia_lead"] > 0.005, 1,
                         np.where(df["asia_lead"] < -0.005, -1, 0))
    else:
        sig["s2_asia"] = 0

    # 3. Mean reversion
    sig["s3_meanrev"] = np.where(df["ret_1d"] < -0.01, 1,
                        np.where(df["ret_1d"] > 0.01, -1, 0))

    # 4. RSI ekstrem
    sig["s4_rsi"] = np.where(df["rsi14"] < 30, 1,
                    np.where(df["rsi14"] > 70, -1, 0))

    # 5. DXY overnight
    if "DXY_lead1" in df.columns:
        sig["s5_dxy"] = np.where(df["DXY_lead1"] < -0.003, 1,
                        np.where(df["DXY_lead1"] > 0.003, -1, 0))
    else:
        sig["s5_dxy"] = 0

    # 6. VIX overnight
    if "VIX_lead1" in df.columns:
        sig["s6_vix"] = np.where(df["VIX_lead1"] < -0.03, 1,
                        np.where(df["VIX_lead1"] > 0.03, -1, 0))
    else:
        sig["s6_vix"] = 0

    sig["score"] = sig.sum(axis=1)
    return sig


# ============================================================
# UJI KOMPOSIT
# ============================================================
def test_composite(df, sig, label=""):
    sub = pd.concat([sig["score"], df["target_up"]], axis=1).dropna()
    sub.columns = ["score", "target_up"]
    base = sub["target_up"].mean()

    print(f"\n{'='*60}")
    print(f"UJI KOMPOSIT {label}")
    print(f"{'='*60}")
    print(f"Base rate hijau: {base*100:.1f}%  (n={len(sub)})")
    print(f"\n{'Skor':>6} {'N':>6} {'Hijau':>8} {'Lift':>8} {'p-val':>8}")
    print("-" * 60)

    results = []
    for k in sorted(sub["score"].unique()):
        s = sub[sub["score"] == k]
        if len(s) < 20:
            continue
        p = s["target_up"].mean()
        kk = int(round(p * len(s)))
        try:
            pval = stats.binomtest(kk, len(s), base, alternative="two-sided").pvalue
        except Exception:
            pval = 1.0
        results.append({"score": k, "n": len(s), "p_green": p, "pval": pval})
        mark = "⭐" if pval < 0.05 else "  "
        print(f"{mark} {k:>4} {len(s):>6} {p*100:>7.1f}% "
              f"{(p-base)*100:>+7.1f}pp {pval:>8.3f}")

    if len(results) > 2:
        scores = [r["score"] for r in results]
        greens = [r["p_green"] for r in results]
        corr, pcorr = stats.spearmanr(scores, greens)
        print(f"\nKorelasi skor vs P(hijau): rho={corr:.3f}, p={pcorr:.4f}")
        if pcorr < 0.10 and corr > 0:
            print("  ➡️  MONOTONIK: skor tinggi → hijau, skor rendah → merah ⭐")
        elif pcorr < 0.10 and corr < 0:
            print("  ➡️  MONOTONIK TERBALIK")
        else:
            print("  ➡️  Tidak monotonik")

    return results, base


# ============================================================
# UJI EKSTREM
# ============================================================
def test_extreme(df, sig):
    sub = pd.concat([sig["score"], df["target_up"]], axis=1).dropna()
    sub.columns = ["score", "target_up"]
    base = sub["target_up"].mean()

    print(f"\n{'='*60}")
    print("UJI EKSTREM (skor >= 2 vs skor <= -2)")
    print(f"{'='*60}")

    bull = sub[sub["score"] >= 2]
    bear = sub[sub["score"] <= -2]
    neutral = sub[sub["score"].abs() < 2]

    for label, s in [("BULL (>=2)", bull), ("BEAR (<=-2)", bear), ("NEUTRAL", neutral)]:
        if len(s) < 10:
            continue
        p = s["target_up"].mean()
        k = int(round(p * len(s)))
        try:
            pval = stats.binomtest(k, len(s), base, alternative="two-sided").pvalue
        except Exception:
            pval = 1.0
        mark = "⭐" if pval < 0.05 else "  "
        print(f"{mark} {label:<14} N={len(s):>5}  hijau {p*100:>5.1f}%  "
              f"lift {(p-base)*100:>+5.1f}pp  p={pval:.3f}")

    return bull, bear


# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward(df, sig, n_periods=5):
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD KOMPOSIT ({n_periods} periode)")
    print(f"{'='*60}")

    chunks = np.array_split(np.arange(len(df)), n_periods)
    print(f"\n{'Periode':<10} {'N':>5} {'Bull hijau':>18} {'Bear hijau':>18}")
    print("-" * 60)

    for i, chunk in enumerate(chunks):
        sub_sig = sig.iloc[chunk]
        sub_df = df.iloc[chunk]
        comb = pd.concat([sub_sig["score"], sub_df["target_up"]], axis=1).dropna()
        comb.columns = ["score", "target_up"]

        bull = comb[comb["score"] >= 2]
        bear = comb[comb["score"] <= -2]

        b_str = (f"{bull['target_up'].mean()*100:>5.1f}% (n={len(bull)})"
                 if len(bull) > 5 else "  -")
        r_str = (f"{bear['target_up'].mean()*100:>5.1f}% (n={len(bear)})"
                 if len(bear) > 5 else "  -")
        print(f"  P{i+1:<8} {len(comb):>5} {b_str:>18} {r_str:>18}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("KOMPOSIT SINYAL PREDIKSI IHSG 1 HARI")
    print(f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    df = fetch_data()
    df = build_features(df)

    # Drop NaN & simpan tanggal sebelum reset index
    df = df.dropna(subset=["ma50", "target_up", "rsi14"]).copy()
    tanggal_awal = df.index[0]
    tanggal_akhir = df.index[-1]
    df = df.reset_index(drop=True)

    print(f"\nData final: {len(df)} hari")
    print(f"Periode: {tanggal_awal.date()} s/d {tanggal_akhir.date()}")

    sig = compute_signals(df)

    # Distribusi skor
    print("\nDistribusi skor komposit:")
    for k in sorted(sig["score"].unique()):
        n = (sig["score"] == k).sum()
        bar = "█" * int(n / len(sig) * 50)
        print(f"  {k:>3}: {n:>4}  {bar}")

    # Uji
    results, base = test_composite(df, sig, "(SELURUH DATA)")
    bull, bear = test_extreme(df, sig)

    # Walk-forward
    walk_forward(df, sig, N_WALK)

    # ========================================================
    # PREDIKSI HARI INI
    # ========================================================
    print(f"\n{'='*60}")
    print("PREDIKSI BESOK (pakai komposit)")
    print(f"{'='*60}")

    last = sig.iloc[-1]
    print("\nSinyal aktif:")
    signal_names = {
        "s1_us": "US overnight",
        "s2_asia": "Asia overnight",
        "s3_meanrev": "Mean reversion (kemarin gerak >1%)",
        "s4_rsi": "RSI ekstrem",
        "s5_dxy": "DXY overnight",
        "s6_vix": "VIX overnight",
    }
    for s, name in signal_names.items():
        v = last[s]
        if v > 0:
            print(f"  🟢 {name:<40} +1")
        elif v < 0:
            print(f"  🔴 {name:<40} -1")
        else:
            print(f"  ⚪ {name:<40}  0")

    print(f"\n  SKOR KOMPOSIT: {int(last['score']):+d}")

    if last["score"] >= 2:
        print("  ➡️  Prediksi: 🟢 HIJAU (bullish)")
    elif last["score"] <= -2:
        print("  ➡️  Prediksi: 🔴 MERAH (bearish)")
    else:
        print("  ➡️  Prediksi: 🟡 NETRAL (tidak ada edge jelas)")

    # Cek historis untuk skor ini
    hist = pd.concat([sig["score"], df["target_up"]], axis=1).dropna()
    hist.columns = ["score", "target_up"]
    same_score = hist[hist["score"] == last["score"]]
    if len(same_score) > 10:
        p = same_score["target_up"].mean()
        print(f"\n  Historis, ketika skor = {int(last['score'])}:")
        print(f"    → {p*100:.1f}% hijau (n={len(same_score)})")


if __name__ == "__main__":
    main()