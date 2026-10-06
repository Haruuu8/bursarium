"""
Event study awal: fitur apa yang berbeda signifikan antara top gainer
dan non-top-gainer pada T-1.
"""

import numpy as np
import pandas as pd
from scipy import stats


def compare_features(df: pd.DataFrame, event_col: str,
                     feature_cols: list[str]) -> pd.DataFrame:
    """
    Bandingkan distribusi fitur antara baris event=1 vs event=0.
    Hanya memproses kolom numerik.
    """
    # Filter hanya kolom numerik yang benar-benar ada
    numeric_cols = [
        c for c in feature_cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    skipped = [c for c in feature_cols if c not in numeric_cols]
    if skipped:
        print(f"  ⚠️ Skip {len(skipped)} kolom non-numerik: {skipped[:10]}"
              + (" ..." if len(skipped) > 10 else ""))

    hasil = []
    event = df[df[event_col] == 1]
    non = df[df[event_col] == 0]

    for col in numeric_cols:
        ev = event[col].dropna()
        nn = non[col].dropna()
        if len(ev) < 30 or len(nn) < 100:
            continue

        mean_ev, mean_nn = ev.mean(), nn.mean()
        std_ev, std_nn = ev.std(), nn.std()
        n_ev, n_nn = len(ev), len(nn)

        # Cohen's d
        pooled_std = np.sqrt(
            ((n_ev - 1) * std_ev**2 + (n_nn - 1) * std_nn**2)
            / (n_ev + n_nn - 2)
        )
        d = (mean_ev - mean_nn) / pooled_std if pooled_std > 0 else 0

        # Welch t-test
        t, p = stats.ttest_ind(ev, nn, equal_var=False)

        hasil.append({
            "feature": col,
            "mean_event": mean_ev,
            "mean_non": mean_nn,
            "diff": mean_ev - mean_nn,
            "effect_size_d": d,
            "t_stat": t,
            "p_value": p,
            "n_event": n_ev,
            "n_non": n_nn,
        })

    df_res = pd.DataFrame(hasil)
    df_res["abs_d"] = df_res["effect_size_d"].abs()
    df_res = df_res.sort_values("abs_d", ascending=False)
    return df_res.reset_index(drop=True)


def main():
    from .data_loader import load_panel_from_gabungan
    from .data_cleaner import clean_panel
    from .feature_engineering import build_features
    from .event_detector import detect_events

    print("Memuat data...")
    panel = load_panel_from_gabungan()
    panel = clean_panel(panel)
    panel = build_features(panel)
    panel = detect_events(panel)

    # Daftar fitur (buang kolom label & non-fitur)
    non_features = {
        "date", "kode", "open", "high", "low", "close", "volume",
        "value", "frequency", "prev_close",
        "foreign_buy", "foreign_sell", "foreign_net",
        "listed_shares", "tradeable_shares",
        "ret_1d_rank_pct", "avg_trade_size",
        "target_A", "target_B", "target_C", "target_D", "target_E",
    }
    non_features |= {c for c in panel.columns if c.startswith("fwd_")}
    non_features |= {c for c in panel.columns if c.startswith("mfe_")}
    non_features |= {c for c in panel.columns if c.startswith("mae_")}
    non_features |= {c for c in panel.columns if c.startswith("is_")}

    feature_cols = [c for c in panel.columns if c not in non_features]

    print(f"\nMenguji {len(feature_cols)} fitur untuk event 'is_up_10pct'...")
    hasil = compare_features(panel, "is_up_10pct", feature_cols)

    print("\nTop 20 fitur yang paling membedakan (Cohen's d):")
    print(hasil.head(20).to_string(index=False))

    hasil.to_csv("quant/output/event_study_up10pct.csv", index=False)
    print("\n✅ Hasil lengkap disimpan di quant/output/event_study_up10pct.csv")


if __name__ == "__main__":
    main()