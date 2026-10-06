"""
AUTO SCREENER V3 - gabungan v1 (A,B) + v2 (T,V) + RATING.

Rules: A + B + T + V (4 rules)
Exit : TP +3% / SL -2% (realistis)
Rating: 0-100 per kandidat
Filter: strength >= 2, rating >= 45 → auto watchlist

Output:
  - quant/output/kandidat_v3_YYYY-MM-DD.csv
  - quant/output/kandidat_rated_YYYY-MM-DD.csv
  - quant/output/Rekap_V3_YYYY-MM.xlsx
  - quant/output/historical_stats.json
  - data_watchlist.json (auto-update)
"""

import os
import subprocess
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np

from .data_loader import load_panel_from_gabungan
from .data_cleaner import clean_panel
from .feature_engineering import build_features
from .event_detector import detect_events


# ============================================================
# KONFIGURASI
# ============================================================
COST_RT = 0.008
TP_THRESHOLD = 0.03
SL_PCT = -0.02

WATCHLIST_FILE = "data_watchlist.json"
OUTPUT_DIR = "quant/output"

MIN_STRENGTH = 2
MIN_RATING = 45


# ============================================================
# UTIL
# ============================================================
def get_bulan_berjalan():
    today = pd.Timestamp(datetime.now().date())
    start = today.replace(day=1)
    if today.month == 12:
        end = today.replace(year=today.year + 1, month=1, day=1) - pd.Timedelta(days=1)
    else:
        end = today.replace(month=today.month + 1, day=1) - pd.Timedelta(days=1)
    return start, end, today.strftime("%B %Y"), today.strftime("%Y-%m")


def jalankan_update():
    print(f"\n{'='*100}")
    print("STEP 1: Update data dari IDX")
    print(f"{'='*100}\n")

    today = datetime.now()
    if today.weekday() >= 5:
        print("[SKIP] Weekend, tidak ada data baru.")
        return True

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    update_script = os.path.join(root, "update.py")
    gabung_script = os.path.join(root, "gabung.py")

    if not os.path.exists(update_script):
        print(f"[!] File {update_script} tidak ditemukan.")
        return False

    print("Menjalankan update.py...")
    try:
        result = subprocess.run([sys.executable, update_script],
                                 cwd=root, timeout=1800)
        if result.returncode != 0:
            print(f"[!] Error update (code {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print("[!] Update timeout. Lanjut.")
        return False
    except Exception as e:
        print(f"[!] Gagal update: {e}")
        return False

    if os.path.exists(gabung_script):
        print("\nMenjalankan gabung.py...")
        try:
            subprocess.run([sys.executable, gabung_script], cwd=root, timeout=300)
        except Exception:
            pass

    print("\n[OK] Update selesai.")
    return True


def next_trading_day(ts):
    nxt = pd.Timestamp(ts) + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += pd.Timedelta(days=1)
    return nxt


# ============================================================
# RULES — 4 rules gabungan v1 + v2
# ============================================================
def rule_a(d):
    """Momentum kuat + di atas MA20 (v1 & v2)."""
    return ((d["ret_1d"] > 0.15) &
            (d["ret_5d"] > 0.15) &
            (d["px_vs_sma20"] > 0.20))


def rule_b(d):
    """Momentum + range lebar + volume spike (v1)."""
    return ((d["ret_1d"] > 0.15) &
            (d["ret_5d"] > 0.20) &
            (d["range"] > 0.10) &
            (d["vol_z_20"] > 2))


def rule_t(d):
    """Momentum + turnover tinggi (v2)."""
    return ((d["ret_1d"] > 0.15) &
            (d["px_vs_sma20"] > 0.20) &
            (d["rk_turnover"] > 0.90))


def rule_v(d):
    """Volatilitas tinggi + turnover (v2)."""
    return ((d["ret_5d"] > 0.15) &
            (d["rk_turnover"] > 0.90) &
            (d["vol_regime_high"] == 1))


def rule_union(d):
    """Union semua rule."""
    return rule_a(d) | rule_b(d) | rule_t(d) | rule_v(d)


# ============================================================
# FEATURES
# ============================================================
def add_features(df):
    df = df.copy()
    df = df.sort_values(["kode", "date"]).reset_index(drop=True)

    if "listed_shares" in df.columns:
        df["turnover"] = df["volume"] / df["listed_shares"].replace(0, np.nan)
        df["rk_turnover"] = df.groupby("date")["turnover"].rank(pct=True)
    else:
        df["rk_turnover"] = 0

    if "atr_pct_rank_60" in df.columns:
        df["vol_regime_high"] = (df["atr_pct_rank_60"] > 0.8).astype(int)
    else:
        df["vol_regime_high"] = 0

    for h in [1, 2, 3]:
        df[f"open_T{h}"] = df.groupby("kode")["open"].shift(-h)
        df[f"high_T{h}"] = df.groupby("kode")["high"].shift(-h)
        df[f"low_T{h}"] = df.groupby("kode")["low"].shift(-h)
        df[f"close_T{h}"] = df.groupby("kode")["close"].shift(-h)

    return df


# ============================================================
# SIMULATE EXIT
# ============================================================
def simulate(row):
    close_t = row["close"]
    entry = row["open_T1"]

    if pd.isna(entry) or entry <= 0:
        return None, "NO_DATA"
    if entry < close_t:
        return None, "SKIP_GAP_DOWN"

    high = row["high_T1"]
    low = row["low_T1"]
    close_t1 = row["close_T1"]

    if pd.isna(high) or high <= 0:
        return None, "NO_DATA"

    if high >= entry * (1 + TP_THRESHOLD):
        return TP_THRESHOLD - COST_RT, "TP"
    if not pd.isna(low) and low <= entry * (1 + SL_PCT):
        return SL_PCT - COST_RT, "SL"
    if high > entry:
        return high / entry - 1 - COST_RT, "PROFIT"
    if pd.isna(close_t1) or close_t1 <= 0:
        return None, "NO_DATA"
    return close_t1 / entry - 1 - COST_RT, "CLOSE"


# ============================================================
# HISTORICAL STATS
# ============================================================
def hitung_historical_stats(panel):
    print("Menghitung historical stats (4 rules)...")

    panel = panel.copy()
    m_a = rule_a(panel).fillna(False).values
    m_b = rule_b(panel).fillna(False).values
    m_t = rule_t(panel).fillna(False).values
    m_v = rule_v(panel).fillna(False).values
    m_union = m_a | m_b | m_t | m_v

    if m_union.sum() == 0:
        return {}

    signals = panel[m_union].copy()
    signals["sig_a"] = m_a[m_union]
    signals["sig_b"] = m_b[m_union]
    signals["sig_t"] = m_t[m_union]
    signals["sig_v"] = m_v[m_union]

    signals["match"] = ""
    signals.loc[signals["sig_a"], "match"] += "A"
    signals.loc[signals["sig_b"], "match"] += "B"
    signals.loc[signals["sig_t"], "match"] += "T"
    signals.loc[signals["sig_v"], "match"] += "V"
    signals["strength"] = signals["match"].str.len()

    rets, statuses = [], []
    for _, row in signals.iterrows():
        ret, status = simulate(row)
        rets.append(ret if ret is not None else np.nan)
        statuses.append(status)
    signals["ret"] = rets
    signals["status"] = statuses

    valid = signals[signals["status"].isin(["TP", "PROFIT", "SL", "CLOSE"])].copy()

    per_match = {}
    for match, group in valid.groupby("match"):
        if len(group) < 10:
            continue
        per_match[match] = {
            "n": len(group),
            "win_rate": float((group["ret"] > 0).mean()),
            "avg_pl": float(group["ret"].mean()),
        }

    per_strength = {}
    for s, group in valid.groupby("strength"):
        if len(group) < 10:
            continue
        per_strength[int(s)] = {
            "n": len(group),
            "win_rate": float((group["ret"] > 0).mean()),
            "avg_pl": float(group["ret"].mean()),
        }

    # Stats per rule tunggal
    per_rule = {}
    for rname, col in [("A", "sig_a"), ("B", "sig_b"),
                       ("T", "sig_t"), ("V", "sig_v")]:
        sub = valid[valid[col]]
        if len(sub) < 10:
            continue
        per_rule[rname] = {
            "n": len(sub),
            "win_rate": float((sub["ret"] > 0).mean()),
            "avg_pl": float(sub["ret"].mean()),
        }

    print(f"  Total signals: {len(valid)}")
    print(f"  Match unik   : {len(per_match)}")
    print(f"  Strength     : {len(per_strength)} levels")
    print(f"  Rule tunggal : {len(per_rule)}")

    return {
        "per_match": per_match,
        "per_strength": per_strength,
        "per_rule": per_rule,
    }


# ============================================================
# RATING
# ============================================================
def rate_kandidat(kandidat_df, hist_stats, panel):
    print("\nMenghitung rating...")

    pct = {}
    for feat in ["ret_1d", "ret_5d", "px_vs_sma20", "vol_z_20",
                 "rk_turnover", "range"]:
        if feat in panel.columns:
            vals = panel[feat].dropna()
            pct[feat] = vals.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()

    per_match = hist_stats.get("per_match", {})
    per_strength = hist_stats.get("per_strength", {})

    rows = []
    for _, r in kandidat_df.iterrows():
        kode = str(r["kode"]).upper()
        match = str(r.get("match", "-"))
        strength = len(match) if match != "-" else 0

        # Strength score (max 4 rules)
        strength_score = min(strength / 4 * 100, 100)

        hist = per_match.get(match, None)
        if hist and hist["n"] >= 10:
            wr_score = min(hist["win_rate"] * 100, 100)
            hist_n = hist["n"]
            hist_avg = hist["avg_pl"]
        else:
            ss = per_strength.get(strength, None)
            if ss and ss["n"] >= 10:
                wr_score = min(ss["win_rate"] * 100, 100)
                hist_n = ss["n"]
                hist_avg = ss["avg_pl"]
            else:
                wr_score = 50
                hist_n = 0
                hist_avg = 0

        metric_scores = []
        for feat in ["ret_1d", "ret_5d", "px_vs_sma20", "vol_z_20", "range"]:
            if feat not in r.index or feat not in panel.columns:
                continue
            try:
                val = float(r[feat])
                if pd.isna(val):
                    continue
                qs = pct[feat]
                if val <= qs[0.1]: s = 10
                elif val <= qs[0.25]: s = 30
                elif val <= qs[0.5]: s = 50
                elif val <= qs[0.75]: s = 70
                elif val <= qs[0.9]: s = 85
                else: s = 100
                metric_scores.append(s)
            except Exception:
                continue

        metric_score = np.mean(metric_scores) if metric_scores else 50

        rating = (
            strength_score * 0.30 +
            wr_score * 0.50 +
            metric_score * 0.20
        )

        if rating >= 80:
            cls, color = "🔥 ELITE", "#ff5252"
        elif rating >= 65:
            cls, color = "⭐ KUAT", "#00d47a"
        elif rating >= 50:
            cls, color = "🟢 SEDANG", "#3b82f6"
        elif rating >= 35:
            cls, color = "🟡 LEMAH", "#ffa726"
        else:
            cls, color = "🔴 HINDARI", "#8091b8"

        rows.append({
            "kode": kode,
            "rating": round(rating, 1),
            "class": cls,
            "color": color,
            "strength": strength,
            "match": match,
            "hist_wr": round(wr_score, 1),
            "hist_n": hist_n,
            "hist_avg_pl": round(hist_avg * 100, 2),
            "metric_score": round(metric_score, 1),
            "close": r.get("close", 0),
            "value": r.get("value", 0),
            "ret_1d": r.get("ret_1d", 0),
            "ret_5d": r.get("ret_5d", 0),
            "range": r.get("range", 0),
            "vol_z_20": r.get("vol_z_20", 0),
            "px_vs_sma20": r.get("px_vs_sma20", 0),
            "rk_turnover": r.get("rk_turnover", 0),
            "vol_regime_high": r.get("vol_regime_high", 0),
        })

    df = pd.DataFrame(rows).sort_values(
        ["strength", "rating"], ascending=[False, False]
    ).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


# ============================================================
# EXPORT WATCHLIST
# ============================================================
def export_to_watchlist(rated_df):
    print("\nExport ke watchlist...")

    wl = {}
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
                wl = json.load(f)
        except Exception:
            wl = {}

    df_filtered = rated_df[
        (rated_df["strength"] >= MIN_STRENGTH) &
        (rated_df["rating"] >= MIN_RATING)
    ].copy()

    added = 0
    for _, r in df_filtered.iterrows():
        kode = r["kode"]
        if kode in wl:
            continue
        wl[kode] = {
            "tanggal_tambah": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "harga_entry": float(r["close"]),
            "match": str(r["match"]),
            "strength": int(r["strength"]),
            "rating": float(r["rating"]),
            "class": str(r["class"]),
            "hist_wr": float(r["hist_wr"]),
            "hist_n": int(r["hist_n"]),
            "source": "screener_v3",
        }
        added += 1

    with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(wl, f, indent=2, ensure_ascii=False)

    print(f"  Filter: strength>={MIN_STRENGTH}, rating>={MIN_RATING}")
    print(f"  Ditambahkan: {added}")
    print(f"  Total watchlist: {len(wl)}")
    return added


# ============================================================
# EXCEL
# ============================================================
def excel_style(ws, title_text, subtitle_text, header_row=4):
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ws.max_column)
    tc = ws.cell(row=1, column=1)
    tc.value = title_text
    tc.font = Font(bold=True, size=16, color="1F4E78")
    tc.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 25

    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ws.max_column)
    sc = ws.cell(row=2, column=1)
    sc.value = subtitle_text
    sc.font = Font(italic=True, size=10, color="666666")
    sc.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[2].height = 15

    hf = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    hfont = Font(color="FFFFFF", bold=True, size=11)
    border = Border(left=Side(style="thin", color="D0D0D0"),
                    right=Side(style="thin", color="D0D0D0"),
                    top=Side(style="thin", color="D0D0D0"),
                    bottom=Side(style="thin", color="D0D0D0"))

    for cell in ws[header_row]:
        cell.fill = hf
        cell.font = hfont
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border
    ws.row_dimensions[header_row].height = 30
    ws.freeze_panes = ws.cell(row=header_row + 1, column=1).coordinate


def autofit_columns(ws, skip_rows=3, max_width=22):
    from openpyxl.utils import get_column_letter
    for col_idx, col in enumerate(ws.iter_cols(min_row=skip_rows + 1,
                                                 max_row=ws.max_row), 1):
        max_len = 0
        cl = get_column_letter(col_idx)
        for cell in col:
            if cell.value:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[cl].width = min(max_len + 3, max_width)


def simpan_excel(kandidat_rated, signals_bulan, pred_date, path_dir,
                 bulan_label, bulan_file, hist_stats):
    try:
        from openpyxl import load_workbook
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter
        from openpyxl.formatting.rule import CellIsRule
    except ImportError:
        print("  [!] openpyxl tidak terinstall.")
        return

    base_name = f"Rekap_V3_{bulan_file}"
    filepath = os.path.join(path_dir, f"{base_name}.xlsx")

    if os.path.exists(filepath):
        try:
            with open(filepath, 'a'):
                pass
        except PermissionError:
            ts = datetime.now().strftime("%H%M%S")
            filepath = os.path.join(path_dir, f"{base_name}_{ts}.xlsx")

    # SHEET 1: KANDIDAT + RATING
    if not kandidat_rated.empty:
        cols = ["rank", "kode", "rating", "class", "strength", "match",
                "close", "value", "ret_1d", "ret_5d", "range",
                "vol_z_20", "px_vs_sma20", "rk_turnover",
                "hist_wr", "hist_n", "hist_avg_pl"]
        cols = [c for c in cols if c in kandidat_rated.columns]
        df_kand = kandidat_rated[cols].copy()

        for c in ["ret_1d", "ret_5d", "range", "px_vs_sma20"]:
            if c in df_kand.columns:
                df_kand[c] = (df_kand[c] * 100).round(2)
        if "vol_z_20" in df_kand.columns:
            df_kand["vol_z_20"] = df_kand["vol_z_20"].round(2)
        if "rk_turnover" in df_kand.columns:
            df_kand["rk_turnover"] = df_kand["rk_turnover"].round(3)
        if "value" in df_kand.columns:
            df_kand["value"] = (df_kand["value"] / 1e9).round(2)
        if "rating" in df_kand.columns:
            df_kand["rating"] = df_kand["rating"].round(1)

        df_kand = df_kand.rename(columns={
            "rank": "Rank", "kode": "Kode", "rating": "Rating",
            "class": "Class", "strength": "Str", "match": "Match",
            "close": "Close", "value": "Value (M)",
            "ret_1d": "Ret1d%", "ret_5d": "Ret5d%", "range": "Range%",
            "vol_z_20": "VolZ", "px_vs_sma20": "Px>sma20%",
            "rk_turnover": "Turnover",
            "hist_wr": "HistWR%", "hist_n": "HistN",
            "hist_avg_pl": "HistAvgPL%",
        })
    else:
        df_kand = pd.DataFrame({"Info": ["Tidak ada kandidat"]})

    # SHEET 2: RULE STATS
    stats_rows = []
    per_rule = hist_stats.get("per_rule", {})
    for rule_name in ["A", "B", "T", "V"]:
        if rule_name in per_rule:
            d = per_rule[rule_name]
            stats_rows.append([
                f"Rule {rule_name}", d["n"],
                f"{d['win_rate']*100:.1f}%",
                f"{d['avg_pl']*100:+.2f}%",
            ])

    per_strength = hist_stats.get("per_strength", {})
    for s in sorted(per_strength.keys()):
        d = per_strength[s]
        stats_rows.append([
            f"Strength {s}", d["n"],
            f"{d['win_rate']*100:.1f}%",
            f"{d['avg_pl']*100:+.2f}%",
        ])

    if stats_rows:
        df_stats = pd.DataFrame(stats_rows, columns=["Metrik", "N", "Win Rate", "Avg P/L"])
    else:
        df_stats = pd.DataFrame({"Info": ["Belum ada stats"]})

    # SHEET 3: REVIEW HARIAN
    if not signals_bulan.empty:
        review_rows = []
        for _, row in signals_bulan.iterrows():
            tgl_sinyal = pd.Timestamp(row["date"])
            tgl_eksekusi = next_trading_day(tgl_sinyal)
            ret, status = simulate(row)

            match = ""
            if row.get("sig_a", False): match += "A"
            if row.get("sig_b", False): match += "B"
            if row.get("sig_t", False): match += "T"
            if row.get("sig_v", False): match += "V"

            entry = row["open_T1"]
            high = row["high_T1"]

            review_rows.append({
                "Tanggal": tgl_eksekusi.date(),
                "Hari": ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat',
                         'Sabtu', 'Minggu'][tgl_eksekusi.weekday()],
                "Sinyal Dari": tgl_sinyal.date(),
                "Kode": row["kode"],
                "Close T": round(row["close"], 0),
                "Entry": round(entry, 0) if not pd.isna(entry) else "",
                "High T+1": round(high, 0) if not pd.isna(high) else "",
                "Match": match,
                "Strength": len(match),
                "Status": status,
                "P/L %": round(ret * 100, 2) if ret is not None else "",
            })
        df_review = pd.DataFrame(review_rows)
        df_review = df_review.sort_values(["Tanggal", "Kode"]).reset_index(drop=True)
    else:
        df_review = pd.DataFrame({"Info": ["Belum ada eksekusi"]})

    # SHEET 4: RINGKASAN
    if not signals_bulan.empty:
        total_pl = n_exec = n_win = 0
        n_tp = n_sl = n_profit = n_close = 0
        n_skip = n_pending = 0

        for _, row in signals_bulan.iterrows():
            ret, status = simulate(row)
            if status in ("TP", "PROFIT", "SL", "CLOSE"):
                n_exec += 1
                total_pl += ret
                if ret > 0: n_win += 1
                if status == "TP": n_tp += 1
                elif status == "PROFIT": n_profit += 1
                elif status == "SL": n_sl += 1
                else: n_close += 1
            elif status == "SKIP_GAP_DOWN":
                n_skip += 1
            else:
                n_pending += 1

        win_rate = (n_win / n_exec * 100) if n_exec > 0 else 0
        avg_pl = (total_pl / n_exec * 100) if n_exec > 0 else 0

        n_a = int(signals_bulan["sig_a"].sum())
        n_b = int(signals_bulan["sig_b"].sum())
        n_t = int(signals_bulan["sig_t"].sum())
        n_v = int(signals_bulan["sig_v"].sum())

        signals_bulan["_match"] = ""
        signals_bulan.loc[signals_bulan["sig_a"], "_match"] += "A"
        signals_bulan.loc[signals_bulan["sig_b"], "_match"] += "B"
        signals_bulan.loc[signals_bulan["sig_t"], "_match"] += "T"
        signals_bulan.loc[signals_bulan["sig_v"], "_match"] += "V"
        signals_bulan["_strength"] = signals_bulan["_match"].str.len()

        s1 = int((signals_bulan["_strength"] == 1).sum())
        s2 = int((signals_bulan["_strength"] == 2).sum())
        s3 = int((signals_bulan["_strength"] == 3).sum())
        s4 = int((signals_bulan["_strength"] == 4).sum())

        ringkasan_rows = [
            ["PERIODE EKSEKUSI", f"Bulan {bulan_label}"],
            ["Rules", "A + B + T + V (V3 Combined)"],
            ["Exit", f"TP +{TP_THRESHOLD*100:.0f}% | SL {SL_PCT*100:.0f}%"],
            ["Rating Formula", "Strength*0.3 + HistWR*0.5 + Metric*0.2"],
            ["", ""],
            ["SINYAL", ""],
            ["Total", len(signals_bulan)],
            ["  A", n_a], ["  B", n_b], ["  T", n_t], ["  V", n_v],
            ["", ""],
            ["STRENGTH", ""],
            ["  1 rule", s1], ["  2 rules", s2],
            ["  3 rules", s3], ["  4 rules", s4],
            ["", ""],
            ["EKSEKUSI", ""],
            ["Tereksekusi", n_exec],
            ["  TP", n_tp], ["  PROFIT", n_profit],
            ["  SL", n_sl], ["  CLOSE", n_close],
            ["  Skip", n_skip], ["  Pending", n_pending],
            ["", ""],
            ["PERFORMA", ""],
            ["Win Rate", f"{win_rate:.1f}%"],
            ["Total P/L", f"{total_pl*100:+.2f}%"],
            ["Avg/Trade", f"{avg_pl:+.2f}%"],
            ["", ""],
            ["VERDICT", "[OK]" if win_rate >= 60 and avg_pl > 1.5
             else ("[!] MARGINAL" if win_rate >= 50 else "[X] EVALUASI")],
        ]
    else:
        ringkasan_rows = [["Info", "Belum ada sinyal"]]

    df_ringkasan = pd.DataFrame(ringkasan_rows, columns=["Metrik", "Nilai"])

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df_ringkasan.to_excel(writer, sheet_name="Ringkasan", index=False, startrow=3)
        df_kand.to_excel(writer, sheet_name="Kandidat V3", index=False, startrow=3)
        df_stats.to_excel(writer, sheet_name="Rule Stats", index=False, startrow=3)
        df_review.to_excel(writer, sheet_name="Review Harian", index=False, startrow=3)

    wb = load_workbook(filepath)

    GREEN = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    GREEN_F = Font(color="006100", bold=True)
    RED = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    RED_F = Font(color="9C0006", bold=True)
    SEC_F = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    SEC_FONT = Font(bold=True, size=11, color="1F4E78")

    ws = wb["Ringkasan"]
    excel_style(ws, f"RINGKASAN V3 - {bulan_label.upper()}",
                f"Dibuat: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    for row in ws.iter_rows(min_row=5, max_row=ws.max_row, min_col=1, max_col=2):
        c = row[0]
        if c.value and isinstance(c.value, str) and c.value.isupper() and c.value.strip():
            c.fill = SEC_F
            c.font = SEC_FONT
            row[1].fill = SEC_F
    autofit_columns(ws)

    ws = wb["Kandidat V3"]
    excel_style(ws, f"KANDIDAT V3 {pred_date.strftime('%A, %d %B %Y').upper()}",
                f"Sorted by Strength + Rating | Filter export: str>={MIN_STRENGTH}, rating>={MIN_RATING}")
    autofit_columns(ws)
    if not kandidat_rated.empty:
        for row in ws.iter_rows(min_row=5, max_row=ws.max_row,
                                 min_col=1, max_col=ws.max_column):
            for cell in row:
                cell.alignment = Alignment(horizontal="center")

    ws = wb["Rule Stats"]
    excel_style(ws, f"RULE STATS - HISTORICAL", "Performa setiap rule & strength")
    autofit_columns(ws)

    ws = wb["Review Harian"]
    excel_style(ws, f"REVIEW HARIAN - {bulan_label.upper()}",
                "Tanggal = hari EKSEKUSI (T+1)")
    autofit_columns(ws)
    if not df_review.empty and "P/L %" in df_review.columns:
        for i, cell in enumerate(ws[4], 1):
            if cell.value == "P/L %":
                cl = get_column_letter(i)
                rng = f"{cl}5:{cl}{ws.max_row}"
                ws.conditional_formatting.add(rng,
                    CellIsRule(operator="greaterThan", formula=["0"],
                               fill=GREEN, font=GREEN_F))
                ws.conditional_formatting.add(rng,
                    CellIsRule(operator="lessThan", formula=["0"],
                               fill=RED, font=RED_F))

    wb.active = wb.sheetnames.index("Kandidat V3")
    wb.save(filepath)
    print(f"  -> Excel: {filepath}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 110)
    print(f"AUTO SCREENER V3 (A+B+T+V + RATING) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TP +{TP_THRESHOLD*100:.0f}% | SL {SL_PCT*100:.0f}%")
    print(f"Export filter: strength>={MIN_STRENGTH}, rating>={MIN_RATING}")
    print("=" * 110)

    bulan_start, bulan_end, bulan_label, bulan_file = get_bulan_berjalan()
    print(f"\nBulan: {bulan_label}")

    jalankan_update()

    print(f"\n{'='*110}")
    print("STEP 2: Screening V3 (4 rules)")
    print(f"{'='*110}\n")

    print("Memuat data...")
    try:
        panel = load_panel_from_gabungan()
    except FileNotFoundError:
        print("[!] File data tidak ditemukan.")
        return

    panel = clean_panel(panel)
    panel = build_features(panel)
    panel = detect_events(panel)
    panel = add_features(panel)
    panel = panel[(panel["open"] > 0) & (panel["close"] > 0)].copy()

    last_date = pd.Timestamp(panel["date"].max())
    pred_date = next_trading_day(last_date)

    print(f"\nData terakhir    : {last_date.date()}")
    print(f"Tanggal eksekusi : {pred_date.date()}")

    # Kandidat (4 rules)
    today_data = panel[panel["date"] == last_date].copy()
    m_a = rule_a(today_data).fillna(False).values
    m_b = rule_b(today_data).fillna(False).values
    m_t = rule_t(today_data).fillna(False).values
    m_v = rule_v(today_data).fillna(False).values
    m_union = m_a | m_b | m_t | m_v

    kandidat = today_data[m_union].copy()

    print(f"\n{'#'*110}")
    print(f"#  KANDIDAT V3 EKSEKUSI {pred_date.date()}")
    print(f"#  4 rules: A + B + T + V")
    print(f"{'#'*110}")

    if kandidat.empty:
        print("\n  Tidak ada kandidat.")
        return

    kandidat["match"] = ""
    kandidat.loc[m_a[m_union], "match"] += "A"
    kandidat.loc[m_b[m_union], "match"] += "B"
    kandidat.loc[m_t[m_union], "match"] += "T"
    kandidat.loc[m_v[m_union], "match"] += "V"
    kandidat["strength"] = kandidat["match"].str.len()

    print(f"\n  {len(kandidat)} KANDIDAT (sebelum rating)")

    # Historical stats
    print(f"\n{'='*110}")
    print("STEP 3: Historical Stats & Rating")
    print(f"{'='*110}")
    hist_stats = hitung_historical_stats(panel)

    # Rating
    kandidat_rated = rate_kandidat(kandidat, hist_stats, panel)

    # Tampilkan
    print(f"\n{'='*110}")
    print(f"RATING KANDIDAT V3")
    print(f"{'='*110}\n")
    print(f"{'#':>3} {'Kode':<7} {'Rating':>7} {'Class':<14} "
          f"{'Str':>4} {'Match':>7} {'HistWR':>8} {'N':>5} "
          f"{'Close':>8} {'Value':>8}")
    print("-" * 110)
    for _, r in kandidat_rated.iterrows():
        print(f"{r['rank']:>3} {r['kode']:<7} {r['rating']:>6.1f} "
              f"{r['class']:<14} {r['strength']:>4} {r['match']:>7} "
              f"{r['hist_wr']:>7.1f}% {r['hist_n']:>5} "
              f"{r['close']:>8.0f} {r['value']/1e9:>7.2f}")

    # Review bulan
    print(f"\n{'#'*110}")
    print(f"#  REVIEW BULAN {bulan_label.upper()}")
    print(f"{'#'*110}")

    buffer_start = bulan_start - pd.Timedelta(days=10)
    sub_calon = panel[(panel["date"] >= buffer_start) &
                       (panel["date"] <= bulan_end)].copy()

    m_a_s = rule_a(sub_calon).fillna(False).values
    m_b_s = rule_b(sub_calon).fillna(False).values
    m_t_s = rule_t(sub_calon).fillna(False).values
    m_v_s = rule_v(sub_calon).fillna(False).values
    m_union_s = m_a_s | m_b_s | m_t_s | m_v_s

    signals_calon = sub_calon[m_union_s].copy()
    signals_calon["sig_a"] = m_a_s[m_union_s]
    signals_calon["sig_b"] = m_b_s[m_union_s]
    signals_calon["sig_t"] = m_t_s[m_union_s]
    signals_calon["sig_v"] = m_v_s[m_union_s]

    if not signals_calon.empty:
        signals_calon["tanggal_eksekusi"] = signals_calon["date"].apply(next_trading_day)
        signals_bulan = signals_calon[
            (signals_calon["tanggal_eksekusi"] >= bulan_start) &
            (signals_calon["tanggal_eksekusi"] <= bulan_end)
        ].copy()
    else:
        signals_bulan = pd.DataFrame()

    if not signals_bulan.empty:
        total_pl = n_exec = n_win = n_tp = n_sl = 0
        for _, row in signals_bulan.iterrows():
            ret, status = simulate(row)
            if status in ("TP", "PROFIT", "SL", "CLOSE"):
                n_exec += 1
                total_pl += ret
                if ret > 0: n_win += 1
                if status == "TP": n_tp += 1
                elif status == "SL": n_sl += 1

        if n_exec > 0:
            print(f"\n  Total sinyal   : {len(signals_bulan)}")
            print(f"  Tereksekusi    : {n_exec}")
            print(f"  Win rate       : {n_win}/{n_exec} ({n_win/n_exec*100:.1f}%)")
            print(f"  Total P/L      : {total_pl*100:+.2f}%")
            print(f"  Avg per trade  : {total_pl/n_exec*100:+.2f}%")
    else:
        print(f"\n  Belum ada sinyal di bulan ini.")

    # Simpan
    print(f"\n{'#'*110}")
    print(f"#  MENYIMPAN")
    print(f"{'#'*110}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CSV V3 raw
    csv_v3 = f"{OUTPUT_DIR}/kandidat_v3_{pred_date.date()}.csv"
    cols_v3 = ["kode", "close", "value", "ret_1d", "ret_5d",
               "range", "vol_z_20", "px_vs_sma20",
               "rk_turnover", "vol_regime_high",
               "match", "strength"]
    cols_v3 = [c for c in cols_v3 if c in kandidat.columns]
    kandidat[cols_v3].to_csv(csv_v3, index=False)
    print(f"  -> CSV V3: {csv_v3}")

    # CSV rated
    csv_rated = f"{OUTPUT_DIR}/kandidat_rated_{pred_date.date()}.csv"
    kandidat_rated.to_csv(csv_rated, index=False)
    print(f"  -> CSV Rated: {csv_rated}")

    # Stats JSON
    stats_path = f"{OUTPUT_DIR}/historical_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(hist_stats, f, indent=2, ensure_ascii=False, default=str)
    print(f"  -> Stats: {stats_path}")

    # Excel
    simpan_excel(kandidat_rated, signals_bulan, pred_date.date(),
                 OUTPUT_DIR, bulan_label, bulan_file, hist_stats)

    # Export watchlist
    export_to_watchlist(kandidat_rated)

    print(f"\n{'='*110}")
    print(f"SELESAI - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*110}")


if __name__ == "__main__":
    main()