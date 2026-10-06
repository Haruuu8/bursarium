"""
Export DATA LENGKAP September 2026 ke Excel dengan catatan.
"""

import os
import numpy as np
import pandas as pd

from .data_loader import load_panel_from_gabungan
from .data_cleaner import clean_panel
from .feature_engineering import build_features
from .event_detector import detect_events


BULAN = "2026-09"
BULAN_LABEL = "September 2026"
COST_RT = 0.008
OUTPUT = "quant/output/Data_September_2026.xlsx"


def rule_a(d):
    return (
        (d["ret_1d"] > 0.15) &
        (d["ret_5d"] > 0.15) &
        (d["px_vs_sma20"] > 0.20)
    )


def rule_b(d):
    return (
        (d["ret_1d"] > 0.15) &
        (d["ret_5d"] > 0.20) &
        (d["range"] > 0.10) &
        (d["vol_z_20"] > 2)
    )


def next_trading_day(ts):
    nxt = pd.Timestamp(ts) + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += pd.Timedelta(days=1)
    return nxt


def add_forward(df):
    df = df.copy()
    df = df.sort_values(["kode", "date"]).reset_index(drop=True)
    for h in [1, 2, 3]:
        df[f"open_T{h}"] = df.groupby("kode")["open"].shift(-h)
        df[f"high_T{h}"] = df.groupby("kode")["high"].shift(-h)
        df[f"low_T{h}"] = df.groupby("kode")["low"].shift(-h)
        df[f"close_T{h}"] = df.groupby("kode")["close"].shift(-h)
    return df


def main():
    print("=" * 100)
    print(f"EXPORT DATA LENGKAP {BULAN_LABEL}")
    print("=" * 100)

    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter
        from openpyxl.formatting.rule import CellIsRule
    except ImportError:
        print("\n[!] Install dulu: python -m pip install openpyxl")
        return

    print("\nMemuat data...")
    panel = load_panel_from_gabungan()
    panel = clean_panel(panel)
    panel = build_features(panel)
    panel = detect_events(panel)
    panel = add_forward(panel)
    panel = panel[(panel["open"] > 0) & (panel["close"] > 0)].copy()

    sub = panel[panel["date"].dt.to_period("M") == BULAN].copy()
    if sub.empty:
        print("Tidak ada data September.")
        return

    sub = sub.sort_values(["date", "kode"]).reset_index(drop=True)
    n_days = sub["date"].nunique()
    n_saham = sub["kode"].nunique()
    print(f"\nData: {len(sub):,} baris, {n_days} hari, {n_saham} saham")

    sub["sig_a"] = rule_a(sub).fillna(False).values
    sub["sig_b"] = rule_b(sub).fillna(False).values
    sub["is_signal"] = sub["sig_a"] | sub["sig_b"]

    print("\nMembuat catatan sinyal...")

    def buat_row(row):
        tgl = pd.Timestamp(row["date"])
        kode = row["kode"]

        if not row["is_signal"]:
            return {
                "Sinyal": "", "Hari": "",
                "Tanggal Sinyal": "", "Prediksi Untuk": "",
                "Entry": "", "TP": "", "SL": "",
                "High T+1": "", "Low T+1": "", "Close T+1": "",
                "Exit": "", "P/L %": "", "Status": "",
                "MFE %": "", "MAE %": "", "Alasan": "",
            }

        match = ""
        if row["sig_a"]:
            match += "A"
        if row["sig_b"]:
            match += "B"

        hari = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu',
                'Minggu'][tgl.weekday()]
        pred_date = next_trading_day(tgl)

        close_t = row["close"]
        entry = row["open_T1"]
        high = row["high_T1"]
        low = row["low_T1"]
        close_t1 = row["close_T1"]

        if pd.isna(entry) or entry <= 0:
            return {
                "Sinyal": match, "Hari": hari,
                "Tanggal Sinyal": tgl.date(),
                "Prediksi Untuk": pred_date.date(),
                "Entry": "", "TP": "", "SL": "",
                "High T+1": "", "Low T+1": "", "Close T+1": "",
                "Exit": "", "P/L %": "", "Status": "NO_DATA",
                "MFE %": "", "MAE %": "",
                "Alasan": "Data T+1 tidak tersedia",
            }

        if entry < close_t:
            return {
                "Sinyal": match, "Hari": hari,
                "Tanggal Sinyal": tgl.date(),
                "Prediksi Untuk": pred_date.date(),
                "Entry": round(entry, 0), "TP": "", "SL": "",
                "High T+1": round(high, 0) if not pd.isna(high) else "",
                "Low T+1": round(low, 0) if not pd.isna(low) else "",
                "Close T+1": round(close_t1, 0) if not pd.isna(close_t1) else "",
                "Exit": "", "P/L %": "",
                "Status": "SKIP_GAP_DOWN",
                "MFE %": "", "MAE %": "",
                "Alasan": f"Gap down: open {entry:.0f} < close T {close_t:.0f}",
            }

        mfe = (high - entry) / entry * 100
        mae = (low - entry) / entry * 100 if not pd.isna(low) else 0
        ret_h = high / entry - 1 - COST_RT

        tp = entry * 1.05
        sl = entry * 0.95

        if high >= tp:
            exit_desc = f"TP @ {tp:.0f}"
        else:
            exit_desc = f"Close @ {close_t1:.0f}" if not pd.isna(close_t1) else ""

        return {
            "Sinyal": match, "Hari": hari,
            "Tanggal Sinyal": tgl.date(),
            "Prediksi Untuk": pred_date.date(),
            "Entry": round(entry, 0),
            "TP": round(tp, 0),
            "SL": round(sl, 0),
            "High T+1": round(high, 0) if not pd.isna(high) else "",
            "Low T+1": round(low, 0) if not pd.isna(low) else "",
            "Close T+1": round(close_t1, 0) if not pd.isna(close_t1) else "",
            "Exit": exit_desc,
            "P/L %": round(ret_h * 100, 2),
            "Status": "OK",
            "MFE %": round(mfe, 2),
            "MAE %": round(mae, 2),
            "Alasan": "",
        }

    catatan_rows = sub.apply(buat_row, axis=1)
    df_catatan = pd.DataFrame(catatan_rows.tolist())
    for col in df_catatan.columns:
        sub[col] = df_catatan[col].values

    print("Sheet 1: Catatan Sinyal...")
    sig_only = sub[sub["is_signal"]].copy()

    kolom_catatan = [
        "Tanggal Sinyal", "Hari", "kode", "Sinyal",
        "Prediksi Untuk", "close", "Entry", "TP", "SL",
        "High T+1", "Low T+1", "Close T+1",
        "MFE %", "MAE %", "Exit", "P/L %", "Status", "Alasan",
    ]
    df_catatan_sheet = sig_only[kolom_catatan].copy()
    df_catatan_sheet = df_catatan_sheet.sort_values(
        ["Tanggal Sinyal", "kode"]).reset_index(drop=True)
    df_catatan_sheet = df_catatan_sheet.rename(columns={
        "kode": "Kode",
        "close": "Close T",
    })

    print("Sheet 2: Semua Data...")

    kolom_all = [
        "date", "kode",
        "open", "high", "low", "close",
        "ret_1d", "ret_5d", "ret_20d",
        "gap", "range", "clv",
        "vol_ratio_5", "vol_z_20", "rsi_14",
        "Sinyal", "Hari",
        "Tanggal Sinyal", "Prediksi Untuk",
        "Entry", "TP", "SL",
        "High T+1", "Low T+1", "Close T+1",
        "Exit", "P/L %", "Status", "Alasan",
    ]
    df_all = sub[kolom_all].copy()
    df_all["date"] = pd.to_datetime(df_all["date"]).dt.date
    df_all = df_all.rename(columns={
        "date": "Tanggal",
        "kode": "Kode",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "ret_1d": "Ret1d%",
        "ret_5d": "Ret5d%",
        "ret_20d": "Ret20d%",
        "gap": "Gap%",
        "range": "Range%",
        "clv": "CLV",
        "vol_ratio_5": "VolRatio5",
        "vol_z_20": "VolZ",
        "rsi_14": "RSI",
    })
    for col in ["Ret1d%", "Ret5d%", "Ret20d%", "Gap%", "Range%"]:
        if col in df_all.columns:
            df_all[col] = (df_all[col] * 100).round(2)

    print("Sheet 3: Rekap Harian...")

    hari_rows = []
    for tgl in sorted(sub["date"].unique()):
        tgl_ts = pd.Timestamp(tgl)
        day = sub[sub["date"] == tgl]
        day_sig = day[day["is_signal"]]
        day_ok = day_sig[day_sig["Status"] == "OK"]

        n_sig = len(day_sig)
        n_ok = len(day_ok)
        if n_ok > 0:
            n_win = int((day_ok["P/L %"].astype(float) > 0).sum())
            pl_total = day_ok["P/L %"].astype(float).sum()
        else:
            n_win = 0
            pl_total = 0

        hari_rows.append({
            "Tanggal": tgl_ts.date(),
            "Hari": ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu',
                     'Minggu'][tgl_ts.weekday()],
            "Total Saham": len(day),
            "Sinyal A": int(day["sig_a"].sum()),
            "Sinyal B": int(day["sig_b"].sum()),
            "Sinyal Total": n_sig,
            "Eksekusi": n_ok,
            "Profit": n_win,
            "Skip Gap Down": int((day_sig["Status"] == "SKIP_GAP_DOWN").sum()),
            "Pending": int((day_sig["Status"] == "PENDING").sum()),
            "Win Rate %": round(n_win / n_ok * 100, 1) if n_ok > 0 else 0,
            "Total P/L %": round(pl_total, 2),
        })

    df_harian = pd.DataFrame(hari_rows)

    print("Sheet 4: Rekap per Saham...")

    saham_rows = []
    for kode in sorted(sub["kode"].unique()):
        df_k = sub[sub["kode"] == kode]
        sig_k = df_k[df_k["is_signal"]]
        if len(sig_k) == 0:
            continue

        n_sig = len(sig_k)
        ok_k = sig_k[sig_k["Status"] == "OK"]
        n_ok = len(ok_k)
        if n_ok > 0:
            n_win = int((ok_k["P/L %"].astype(float) > 0).sum())
            pl_total = ok_k["P/L %"].astype(float).sum()
        else:
            n_win = 0
            pl_total = 0

        tgl_sinyal = sorted(sig_k["date"].dt.strftime("%d/%m").tolist())

        saham_rows.append({
            "Kode": kode,
            "Total Sinyal": n_sig,
            "Tanggal Sinyal": ", ".join(tgl_sinyal),
            "Eksekusi": n_ok,
            "Profit": n_win,
            "Win Rate %": round(n_win / n_ok * 100, 1) if n_ok > 0 else 0,
            "Total P/L %": round(pl_total, 2),
            "Avg P/L %": round(pl_total / n_ok, 2) if n_ok > 0 else 0,
            "Harga Terakhir": round(df_k.iloc[-1]["close"], 0),
        })

    df_saham = pd.DataFrame(saham_rows)
    if not df_saham.empty:
        df_saham = df_saham.sort_values("Total P/L %", ascending=False)

    print("Sheet 5: Ringkasan...")

    sig_only_all = sub[sub["is_signal"]]
    ok_all = sig_only_all[sig_only_all["Status"] == "OK"]
    n_exec = len(ok_all)
    if n_exec > 0:
        n_win = int((ok_all["P/L %"].astype(float) > 0).sum())
        total_pl = ok_all["P/L %"].astype(float).sum()
    else:
        n_win = 0
        total_pl = 0

    ringkasan = [
        ["PERIODE", BULAN_LABEL],
        ["Hari Bursa", n_days],
        ["Total Saham", n_saham],
        ["", ""],
        ["SINYAL", ""],
        ["Total Sinyal Muncul", len(sig_only_all)],
        ["  Sinyal A", int(sig_only_all["sig_a"].sum())],
        ["  Sinyal B", int(sig_only_all["sig_b"].sum())],
        ["  Skip (gap down)",
         int((sig_only_all["Status"] == "SKIP_GAP_DOWN").sum())],
        ["  Pending",
         int((sig_only_all["Status"] == "PENDING").sum())],
        ["  Tereksekusi", n_exec],
        ["  Profit", n_win],
        ["  Win Rate", f"{n_win / n_exec * 100:.1f}%" if n_exec > 0 else "0%"],
        ["", ""],
        ["P/L", ""],
        ["Total P/L", f"{total_pl:+.2f}%"],
        ["Avg P/L per Trade",
         f"{total_pl / n_exec:+.2f}%" if n_exec > 0 else "0%"],
        ["", ""],
        ["FREKUENSI", ""],
        ["Sinyal per Hari", f"{len(sig_only_all) / n_days:.1f}"],
        ["Trade per Hari", f"{n_exec / n_days:.1f}"],
    ]

    df_ringkasan = pd.DataFrame(ringkasan, columns=["Metrik", "Nilai"])

    print(f"\nMenulis Excel: {OUTPUT}")
    os.makedirs("quant/output", exist_ok=True)

    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        df_ringkasan.to_excel(writer, sheet_name="Ringkasan", index=False)
        df_harian.to_excel(writer, sheet_name="Rekap Harian", index=False)
        df_saham.to_excel(writer, sheet_name="Rekap per Saham", index=False)
        df_catatan_sheet.to_excel(writer, sheet_name="Catatan Sinyal", index=False)
        df_all.to_excel(writer, sheet_name="Semua Data", index=False)

    print("Formatting...")
    from openpyxl import load_workbook
    wb = load_workbook(OUTPUT)

    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78",
                               fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True, size=11)
    section_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2",
                                fill_type="solid")
    section_font = Font(bold=True, size=11, color="1F4E78")
    signal_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC",
                               fill_type="solid")

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        if sheet_name == "Ringkasan":
            for row in ws.iter_rows(min_row=2, max_row=ws.max_row,
                                     min_col=1, max_col=2):
                cell = row[0]
                if cell.value and isinstance(cell.value, str):
                    if cell.value.isupper() and cell.value.strip():
                        cell.fill = section_fill
                        cell.font = section_font
                        row[1].fill = section_fill
        else:
            ws.freeze_panes = "A2"

        for col_idx, col in enumerate(ws.columns, 1):
            max_len = 0
            col_letter = get_column_letter(col_idx)
            for cell in col[:200]:
                if cell.value:
                    max_len = max(max_len, len(str(cell.value)))
            ws.column_dimensions[col_letter].width = min(max_len + 3, 22)

    if "Semua Data" in wb.sheetnames:
        ws = wb["Semua Data"]
        sig_col_idx = None
        for i, cell in enumerate(ws[1], 1):
            if cell.value == "Sinyal":
                sig_col_idx = i
                break

        if sig_col_idx:
            sig_letter = get_column_letter(sig_col_idx)
            for row_idx in range(2, ws.max_row + 1):
                cell = ws[f"{sig_letter}{row_idx}"]
                if cell.value and str(cell.value).strip():
                    for col_idx in range(1, ws.max_column + 1):
                        ws.cell(row=row_idx, column=col_idx).fill = signal_fill

    green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE",
                              fill_type="solid")
    green_font = Font(color="006100")
    red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE",
                            fill_type="solid")
    red_font = Font(color="9C0006")

    for sheet_name in ["Catatan Sinyal", "Rekap Harian", "Rekap per Saham",
                        "Semua Data"]:
        if sheet_name not in wb.sheetnames:
            continue
        ws = wb[sheet_name]
        for col_name in ["P/L %", "Total P/L %", "Avg P/L %"]:
            for i, cell in enumerate(ws[1], 1):
                if cell.value == col_name:
                    col_letter = get_column_letter(i)
                    rng = f"{col_letter}2:{col_letter}{ws.max_row}"
                    ws.conditional_formatting.add(
                        rng,
                        CellIsRule(operator="greaterThan", formula=["0"],
                                   fill=green_fill, font=green_font)
                    )
                    ws.conditional_formatting.add(
                        rng,
                        CellIsRule(operator="lessThan", formula=["0"],
                                   fill=red_fill, font=red_font)
                    )

    wb.save(OUTPUT)
    print(f"\n✅ Tersimpan: {OUTPUT}")

    try:
        os.startfile(os.path.abspath(OUTPUT))
    except Exception:
        pass

    print(f"\nISI FILE:")
    print(f"  Sheet 1 - Ringkasan         : total bulanan")
    print(f"  Sheet 2 - Rekap Harian      : per hari")
    print(f"  Sheet 3 - Rekap per Saham   : per kode")
    print(f"  Sheet 4 - Catatan Sinyal    : detail setiap sinyal")
    print(f"  Sheet 5 - Semua Data        : semua saham + catatan")


if __name__ == "__main__":
    main()