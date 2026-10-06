"""
AUTO SCREENER - filter otomatis per bulan.
- Oktober: hanya tampilkan data Oktober
- November: file baru khusus November
- dst.
"""

import os
import subprocess
import sys
from datetime import datetime
import pandas as pd

from .data_loader import load_panel_from_gabungan
from .data_cleaner import clean_panel
from .feature_engineering import build_features
from .event_detector import detect_events


COST_RT = 0.008


def get_bulan_berjalan():
    """Return (start, end, label_bulan, label_file) untuk bulan berjalan."""
    today = pd.Timestamp(datetime.now().date())
    start = today.replace(day=1)
    if today.month == 12:
        end = today.replace(year=today.year + 1, month=1, day=1) - pd.Timedelta(days=1)
    else:
        end = today.replace(month=today.month + 1, day=1) - pd.Timedelta(days=1)

    label_bulan = today.strftime("%B %Y")     # "October 2026"
    label_file = today.strftime("%Y-%m")      # "2026-10"
    return start, end, label_bulan, label_file


def jalankan_update():
    print(f"\n{'='*100}")
    print("STEP 1: Update data dari IDX")
    print(f"{'='*100}\n")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    update_script = os.path.join(root, "update.py")
    gabung_script = os.path.join(root, "gabung.py")

    if not os.path.exists(update_script):
        print(f"[!] File {update_script} tidak ditemukan.")
        return False

    print("Menjalankan update.py...")
    try:
        result = subprocess.run(
            [sys.executable, update_script],
            cwd=root, capture_output=True, text=True, timeout=1800,
        )
        print(result.stdout[-1500:] if result.stdout else "(tidak ada output)")
        if result.returncode != 0:
            print(f"[!] Error update:")
            print(result.stderr[-800:] if result.stderr else "")
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
            result = subprocess.run(
                [sys.executable, gabung_script],
                cwd=root, capture_output=True, text=True, timeout=300,
            )
            print(result.stdout[-800:] if result.stdout else "(selesai)")
        except Exception:
            pass

    print("\n[OK] Update selesai.")
    return True


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
        df[f"close_T{h}"] = df.groupby("kode")["close"].shift(-h)
    return df


def simulate(row):
    close_t = row["close"]
    entry = row["open_T1"]
    if pd.isna(entry) or entry <= 0:
        return None, "NO_DATA"
    if entry < close_t:
        return None, "SKIP_GAP_DOWN"
    high = row["high_T1"]
    if pd.isna(high) or high <= 0:
        return None, "PENDING"
    return high / entry - 1 - COST_RT, "OK"


def excel_style(ws, title_text, subtitle_text, header_row=4):
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    ws.merge_cells(start_row=1, start_column=1, end_row=1,
                   end_column=ws.max_column)
    title_cell = ws.cell(row=1, column=1)
    title_cell.value = title_text
    title_cell.font = Font(bold=True, size=16, color="1F4E78")
    title_cell.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 25

    ws.merge_cells(start_row=2, start_column=1, end_row=2,
                   end_column=ws.max_column)
    sub_cell = ws.cell(row=2, column=1)
    sub_cell.value = subtitle_text
    sub_cell.font = Font(italic=True, size=10, color="666666")
    sub_cell.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[2].height = 15

    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78",
                               fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True, size=11)
    thin_border = Border(
        left=Side(style="thin", color="D0D0D0"),
        right=Side(style="thin", color="D0D0D0"),
        top=Side(style="thin", color="D0D0D0"),
        bottom=Side(style="thin", color="D0D0D0"),
    )

    for cell in ws[header_row]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center",
                                    wrap_text=True)
        cell.border = thin_border
    ws.row_dimensions[header_row].height = 30

    ws.freeze_panes = ws.cell(row=header_row + 1, column=1).coordinate
    return thin_border


def autofit_columns(ws, skip_rows=3):
    from openpyxl.utils import get_column_letter

    for col_idx, col in enumerate(ws.iter_cols(min_row=skip_rows + 1,
                                                 max_row=ws.max_row), 1):
        max_len = 0
        col_letter = get_column_letter(col_idx)
        for cell in col:
            if cell.value:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = min(max_len + 3, 25)


def simpan_excel(kandidat, signals_bulan, pred_date, path_dir,
                 bulan_label, bulan_file):
    """
    Simpan Excel KHUSUS bulan berjalan.
    signals_bulan = hanya sinyal dari bulan berjalan.
    """
    try:
        from openpyxl import load_workbook
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter
        from openpyxl.formatting.rule import CellIsRule
    except ImportError:
        print("  [!] openpyxl tidak terinstall.")
        return

    # Nama file pakai bulan
    filepath = os.path.join(path_dir, f"Rekap_Trading_{bulan_file}.xlsx")

    # ============================================================
    # SHEET 1: KANDIDAT BESOK
    # ============================================================
    if not kandidat.empty:
        df_kand = kandidat[["kode", "close", "value", "ret_1d", "ret_5d",
                             "range", "vol_z_20", "px_vs_sma20", "match"]].copy()
        df_kand["ret_1d"] = (df_kand["ret_1d"] * 100).round(2)
        df_kand["ret_5d"] = (df_kand["ret_5d"] * 100).round(2)
        df_kand["range"] = (df_kand["range"] * 100).round(2)
        df_kand["vol_z_20"] = df_kand["vol_z_20"].round(2)
        df_kand["px_vs_sma20"] = (df_kand["px_vs_sma20"] * 100).round(2)
        df_kand["value"] = (df_kand["value"] / 1e9).round(2)
        df_kand = df_kand.rename(columns={
            "kode": "Kode", "close": "Close", "value": "Value (M)",
            "ret_1d": "Ret1d%", "ret_5d": "Ret5d%",
            "range": "Range%", "vol_z_20": "VolZ",
            "px_vs_sma20": "Px>sma20%", "match": "Match",
        })
        df_kand.insert(0, "No", range(1, len(df_kand) + 1))
    else:
        df_kand = pd.DataFrame({"Info": ["Tidak ada kandidat"]})

    # ============================================================
    # SHEET 2: REVIEW HARIAN (hanya bulan berjalan)
    # ============================================================
    if not signals_bulan.empty:
        review_rows = []
        for _, row in signals_bulan.sort_values("date").iterrows():
            tgl = pd.Timestamp(row["date"])
            ret, status = simulate(row)

            match = ""
            if row.get("sig_a", False):
                match += "A"
            if row.get("sig_b", False):
                match += "B"

            entry = row["open_T1"]
            high = row["high_T1"]

            review_rows.append({
                "Tanggal": tgl.date(),
                "Hari": ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat',
                         'Sabtu', 'Minggu'][tgl.weekday()],
                "Kode": row["kode"],
                "Close T": round(row["close"], 0),
                "Entry": round(entry, 0) if not pd.isna(entry) else "",
                "High T+1": round(high, 0) if not pd.isna(high) else "",
                "Match": match,
                "Status": status,
                "P/L %": round(ret * 100, 2) if ret is not None else "",
            })
        df_review = pd.DataFrame(review_rows)
    else:
        df_review = pd.DataFrame({
            "Info": [f"Belum ada sinyal di {bulan_label}"]
        })

    # ============================================================
    # SHEET 3: REVIEW MINGGUAN (hanya bulan berjalan)
    # ============================================================
    minggu_rows = []
    if not signals_bulan.empty:
        sr = signals_bulan.copy()
        sr["tanggal"] = pd.to_datetime(sr["date"])
        sr["dom"] = sr["tanggal"].dt.day
        sr["minggu_ke"] = ((sr["dom"] - 1) // 7) + 1

        for wk in sorted(sr["minggu_ke"].unique()):
            wk_data = sr[sr["minggu_ke"] == wk]
            n_sig = len(wk_data)
            n_exec = 0
            n_win = 0
            n_skip = 0
            n_pending = 0
            pl_total = 0

            for _, row in wk_data.iterrows():
                ret, status = simulate(row)
                if status == "SKIP_GAP_DOWN":
                    n_skip += 1
                elif status in ("PENDING", "NO_DATA"):
                    n_pending += 1
                else:
                    n_exec += 1
                    pl_total += ret
                    if ret > 0:
                        n_win += 1

            tgl_min = wk_data["tanggal"].min().date()
            tgl_max = wk_data["tanggal"].max().date()

            minggu_rows.append({
                "Minggu Ke": f"Minggu {wk}",
                "Periode": f"{tgl_min} s/d {tgl_max}",
                "Sinyal": n_sig,
                "Eksekusi": n_exec,
                "Profit": n_win,
                "Skip": n_skip,
                "Pending": n_pending,
                "Win Rate %": round(n_win / n_exec * 100, 1) if n_exec > 0 else 0,
                "Total P/L %": round(pl_total * 100, 2),
                "Avg P/L %": round(pl_total / n_exec * 100, 2) if n_exec > 0 else 0,
            })

    if minggu_rows:
        df_minggu = pd.DataFrame(minggu_rows)
    else:
        df_minggu = pd.DataFrame({
            "Info": [f"Belum ada sinyal mingguan di {bulan_label}"]
        })

    # ============================================================
    # SHEET 4: RINGKASAN (hanya bulan berjalan)
    # ============================================================
    if not signals_bulan.empty:
        total_pl = 0
        n_exec = 0
        n_win = 0
        n_skip = 0
        n_pending = 0

        for _, row in signals_bulan.iterrows():
            ret, status = simulate(row)
            if status == "SKIP_GAP_DOWN":
                n_skip += 1
            elif status in ("PENDING", "NO_DATA"):
                n_pending += 1
            else:
                n_exec += 1
                total_pl += ret
                if ret > 0:
                    n_win += 1

        win_rate = (n_win / n_exec * 100) if n_exec > 0 else 0
        avg_pl = (total_pl / n_exec * 100) if n_exec > 0 else 0

        ringkasan_rows = [
            ["PERIODE", f"Bulan {bulan_label}"],
            ["", ""],
            ["RINGKASAN SINYAL", ""],
            ["Total Sinyal", len(signals_bulan)],
            ["Tereksekusi", n_exec],
            ["Skip (gap down)", n_skip],
            ["Pending", n_pending],
            ["", ""],
            ["PERFORMA", ""],
            ["Win Rate", f"{win_rate:.1f}%"],
            ["Total P/L", f"{total_pl*100:+.2f}%"],
            ["Avg P/L per Trade", f"{avg_pl:+.2f}%"],
            ["", ""],
            ["VERDICT", ""],
            ["Status", "[OK] LAYAK" if win_rate >= 60 and avg_pl > 2
             else ("[!] MARGINAL" if win_rate >= 50 else "[X] EVALUASI")],
        ]
    else:
        ringkasan_rows = [
            ["PERIODE", f"Bulan {bulan_label}"],
            ["Info", "Belum ada sinyal"],
        ]

    df_ringkasan = pd.DataFrame(ringkasan_rows,
                                  columns=["Metrik", "Nilai"])

    # ============================================================
    # TULIS EXCEL
    # ============================================================
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df_ringkasan.to_excel(writer, sheet_name="Ringkasan",
                               index=False, startrow=3)
        df_kand.to_excel(writer, sheet_name="Kandidat Besok",
                          index=False, startrow=3)
        df_minggu.to_excel(writer, sheet_name="Review Mingguan",
                            index=False, startrow=3)
        df_review.to_excel(writer, sheet_name="Review Harian",
                            index=False, startrow=3)

    # ============================================================
    # FORMATTING
    # ============================================================
    wb = load_workbook(filepath)

    GREEN_FILL = PatternFill(start_color="C6EFCE", end_color="C6EFCE",
                              fill_type="solid")
    GREEN_FONT = Font(color="006100", bold=True)
    RED_FILL = PatternFill(start_color="FFC7CE", end_color="FFC7CE",
                            fill_type="solid")
    RED_FONT = Font(color="9C0006", bold=True)
    SECTION_FILL = PatternFill(start_color="D9E1F2", end_color="D9E1F2",
                                fill_type="solid")
    SECTION_FONT = Font(bold=True, size=11, color="1F4E78")

    # Sheet Ringkasan
    ws = wb["Ringkasan"]
    excel_style(ws, f"RINGKASAN TRADING - {bulan_label.upper()}",
                f"Dibuat: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    for row in ws.iter_rows(min_row=5, max_row=ws.max_row,
                             min_col=1, max_col=2):
        cell = row[0]
        if cell.value and isinstance(cell.value, str):
            if cell.value.isupper() and cell.value.strip():
                cell.fill = SECTION_FILL
                cell.font = SECTION_FONT
                row[1].fill = SECTION_FILL

    autofit_columns(ws)

    # Sheet Kandidat
    ws = wb["Kandidat Besok"]
    excel_style(ws, "KANDIDAT BESOK",
                f"Prediksi untuk: {pred_date}")
    autofit_columns(ws)

    if not kandidat.empty:
        for row in ws.iter_rows(min_row=5, max_row=ws.max_row,
                                 min_col=1, max_col=ws.max_column):
            for cell in row:
                cell.alignment = Alignment(horizontal="center")

    # Sheet Review Mingguan
    ws = wb["Review Mingguan"]
    excel_style(ws, f"REVIEW MINGGUAN - {bulan_label.upper()}",
                f"Performa mingguan bulan {bulan_label}")
    autofit_columns(ws)

    if minggu_rows:
        for i, cell in enumerate(ws[4], 1):
            if cell.value in ("Total P/L %", "Avg P/L %"):
                col_letter = get_column_letter(i)
                rng = f"{col_letter}5:{col_letter}{ws.max_row}"
                ws.conditional_formatting.add(
                    rng,
                    CellIsRule(operator="greaterThan", formula=["0"],
                               fill=GREEN_FILL, font=GREEN_FONT)
                )
                ws.conditional_formatting.add(
                    rng,
                    CellIsRule(operator="lessThan", formula=["0"],
                               fill=RED_FILL, font=RED_FONT)
                )

    # Sheet Review Harian
    ws = wb["Review Harian"]
    excel_style(ws, f"REVIEW HARIAN - {bulan_label.upper()}",
                f"Semua sinyal bulan {bulan_label}")
    autofit_columns(ws)

    if not df_review.empty and "P/L %" in df_review.columns:
        for i, cell in enumerate(ws[4], 1):
            if cell.value == "P/L %":
                col_letter = get_column_letter(i)
                rng = f"{col_letter}5:{col_letter}{ws.max_row}"
                ws.conditional_formatting.add(
                    rng,
                    CellIsRule(operator="greaterThan", formula=["0"],
                               fill=GREEN_FILL, font=GREEN_FONT)
                )
                ws.conditional_formatting.add(
                    rng,
                    CellIsRule(operator="lessThan", formula=["0"],
                               fill=RED_FILL, font=RED_FONT)
                )

    wb.active = wb.sheetnames.index("Kandidat Besok")
    wb.save(filepath)
    print(f"  -> Excel: {filepath}")


def main():
    print("=" * 100)
    print(f"AUTO SCREENER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    # Ambil info bulan berjalan
    bulan_start, bulan_end, bulan_label, bulan_file = get_bulan_berjalan()
    print(f"\nBulan berjalan : {bulan_label}")
    print(f"Periode        : {bulan_start.date()} s/d {bulan_end.date()}")

    # STEP 1: Update
    jalankan_update()

    # STEP 2: Load
    print(f"\n{'='*100}")
    print("STEP 2: Proses screening")
    print(f"{'='*100}\n")

    print("Memuat data...")
    try:
        panel = load_panel_from_gabungan()
    except FileNotFoundError:
        print("[!] File data tidak ditemukan.")
        return

    panel = clean_panel(panel)
    panel = build_features(panel)
    panel = detect_events(panel)
    panel = add_forward(panel)
    panel = panel[(panel["open"] > 0) & (panel["close"] > 0)].copy()

    last_date = pd.Timestamp(panel["date"].max())
    today = pd.Timestamp(datetime.now().date())
    pred_date = next_trading_day(last_date)

    print(f"\nData terakhir  : {last_date.date()} "
          f"({['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu'][last_date.weekday()]})")
    print(f"Prediksi untuk : {pred_date.date()} "
          f"({['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu'][pred_date.weekday()]})")

    # STEP 3: Kandidat besok
    today_data = panel[panel["date"] == last_date].copy()
    m_a = rule_a(today_data).fillna(False).values
    m_b = rule_b(today_data).fillna(False).values
    m_union = m_a | m_b

    kandidat = today_data[m_union].copy()

    print(f"\n{'#'*100}")
    print(f"#  KANDIDAT UNTUK {pred_date.date()}")
    print(f"{'#'*100}")

    if not kandidat.empty:
        kandidat["match"] = ""
        kandidat.loc[m_a[m_union], "match"] += "A"
        kandidat.loc[m_b[m_union], "match"] += "B"
        kandidat = kandidat.sort_values("value", ascending=False)

        print(f"\n  {len(kandidat)} KANDIDAT:\n")
        print(f"  {'No':>3} {'Kode':<7} {'Close':>7} {'Ret1d%':>8} "
              f"{'Ret5d%':>8} {'Range%':>8} {'VolZ':>6} "
              f"{'Px>sma20':>9} {'Match':>5} {'Value(M)':>10}")
        print("  " + "-" * 90)

        for i, (_, r) in enumerate(kandidat.iterrows(), 1):
            print(f"  {i:>3} {r['kode']:<7} {r['close']:>7.0f} "
                  f"{r['ret_1d']*100:>+7.1f}% {r['ret_5d']*100:>+7.1f}% "
                  f"{r['range']*100:>7.1f}% {r['vol_z_20']:>5.2f} "
                  f"{r['px_vs_sma20']*100:>+8.1f}% {r['match']:>5} "
                  f"{r['value']/1e9:>9.2f}")
    else:
        print(f"\n  Tidak ada kandidat untuk {pred_date.date()}.")

    # STEP 4: Review BULAN BERJALAN SAJA
    print(f"\n{'#'*100}")
    print(f"#  REVIEW BULAN {bulan_label.upper()}")
    print(f"{'#'*100}")

    # FILTER: hanya bulan berjalan
    sub_bulan = panel[(panel["date"] >= bulan_start) &
                       (panel["date"] <= bulan_end)].copy()

    m_a_sub = rule_a(sub_bulan).fillna(False).values
    m_b_sub = rule_b(sub_bulan).fillna(False).values
    m_union_sub = m_a_sub | m_b_sub

    signals_bulan = sub_bulan[m_union_sub].copy()
    signals_bulan["sig_a"] = m_a_sub[m_union_sub]
    signals_bulan["sig_b"] = m_b_sub[m_union_sub]

    if not signals_bulan.empty:
        total_pl = 0
        n_exec = 0
        n_win = 0
        n_skip = 0
        n_pending = 0

        for _, row in signals_bulan.iterrows():
            ret, status = simulate(row)
            if status == "SKIP_GAP_DOWN":
                n_skip += 1
            elif status in ("PENDING", "NO_DATA"):
                n_pending += 1
            else:
                n_exec += 1
                total_pl += ret
                if ret > 0:
                    n_win += 1

        print(f"\n  Periode        : {bulan_label}")
        print(f"  Total sinyal   : {len(signals_bulan)}")
        print(f"  Skip           : {n_skip}")
        print(f"  Pending        : {n_pending}")
        print(f"  Tereksekusi    : {n_exec}")

        if n_exec > 0:
            print(f"  Win rate       : {n_win}/{n_exec} "
                  f"({n_win/n_exec*100:.1f}%)")
            print(f"  Total P/L      : {total_pl*100:+.2f}%")
            print(f"  Avg per trade  : {total_pl/n_exec*100:+.2f}%")
    else:
        print(f"\n  Belum ada sinyal di bulan {bulan_label}.")
        print(f"  (Ini normal kalau bulan baru mulai)")

    # STEP 5: Simpan Excel KHUSUS BULAN BERJALAN
    print(f"\n{'#'*100}")
    print(f"#  MENYIMPAN EXCEL BULAN {bulan_label.upper()}")
    print(f"{'#'*100}\n")

    os.makedirs("quant/output", exist_ok=True)

    if not kandidat.empty:
        csv_path = f"quant/output/kandidat_{pred_date.date()}.csv"
        kandidat[["kode", "close", "value", "ret_1d", "ret_5d",
                  "range", "vol_z_20", "px_vs_sma20", "match"]].to_csv(
            csv_path, index=False)
        print(f"  -> CSV: {csv_path}")

    # PENTING: kirim signals_bulan (bukan signals_all)
    simpan_excel(kandidat, signals_bulan, pred_date.date(),
                 "quant/output", bulan_label, bulan_file)

    # Cek file lain yang ada
    print(f"\n  File Excel di folder output:")
    for f in sorted(os.listdir("quant/output")):
        if f.endswith(".xlsx"):
            print(f"    - {f}")

    print(f"\n{'='*100}")
    print(f"SELESAI - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()