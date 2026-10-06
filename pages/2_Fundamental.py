"""Halaman FUNDAMENTAL — tanpa grafik, input manual, minimalis."""

import os, sys, json
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant.theme import load_theme, page_header, section_title, empty_state

st.set_page_config(page_title="Fundamental", page_icon="📈", layout="wide")
load_theme()

DETAIL_DIR = "data_fundamental_detail"


def _rgb(hex_color):
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)}, {int(h[2:4],16)}, {int(h[4:6],16)}"


@st.cache_data(ttl=1800)
def list_available():
    if not os.path.exists(DETAIL_DIR): return []
    return sorted(f.replace(".json", "") for f in os.listdir(DETAIL_DIR) if f.endswith(".json"))


@st.cache_data(ttl=1800)
def load_detail(kode):
    path = os.path.join(DETAIL_DIR, f"{kode}.json")
    if not os.path.exists(path): return None
    with open(path, "r", encoding="utf-8") as f: return json.load(f)


@st.cache_data(ttl=1800)
def load_ohlcv(kode):
    path = os.path.join("data_ohlcv_idx", f"{kode}.csv")
    if not os.path.exists(path): return None
    try:
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        return df.sort_values("Date").reset_index(drop=True)
    except: return None


def fmt_num(v, dec=2, suffix=""):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    try:
        if abs(v) >= 1e12: return f"{v/1e12:.{dec}f}T{suffix}"
        if abs(v) >= 1e9: return f"{v/1e9:.{dec}f}B{suffix}"
        if abs(v) >= 1e6: return f"{v/1e6:.{dec}f}M{suffix}"
        return f"{v:,.{dec}f}{suffix}"
    except: return "—"


def fmt_pct(v, dec=2, plus=False):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    try:
        if abs(v) < 1: v *= 100
        return f"{v:+.{dec}f}%" if plus else f"{v:.{dec}f}%"
    except: return "—"


# ============================================================
# HEADER
# ============================================================
st.markdown(page_header("📈", "Fundamental", "Analisis Rasio Keuangan"),
            unsafe_allow_html=True)

saham_list = list_available()
if not saham_list:
    st.markdown(empty_state("📭", "Belum ada data",
                            "Jalankan: python -m quant.fundamental_detail"),
                unsafe_allow_html=True)
    st.stop()

# ============================================================
# INPUT MANUAL (tanpa chips)
# ============================================================
if "fund_kode" not in st.session_state:
    st.session_state.fund_kode = "BBCA"

kode_input = st.text_input(
    "Kode Saham",
    value=st.session_state.fund_kode,
    placeholder="Ketik kode saham, mis. BBCA",
    max_chars=6,
).upper().strip()

if not kode_input:
    st.info("Ketik kode saham di atas untuk mulai.")
    st.stop()

if kode_input not in saham_list:
    st.warning(f"⚠️ Saham **{kode_input}** tidak ditemukan.")
    st.caption(f"Total tersedia: {len(saham_list)} saham. Contoh: BBCA, BBRI, TLKM, ASII, GOTO, ANTM")
    st.stop()

st.session_state.fund_kode = kode_input
kode_pilih = kode_input

data = load_detail(kode_pilih)
if not data:
    st.error(f"Data {kode_pilih} tidak ditemukan.")
    st.stop()


# ============================================================
# HEADER SAHAM
# ============================================================
harga = data.get("harga") or 0
chg = 0
df_ohlcv = load_ohlcv(kode_pilih)

if df_ohlcv is not None and len(df_ohlcv) >= 2:
    last = float(df_ohlcv["Close"].iloc[-1])
    prev = float(df_ohlcv["Close"].iloc[-2])
    chg = (last / prev - 1) * 100
    harga = last

chg_color = "#10b981" if chg >= 0 else "#f43f5e"
chg_sign = "+" if chg >= 0 else ""

st.markdown(
    '<div style="'
    'background: #0f172a; border: 1px solid #1e293b; '
    'border-radius: 12px; padding: 22px 24px; margin-bottom: 24px; '
    'display: flex; align-items: center; gap: 18px;'
    '">'
    f'<div style="width:60px; height:60px; background: #1e293b; '
    f'border-radius: 12px; display: flex; align-items: center; '
    f'justify-content: center; font-size: 22px; font-weight: 800; '
    f'color: #06b6d4; letter-spacing: -0.5px;">{kode_pilih[:2]}</div>'
    '<div style="flex-grow: 1;">'
    f'<div style="font-size:12px; color:#64748b; margin-bottom:5px; '
    f'font-weight: 500;">{data.get("sektor", "")} · {data.get("industri", "")}</div>'
    f'<div style="font-size:18px; font-weight:700; color:#ffffff; '
    f'margin-bottom:8px;">{data.get("nama", kode_pilih)}</div>'
    '<div style="display:flex; align-items:baseline; gap:12px;">'
    f'<span style="font-size:28px; font-weight:800; color:#ffffff; '
    f'letter-spacing:-0.03em; line-height:1;">{harga:,.0f}</span>'
    f'<span style="font-size:13px; font-weight:700; padding:4px 10px; '
    f'border-radius:6px; background:rgba({_rgb(chg_color)}, 0.12); '
    f'color:{chg_color}; border:1px solid {chg_color}40;">'
    f'{chg_sign}{chg:.2f}%</span>'
    '</div></div></div>',
    unsafe_allow_html=True
)


# ============================================================
# RASIO KUNCI
# ============================================================
rk = data.get("rasio_kunci", {})
tahun = data.get("update", "")[:4] or "2026"

st.markdown(section_title("Rasio Kunci", f"Tahun fiskal {tahun}"),
            unsafe_allow_html=True)

metrics = [
    ("P/E", fmt_num(rk.get("pe"), 2)),
    ("P/B", fmt_num(rk.get("pb"), 2)),
    ("P/S", fmt_num(rk.get("ps"), 2)),
    ("ROE", fmt_pct(rk.get("roe"))),
    ("EPS", fmt_num(rk.get("eps"), 2)),
    ("D/E", fmt_num(rk.get("de"), 2)),
    ("Margin Bersih", fmt_pct(rk.get("margin_net"))),
    ("Current Ratio", fmt_num(rk.get("current_ratio"), 2)),
]

cols = st.columns(2)
for i, (label, val) in enumerate(metrics):
    with cols[i % 2]:
        st.markdown(
            '<div style="background:#0f172a; border:1px solid #1e293b; '
            'border-radius:10px; padding:14px 18px; margin-bottom:10px; '
            'display:flex; justify-content:space-between; align-items:center;">'
            f'<div style="font-size:13px; color:#94a3b8;">{label}</div>'
            f'<div style="font-size:19px; font-weight:700; color:#ffffff; '
            f'letter-spacing:-0.02em;">{val}</div>'
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# RIWAYAT TAHUNAN
# ============================================================
riwayat = data.get("riwayat", {})
eps_hist = riwayat.get("eps", {}) or {}
rev_hist = riwayat.get("revenue_per_share", {}) or riwayat.get("revenue", {}) or {}
tahun_list = sorted(set(list(eps_hist.keys()) + list(rev_hist.keys())), reverse=True)[:5]

if tahun_list:
    st.markdown(section_title("Riwayat Tahunan", "EPS & pendapatan per tahun"),
                unsafe_allow_html=True)

    rows = ""
    for y in tahun_list:
        e = eps_hist.get(y); r = rev_hist.get(y)
        rows += (
            '<tr>'
            f'<td style="padding:12px 14px; color:#e6ebf5; font-weight:600; '
            f'border-bottom:1px solid #1e293b; font-size:13px;">{y}</td>'
            f'<td style="padding:12px 14px; color:#e6ebf5; '
            f'border-bottom:1px solid #1e293b; font-size:13px;">'
            f'{fmt_num(e, 0) if e else "—"}</td>'
            f'<td style="padding:12px 14px; color:#e6ebf5; '
            f'border-bottom:1px solid #1e293b; font-size:13px;">'
            f'{fmt_num(r, 0) if r else "—"}</td>'
            '</tr>'
        )

    st.markdown(
        '<div style="background:#0f172a; border:1px solid #1e293b; '
        'border-radius:12px; padding:4px 4px; margin-bottom:12px; overflow:hidden;">'
        '<table style="width:100%; border-collapse:collapse;">'
        '<thead><tr>'
        '<th style="text-align:left; padding:12px 14px; font-size:11px; '
        'color:#64748b; font-weight:600; letter-spacing:0.06em; '
        'text-transform:uppercase; border-bottom:1px solid #1e293b;">Tahun</th>'
        '<th style="text-align:left; padding:12px 14px; font-size:11px; '
        'color:#64748b; font-weight:600; letter-spacing:0.06em; '
        'text-transform:uppercase; border-bottom:1px solid #1e293b;">EPS</th>'
        '<th style="text-align:left; padding:12px 14px; font-size:11px; '
        'color:#64748b; font-weight:600; letter-spacing:0.06em; '
        'text-transform:uppercase; border-bottom:1px solid #1e293b;">'
        'Pendapatan/Saham</th>'
        '</tr></thead>'
        f'<tbody>{rows}</tbody>'
        '</table></div>',
        unsafe_allow_html=True
    )


# ============================================================
# STATISTIK
# ============================================================
stat = data.get("statistik", {})
st.markdown(section_title("Statistik & Laporan Keuangan", "Mata uang: IDR"),
            unsafe_allow_html=True)

stat_items = [
    ("Market Cap", fmt_num(stat.get("market_cap"), 0)),
    ("Enterprise Value", fmt_num(stat.get("enterprise_value"), 0)),
    ("Saham Beredar", fmt_num(stat.get("shares_outstanding"), 0)),
    ("Free Float", fmt_pct(stat.get("free_float_pct"))),
]

cols = st.columns(2)
for i, (label, val) in enumerate(stat_items):
    with cols[i % 2]:
        st.markdown(
            '<div style="background:#0f172a; border:1px solid #1e293b; '
            'border-radius:10px; padding:14px 18px; margin-bottom:10px; '
            'display:flex; justify-content:space-between; align-items:center;">'
            f'<div style="font-size:13px; color:#94a3b8;">{label}</div>'
            f'<div style="font-size:17px; font-weight:700; color:#ffffff; '
            f'letter-spacing:-0.02em;">{val}</div>'
            '</div>',
            unsafe_allow_html=True
        )

# Earnings
st.markdown(
    '<div style="font-size:11px; color:#64748b; letter-spacing:0.08em; '
    'font-weight:600; text-transform:uppercase; margin:18px 0 10px 2px;">'
    'EARNINGS TERAKHIR</div>',
    unsafe_allow_html=True
)

eps_str = f"{rk.get('eps'):.0f}" if rk.get("eps") else "—"
rev_aktual = riwayat.get("revenue", {})
if isinstance(rev_aktual, dict) and rev_aktual:
    rev_key = sorted(rev_aktual.keys(), reverse=True)[0]
    rev_val = rev_aktual[rev_key]
    rev_str = fmt_num(rev_val, 0)
else:
    rev_str = "—"

st.markdown(
    '<div style="background:#0f172a; border:1px solid #1e293b; '
    'border-radius:10px; padding:14px 18px; margin-bottom:8px; '
    'display:flex; justify-content:space-between; align-items:center;">'
    '<div style="font-size:13px; color:#94a3b8;">EPS Aktual</div>'
    f'<div style="font-size:17px; font-weight:700; color:#06b6d4;">{eps_str}</div>'
    '</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div style="background:#0f172a; border:1px solid #1e293b; '
    'border-radius:10px; padding:14px 18px; margin-bottom:8px; '
    'display:flex; justify-content:space-between; align-items:center;">'
    '<div style="font-size:13px; color:#94a3b8;">Revenue Aktual</div>'
    f'<div style="font-size:17px; font-weight:700; color:#06b6d4;">{rev_str}</div>'
    '</div>',
    unsafe_allow_html=True
)

# Rasio Ekuitas
st.markdown(
    f'<div style="font-size:11px; color:#64748b; letter-spacing:0.08em; '
    f'font-weight:600; text-transform:uppercase; margin:18px 0 10px 2px;">'
    f'RASIO EKUITAS {tahun}</div>',
    unsafe_allow_html=True
)

roe_str = fmt_pct(rk.get("roe"))
roa_str = fmt_pct(rk.get("roa"))

c1, c2 = st.columns(2)
for col, label, val, color in [
    (c1, "ROE", roe_str, "#10b981"),
    (c2, "ROA", roa_str, "#06b6d4"),
]:
    with col:
        st.markdown(
            '<div style="background:#0f172a; border:1px solid #1e293b; '
            'border-radius:10px; padding:16px 18px;">'
            f'<div style="font-size:11px; color:#64748b; font-weight:600; '
            f'letter-spacing:0.06em; text-transform:uppercase; '
            f'margin-bottom:6px;">{label}</div>'
            f'<div style="font-size:24px; font-weight:800; color:{color}; '
            f'letter-spacing:-0.02em;">{val}</div>'
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# AKUMULASI
# ============================================================
akum = data.get("akumulasi") or {}
if akum:
    st.markdown(section_title("Akumulasi vs Distribusi", "20 hari terakhir"),
                unsafe_allow_html=True)

    score = akum.get("score", 5)
    label = akum.get("label", "NETRAL")
    tipe = akum.get("tipe", "NEUTRAL")
    if tipe == "ACCUMULATION": color = "#10b981"
    elif tipe == "DISTRIBUTION": color = "#f43f5e"
    else: color = "#f59e0b"
    cmf = akum.get("cmf", 0)
    vol_trend = akum.get("vol_trend", 1)
    price_chg = akum.get("price_chg_20d", 0)

    st.markdown(
        '<div style="background:#0f172a; border:1px solid #1e293b; '
        f'border-left:3px solid {color}; border-radius:12px; '
        'padding:20px 22px;">'
        '<div style="display:flex; justify-content:space-between; '
        'align-items:center; margin-bottom:14px;">'
        '<div>'
        f'<div style="font-size:14px; font-weight:700; color:{color}; '
        f'margin-bottom:4px;">{label}</div>'
        f'<div style="font-size:12px; color:#94a3b8;">'
        f'CMF: <b style="color:#e6ebf5;">{cmf:+.3f}</b> · '
        f'Vol trend: <b style="color:#e6ebf5;">{vol_trend:.2f}x</b>'
        f'</div>'
        '</div>'
        f'<div style="font-size:30px; font-weight:800; color:{color}; '
        f'letter-spacing:-0.03em; line-height:1;">'
        f'{score:.1f}'
        '<span style="font-size:13px; color:#64748b; font-weight:600;">/10</span>'
        '</div>'
        '</div>'
        '<div style="display:flex; gap:8px;">'
        '<div style="flex:1; background:#0a0e1a; border-radius:8px; '
        'padding:10px 14px; border:1px solid #1e293b;">'
        '<div style="font-size:10px; color:#64748b; font-weight:600; '
        'text-transform:uppercase; letter-spacing:0.06em; '
        'margin-bottom:3px;">Chaikin MF</div>'
        f'<div style="font-size:15px; font-weight:700; color:#e6ebf5;">'
        f'{cmf:+.3f}</div></div>'
        '<div style="flex:1; background:#0a0e1a; border-radius:8px; '
        'padding:10px 14px; border:1px solid #1e293b;">'
        '<div style="font-size:10px; color:#64748b; font-weight:600; '
        'text-transform:uppercase; letter-spacing:0.06em; '
        'margin-bottom:3px;">Perubahan 20H</div>'
        f'<div style="font-size:15px; font-weight:700; color:#e6ebf5;">'
        f'{price_chg:+.1f}%</div></div>'
        '</div></div>',
        unsafe_allow_html=True
    )

st.markdown(
    '<div style="text-align:center; color:#475569; font-size:11px; '
    'margin-top:30px; padding-top:20px; border-top:1px solid #1e293b;">'
    f'Sumber: yfinance · Update: {data.get("update", "")}'
    '</div>',
    unsafe_allow_html=True
)