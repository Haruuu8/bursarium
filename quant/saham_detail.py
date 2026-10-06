"""
Halaman DETAIL SAHAM — 8 section ala Stockbit.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant.technical import (
    hitung_indikator_teknikal,
    hitung_risk_reward,
    hitung_sinyal_trading,
)

st.set_page_config(page_title="Detail Saham", page_icon="📊", layout="wide")

DETAIL_DIR = "data_fundamental_detail"


# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #0a0e27 0%, #121a3a 100%); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1.5rem; max-width: 900px; }

    .stock-header { display: flex; align-items: center; gap: 14px; margin-bottom: 20px; }
    .stock-icon {
        width: 54px; height: 54px;
        background: linear-gradient(135deg, #2563eb, #1e40af);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 22px; font-weight: 700; color: white;
        flex-shrink: 0;
    }
    .stock-title { font-size: 14px; color: #8091b8; margin-bottom: 4px; }
    .stock-name { font-size: 20px; font-weight: 700; color: #ffffff; margin-bottom: 6px; }
    .stock-price {
        font-size: 24px; font-weight: 700; color: #ffffff;
        display: flex; align-items: center; gap: 8px;
    }
    .price-change { font-size: 14px; font-weight: 600; padding: 3px 10px; border-radius: 6px; }
    .price-up { background: rgba(0,212,122,0.15); color: #00d47a; }
    .price-down { background: rgba(255,82,82,0.15); color: #ff5252; }

    .section-card {
        background: #131836;
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 18px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .section-title { font-size: 17px; font-weight: 700; color: #ffffff; margin-bottom: 4px; }
    .section-subtitle { font-size: 12px; color: #8091b8; margin-bottom: 16px; }

    .metric-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
    .metric-box {
        background: #0f1530;
        border-radius: 10px;
        padding: 14px 16px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .metric-box-label { font-size: 12px; color: #8091b8; margin-bottom: 6px; }
    .metric-box-value { font-size: 22px; font-weight: 700; color: #ffffff; line-height: 1.2; }
    .metric-box-value.small { font-size: 18px; }

    .history-table { width: 100%; border-collapse: collapse; }
    .history-table th {
        text-align: left; padding: 10px 12px;
        font-size: 12px; color: #8091b8; font-weight: 600;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .history-table td {
        padding: 12px; font-size: 14px; color: #ffffff;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .history-table tr:last-child td { border-bottom: none; }

    .bar-row { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
    .bar-label { width: 50px; font-size: 12px; color: #8091b8; flex-shrink: 0; }
    .bar-track {
        flex-grow: 1; height: 22px; background: #0f1530;
        border-radius: 4px; overflow: hidden;
    }
    .bar-fill { height: 100%; background: linear-gradient(90deg, #2563eb, #3b82f6); }
    .bar-value { width: 60px; font-size: 12px; color: #ffffff; text-align: right; flex-shrink: 0; }

    .accum-card {
        display: flex; justify-content: space-between; align-items: center;
        padding: 16px 18px; background: #0f1530; border-radius: 10px;
        margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.05);
    }
    .accum-score { font-size: 32px; font-weight: 700; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOADER
# ============================================================
@st.cache_data(ttl=1800)
def list_available():
    if not os.path.exists(DETAIL_DIR):
        return []
    return sorted(f.replace(".json", "") for f in os.listdir(DETAIL_DIR)
                  if f.endswith(".json"))


@st.cache_data(ttl=1800)
def load_detail(kode):
    path = os.path.join(DETAIL_DIR, f"{kode}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# FORMATTERS
# ============================================================
def fmt_num(v, dec=2, suffix=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    try:
        if abs(v) >= 1e12: return f"{v/1e12:.{dec}f}T{suffix}"
        if abs(v) >= 1e9: return f"{v/1e9:.{dec}f}B{suffix}"
        if abs(v) >= 1e6: return f"{v/1e6:.{dec}f}M{suffix}"
        return f"{v:,.{dec}f}{suffix}"
    except Exception:
        return "—"


def fmt_pct(v, dec=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    try:
        if abs(v) < 1: v *= 100
        return f"{v:+.{dec}f}%"
    except Exception:
        return "—"


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style="margin-bottom: 8px;">
    <div style="font-size: 11px; color: #8091b8; letter-spacing: 1px;">DETAIL SAHAM</div>
</div>
""", unsafe_allow_html=True)

saham_list = list_available()
if not saham_list:
    st.warning("⚠️ Belum ada data detail. Jalankan dulu:")
    st.code("python -m quant.fundamental_detail")
    st.stop()

col1, col2 = st.columns([2, 3])
with col1:
    default_idx = saham_list.index("BBCA") if "BBCA" in saham_list else 0
    kode_pilih = st.selectbox("Pilih Saham", saham_list, index=default_idx)
with col2:
    st.write("")

data = load_detail(kode_pilih)
if not data:
    st.error(f"Data {kode_pilih} tidak ditemukan.")
    st.stop()


# ============================================================
# HEADER SAHAM
# ============================================================
harga = data.get("harga") or 0
chg = 0
try:
    ohlcv_path = os.path.join("data_ohlcv_idx", f"{kode_pilih}.csv")
    if os.path.exists(ohlcv_path):
        df = pd.read_csv(ohlcv_path)
        if len(df) >= 2:
            last = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2])
            chg = (last / prev - 1) * 100
            harga = last
except Exception:
    pass

chg_class = "price-up" if chg >= 0 else "price-down"
chg_sign = "+" if chg >= 0 else ""

st.markdown(f"""
<div class="stock-header">
    <div class="stock-icon">{kode_pilih[:1]}</div>
    <div style="flex-grow: 1;">
        <div class="stock-title">{data.get('sektor', '')} · {data.get('industri', '')}</div>
        <div class="stock-name">{data.get('nama', kode_pilih)}</div>
        <div class="stock-price">
            {harga:,.0f}
            <span class="price-change {chg_class}">{chg_sign}{chg:.2f}%</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SECTION 1: RASIO KUNCI
# ============================================================
rk = data.get("rasio_kunci", {})
tahun = data.get("update", "")[:4] or "2026"

st.markdown(f"""
<div class="section-card">
    <div class="section-title">Rasio Kunci</div>
    <div class="section-subtitle">Tahun fiskal {tahun}</div>
    <div class="metric-grid">
        <div class="metric-box"><div class="metric-box-label">P/E</div>
            <div class="metric-box-value">{fmt_num(rk.get('pe'), 2)}</div></div>
        <div class="metric-box"><div class="metric-box-label">P/B</div>
            <div class="metric-box-value">{fmt_num(rk.get('pb'), 2)}</div></div>
        <div class="metric-box"><div class="metric-box-label">P/S</div>
            <div class="metric-box-value">{fmt_num(rk.get('ps'), 2)}</div></div>
        <div class="metric-box"><div class="metric-box-label">ROE</div>
            <div class="metric-box-value">{fmt_pct(rk.get('roe'))}</div></div>
        <div class="metric-box"><div class="metric-box-label">EPS</div>
            <div class="metric-box-value small">{fmt_num(rk.get('eps'), 2)}</div></div>
        <div class="metric-box"><div class="metric-box-label">D/E</div>
            <div class="metric-box-value">{fmt_num(rk.get('de'), 2)}</div></div>
        <div class="metric-box"><div class="metric-box-label">Margin Bersih</div>
            <div class="metric-box-value">{fmt_pct(rk.get('margin_net'))}</div></div>
        <div class="metric-box"><div class="metric-box-label">Current Ratio</div>
            <div class="metric-box-value">{fmt_num(rk.get('current_ratio'), 2)}</div></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SECTION 2: RIWAYAT TAHUNAN
# ============================================================
riwayat = data.get("riwayat", {})
eps_hist = riwayat.get("eps", {}) or {}
rev_hist = riwayat.get("revenue_per_share", {}) or riwayat.get("revenue", {}) or {}
tahun_list = sorted(set(list(eps_hist.keys()) + list(rev_hist.keys())), reverse=True)[:5]

if tahun_list:
    rows_html = ""
    for y in tahun_list:
        eps = eps_hist.get(y)
        rev = rev_hist.get(y)
        rows_html += f"""<tr><td>{y}</td>
            <td>{fmt_num(eps, 0) if eps else '—'}</td>
            <td>{fmt_num(rev, 0) if rev else '—'}</td></tr>"""

    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">Riwayat Tahunan</div>
        <div class="section-subtitle">EPS & pendapatan/saham per tahun fiskal</div>
        <table class="history-table">
            <thead><tr><th>Tahun</th><th>EPS</th><th>Pendapatan/Saham</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SECTION 3: STATISTIK KUNCI
# ============================================================
stat = data.get("statistik", {})

st.markdown(f"""
<div class="section-card">
    <div class="section-title">Statistik Kunci</div>
    <div class="section-subtitle">Mata uang: IDR</div>
    <div class="metric-grid">
        <div class="metric-box"><div class="metric-box-label">Market Cap</div>
            <div class="metric-box-value small">{fmt_num(stat.get('market_cap'), 0)}</div></div>
        <div class="metric-box"><div class="metric-box-label">Enterprise Value</div>
            <div class="metric-box-value small">{fmt_num(stat.get('enterprise_value'), 0)}</div></div>
        <div class="metric-box"><div class="metric-box-label">Saham Beredar</div>
            <div class="metric-box-value small">{fmt_num(stat.get('shares_outstanding'), 0)}</div></div>
        <div class="metric-box"><div class="metric-box-label">Free Float</div>
            <div class="metric-box-value">{fmt_pct(stat.get('free_float_pct'))}</div></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SECTION 4: TREN RASIO
# ============================================================
tren = data.get("tren", {})
roe_hist = tren.get("roe", {}) or {}
pe_hist = tren.get("pe", {}) or {}


def render_bar_chart(data_dict, title, suffix=""):
    if not data_dict:
        return f'<div style="color: #5a6fa5; font-size: 12px;">Data {title} belum tersedia</div>'
    items = sorted(data_dict.items(), reverse=True)[:8]
    max_val = max(abs(v) for _, v in items) or 1
    html = f'<div style="font-size: 13px; color: #8091b8; margin: 14px 0 10px;">{title}</div>'
    for year, val in items:
        width = min(abs(val) / max_val * 100, 100)
        html += f"""<div class="bar-row">
            <div class="bar-label">{year}</div>
            <div class="bar-track"><div class="bar-fill" style="width: {width}%;"></div></div>
            <div class="bar-value">{val:.1f}{suffix}</div></div>"""
    return html


if roe_hist or pe_hist:
    html_roe = render_bar_chart(roe_hist, "ROE (%)", "%")
    html_pe = render_bar_chart(pe_hist, "P/E Ratio (x)", "x")
    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">Tren Rasio</div>
        <div class="section-subtitle">ROE dan P/E per tahun fiskal</div>
        {html_roe}{html_pe}
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SECTION 5: AKUMULASI vs DISTRIBUSI
# ============================================================
akum = data.get("akumulasi") or {}

if akum:
    score = akum.get("score", 5)
    label = akum.get("label", "NETRAL")
    tipe = akum.get("tipe", "NEUTRAL")

    if tipe == "ACCUMULATION":
        color, icon = "#00d47a", "🟢"
    elif tipe == "DISTRIBUTION":
        color, icon = "#ff5252", "🔴"
    else:
        color, icon = "#ffa726", "🟡"

    cmf = akum.get("cmf", 0)
    vol_trend = akum.get("vol_trend", 1)
    price_chg = akum.get("price_chg_20d", 0)

    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">Akumulasi vs Distribusi</div>
        <div class="section-subtitle">Analisis 20 hari terakhir</div>
        <div class="accum-card">
            <div>
                <div style="font-size: 12px; color: #8091b8; margin-bottom: 6px;">{icon} {label}</div>
                <div style="font-size: 13px; color: #a0aec8;">
                    CMF: {cmf:+.3f} · Volume trend: {vol_trend:.2f}x
                </div>
            </div>
            <div class="accum-score" style="color: {color};">
                {score:.1f}
                <div style="font-size: 10px; color: #5a6fa5; font-weight: 400;">/ 10</div>
            </div>
        </div>
        <div class="metric-grid">
            <div class="metric-box"><div class="metric-box-label">Chaikin Money Flow</div>
                <div class="metric-box-value small">{cmf:+.3f}</div></div>
            <div class="metric-box"><div class="metric-box-label">Perubahan 20 Hari</div>
                <div class="metric-box-value small">{price_chg:+.1f}%</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SECTION 6: INDIKATOR TEKNIKAL
# ============================================================
with st.spinner("Hitung indikator teknikal..."):
    ind = hitung_indikator_teknikal(kode_pilih)

if ind:
    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">Indikator Teknikal</div>
        <div class="section-subtitle">Perhitungan dari data 250 hari terakhir</div>

        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-box-label">RSI (14)</div>
                <div class="metric-box-value">{ind['rsi14']:.1f}</div>
                <div style="font-size: 11px; color: #8091b8; margin-top: 4px;">
                    {ind['rsi_label']}
                </div>
            </div>
            <div class="metric-box">
                <div class="metric-box-label">MACD Histogram</div>
                <div class="metric-box-value">{ind['macd_hist']:.2f}</div>
                <div style="font-size: 11px; color: #8091b8; margin-top: 4px;">
                    {ind['macd_label']}
                </div>
            </div>
            <div class="metric-box">
                <div class="metric-box-label">Bollinger %B</div>
                <div class="metric-box-value">{ind['bb_pct_b']:.1f}</div>
                <div style="font-size: 11px; color: #8091b8; margin-top: 4px;">
                    {ind['bb_label']}
                </div>
            </div>
            <div class="metric-box">
                <div class="metric-box-label">ATR</div>
                <div class="metric-box-value">{ind['atr']:.0f} ({ind['atr_pct']:.2f}%)</div>
                <div style="font-size: 11px; color: #8091b8; margin-top: 4px;">
                    {ind['atr_label']}
                </div>
            </div>
            <div class="metric-box">
                <div class="metric-box-label">SMA 20</div>
                <div class="metric-box-value">{ind['sma20']:,.0f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-label">EMA 20</div>
                <div class="metric-box-value">{ind['ema20']:,.0f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Support & Resistance
    if ind['supports']:
        sup_html = "".join([
            f'<div style="display:flex; justify-content:space-between; padding:8px 0; '
            f'border-bottom: 1px solid rgba(255,255,255,0.04);">'
            f'<span style="color: #ffffff; font-size: 14px;">{s:,.0f}</span>'
            f'<span style="font-size: 10px; font-weight:700; color: #00d47a; '
            f'background: rgba(0,212,122,0.15); padding: 2px 8px; border-radius: 4px;">'
            f'{strength}</span></div>'
            for s, strength in ind['supports']
        ])
    else:
        sup_html = '<div style="color:#5a6fa5; font-size:12px; padding: 8px;">Tidak ada support terdeteksi</div>'

    if ind['resistances']:
        res_html = "".join([
            f'<div style="display:flex; justify-content:space-between; padding:8px 0; '
            f'border-bottom: 1px solid rgba(255,255,255,0.04);">'
            f'<span style="color: #ffffff; font-size: 14px;">{r:,.0f}</span>'
            f'<span style="font-size: 10px; font-weight:700; color: #ff5252; '
            f'background: rgba(255,82,82,0.15); padding: 2px 8px; border-radius: 4px;">'
            f'{strength}</span></div>'
            for r, strength in ind['resistances']
        ])
    else:
        res_html = '<div style="color:#5a6fa5; font-size:12px; padding: 8px;">Tidak ada resistance terdeteksi</div>'

    st.markdown(f"""
    <div class="section-card">
        <div class="metric-grid">
            <div style="background: #0f1530; border-radius: 10px; padding: 16px;
                        border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 12px; color: #8091b8; margin-bottom: 12px;">
                    Support
                </div>
                {sup_html}
            </div>
            <div style="background: #0f1530; border-radius: 10px; padding: 16px;
                        border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 12px; color: #8091b8; margin-bottom: 12px;">
                    Resistance
                </div>
                {res_html}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Data OHLCV tidak cukup untuk hitung indikator teknikal.")


# ============================================================
# SECTION 7: RISK / REWARD
# ============================================================
with st.spinner("Hitung risk/reward..."):
    rr = hitung_risk_reward(kode_pilih, ind) if ind else None

if rr:
    rr_ratio = rr['rr_ratio']
    if rr_ratio >= 2:
        rr_color = "#00d47a"
    elif rr_ratio >= 1.5:
        rr_color = "#ffa726"
    else:
        rr_color = "#ff5252"

    targets_html = "".join([
        f'<div style="display: flex; justify-content: space-between; '
        f'padding: 12px 16px; background: #0f1530; border-radius: 8px; '
        f'margin-bottom: 8px; border: 1px solid rgba(255,255,255,0.05);">'
        f'<span style="font-size: 16px; font-weight: 700; color: #ffffff;">'
        f'{t["harga"]:,.0f}</span>'
        f'<span style="font-size: 12px; color: #8091b8; align-self: center;">'
        f'Probabilitas {t["prob"]}%</span>'
        f'<span style="font-size: 14px; font-weight: 700; color: #00d47a;">'
        f'+{t["gain_pct"]:.1f}%</span>'
        f'</div>'
        for t in rr['targets']
    ])

    st.markdown(f"""
    <div class="section-card">
        <div style="display: flex; justify-content: space-between;
                    align-items: center; margin-bottom: 16px;">
            <div>
                <div class="section-title">Risk/Reward</div>
                <div class="section-subtitle" style="margin-bottom: 0;">
                    Entry {rr['entry']:,.0f}
                </div>
            </div>
            <span style="font-size: 11px; font-weight: 700; padding: 4px 12px;
                         border-radius: 6px; background: {rr_color}20;
                         color: {rr_color}; border: 1px solid {rr_color}40;">
                {rr['setup']}
            </span>
        </div>

        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-box-label">Stop Loss</div>
                <div class="metric-box-value" style="color: #ff5252;">
                    {rr['stop_loss']:,.0f}
                </div>
            </div>
            <div class="metric-box">
                <div class="metric-box-label">Risk/Reward</div>
                <div class="metric-box-value">1 : {rr_ratio:.2f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-label">Lot Disarankan</div>
                <div class="metric-box-value">{rr['lot_disarankan']:,}</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-label">Investasi</div>
                <div class="metric-box-value small">Rp {rr['investasi']/1e6:,.1f} M</div>
            </div>
        </div>

        <div style="margin-top: 20px;">
            <div style="font-size: 12px; color: #8091b8; margin-bottom: 12px;">
                Target Harga
            </div>
            {targets_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SECTION 8: SINYAL TRADING
# ============================================================
with st.spinner("Hitung sinyal trading..."):
    sig = hitung_sinyal_trading(kode_pilih)

if sig:
    if sig['sinyal'] == 'BUY':
        sig_color, sig_icon = "#00d47a", "↗"
    elif sig['sinyal'] == 'SELL':
        sig_color, sig_icon = "#ff5252", "↘"
    else:
        sig_color, sig_icon = "#ffa726", "—"

    factors_html = "".join([
        f'<span style="display: inline-block; font-size: 11px; '
        f'padding: 4px 10px; border-radius: 4px; margin: 4px 4px 4px 0; '
        f'background: {"rgba(0,212,122,0.15)" if f[1]=="BULLISH" else "rgba(255,82,82,0.15)" if f[1]=="BEARISH" else "rgba(255,167,38,0.15)"}; '
        f'color: {"#00d47a" if f[1]=="BULLISH" else "#ff5252" if f[1]=="BEARISH" else "#ffa726"};">'
        f'{f[0]}</span>'
        for f in sig['factors']
    ])

    def _trend_row(label, trend):
        if trend == "BULLISH":
            color, icon = "#00d47a", "↗"
        elif trend == "BEARISH":
            color, icon = "#ff5252", "↘"
        else:
            color, icon = "#ffa726", "—"
        return f"""
        <div style="display: flex; align-items: center; gap: 14px;
                    padding: 14px 16px; background: #0f1530; border-radius: 10px;
                    margin-bottom: 8px; border: 1px solid rgba(255,255,255,0.05);">
            <div style="font-size: 20px; color: {color}; width: 30px;
                        text-align: center;">{icon}</div>
            <div style="flex-grow: 1;">
                <div style="font-size: 10px; color: #8091b8; letter-spacing: 0.8px;">
                    {label}
                </div>
                <div style="font-size: 15px; font-weight: 700; color: {color};">
                    {trend}
                </div>
            </div>
        </div>
        """

    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">Sinyal Trading</div>
        <div class="section-subtitle">Timeframe daily · 100 candle terakhir</div>

        <div style="display: flex; align-items: center; gap: 14px;
                    padding: 16px 20px; background: #0f1530; border-radius: 10px;
                    margin-bottom: 16px; border-left: 4px solid {sig_color};">
            <div style="font-size: 32px; color: {sig_color}; font-weight: 700;">
                {sig_icon}
            </div>
            <div style="flex-grow: 1;">
                <div style="font-size: 20px; font-weight: 700; color: {sig_color};">
                    {sig['sinyal']}
                </div>
                <div style="font-size: 12px; color: #8091b8; margin-top: 2px;">
                    Confidence {sig['confidence']}%
                </div>
            </div>
        </div>

        <div style="font-size: 13px; color: #a0aec8; line-height: 1.5;
                    margin-bottom: 16px;">
            {sig['deskripsi']}
        </div>

        <div style="margin-bottom: 16px;">
            {factors_html}
        </div>

        {_trend_row("JANGKA PENDEK", sig['trend_short'])}
        {_trend_row("JANGKA MENENGAH", sig['trend_mid'])}
        {_trend_row("JANGKA PANJANG", sig['trend_long'])}
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown(f"""
<div style="text-align: center; color: #5a6fa5; font-size: 11px;
            margin-top: 20px; padding-top: 16px; border-top: 1px solid #1e2952;">
    Sumber: yfinance + data lokal · Update: {data.get('update', '')}
</div>
""", unsafe_allow_html=True)