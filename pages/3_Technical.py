"""Halaman TECHNICAL — premium."""

import os, sys
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant.technical import hitung_indikator_teknikal, hitung_risk_reward, hitung_sinyal_trading
from quant.theme import load_theme, page_header, section_title, empty_state

st.set_page_config(page_title="Technical", page_icon="📉", layout="wide")
load_theme()

OHLCV_DIR = "data_ohlcv_idx"


@st.cache_data(ttl=1800)
def list_available():
    if not os.path.exists(OHLCV_DIR): return []
    return sorted(f.replace(".csv", "") for f in os.listdir(OHLCV_DIR) if f.endswith(".csv"))


@st.cache_data(ttl=1800)
def load_ohlcv(kode):
    path = os.path.join(OHLCV_DIR, f"{kode}.csv")
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").tail(250).reset_index(drop=True)


# HEADER
st.markdown(page_header("📉", "Technical", "Indikator & Sinyal Trading"), unsafe_allow_html=True)

saham_list = list_available()
if not saham_list:
    st.markdown(empty_state("📭", "Belum ada data", "Jalankan scraper OHLCV dulu"), unsafe_allow_html=True)
    st.stop()

if "technical_kode" not in st.session_state:
    st.session_state.technical_kode = "BBCA" if "BBCA" in saham_list else saham_list[0]

quick = ["BBCA", "BBRI", "TLKM", "ASII", "GOTO", "ANTM"]
cols = st.columns(len(quick) + 1)
for i, q in enumerate(quick):
    if q in saham_list:
        with cols[i]:
            if st.button(q, key=f"tech_chip_{q}", use_container_width=True):
                st.session_state.technical_kode = q
                st.rerun()

kode_pilih = st.selectbox("Pilih Saham", saham_list,
    index=saham_list.index(st.session_state.technical_kode) if st.session_state.technical_kode in saham_list else 0,
    label_visibility="collapsed")
st.session_state.technical_kode = kode_pilih

df = load_ohlcv(kode_pilih)
if df is None or len(df) < 30:
    st.error(f"Data {kode_pilih} tidak cukup.")
    st.stop()

harga = float(df["Close"].iloc[-1])
prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else harga
chg = (harga / prev - 1) * 100 if prev else 0
chg_color = "#00d47a" if chg >= 0 else "#ff5252"
chg_sign = "+" if chg >= 0 else ""

ind = hitung_indikator_teknikal(kode_pilih)

# Header saham
st.markdown(f"""
<div style="display: flex; align-items: center; gap: 16px; margin-bottom: 20px;">
    <div style="width: 56px; height: 56px;
                background: linear-gradient(135deg, #8b5cf6, #a78bfa);
                border-radius: 14px; display: flex; align-items: center;
                justify-content: center; font-size: 24px; font-weight: 700;
                color: white; box-shadow: 0 8px 20px rgba(139,92,246,0.3);">
        {kode_pilih[:1]}</div>
    <div style="flex-grow: 1;">
        <div style="font-size: 13px; color: #8091b8; margin-bottom: 4px;">
            Technical Analysis · 250 hari</div>
        <div style="font-size: 20px; font-weight: 700; color: #ffffff; margin-bottom: 6px;">
            {kode_pilih}</div>
        <div style="font-size: 24px; font-weight: 800; color: #ffffff;">
            {harga:,.0f}
            <span style="font-size: 14px; font-weight: 600; padding: 3px 10px;
                        border-radius: 6px; margin-left: 8px;
                        background: {chg_color}20; color: {chg_color};">
                {chg_sign}{chg:.2f}%</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Indikator
if ind:
    st.markdown(section_title("Indikator Teknikal"), unsafe_allow_html=True)

    ind_items = [
        ("RSI (14)", f"{ind['rsi14']:.1f}", ind['rsi_label']),
        ("MACD Histogram", f"{ind['macd_hist']:.2f}", ind['macd_label']),
        ("Bollinger %B", f"{ind['bb_pct_b']:.1f}", ind['bb_label']),
        ("ATR", f"{ind['atr']:.0f}", f"{ind['atr_pct']:.2f}% · {ind['atr_label']}"),
        ("SMA 20", f"{ind['sma20']:,.0f}", ""),
        ("EMA 20", f"{ind['ema20']:,.0f}", ""),
    ]

    cols = st.columns(3)
    for i, (label, val, sub) in enumerate(ind_items):
        with cols[i % 3]:
            sub_html = f'<div style="font-size: 10px; color: #8091b8; margin-top: 4px;">{sub}</div>' if sub else ''
            st.markdown(f"""
            <div style="background: #161c2e; border-radius: 10px; padding: 14px 16px;
                        margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.06);">
                <div style="font-size: 11px; color: #8091b8;">{label}</div>
                <div style="font-size: 20px; font-weight: 700; color: #ffffff;">{val}</div>
                {sub_html}
            </div>""", unsafe_allow_html=True)

    # Support & Resistance
    st.markdown(section_title("Support & Resistance"), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        sup = "".join([f'<div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.04);"><span style="color:#fff; font-weight:600;">{s:,.0f}</span><span style="font-size:10px; padding:2px 8px; border-radius:4px; background:rgba(0,212,122,0.15); color:#00d47a;">{st_}</span></div>' for s, st_ in ind['supports']]) or '<div style="color:#5a6fa5; font-size:12px;">—</div>'
        st.markdown(f"""<div style="background:#161c2e; border-radius:10px; padding:16px;
                    border:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:12px; color:#8091b8; margin-bottom:12px;">SUPPORT</div>
            {sup}</div>""", unsafe_allow_html=True)
    with c2:
        res = "".join([f'<div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.04);"><span style="color:#fff; font-weight:600;">{r:,.0f}</span><span style="font-size:10px; padding:2px 8px; border-radius:4px; background:rgba(255,82,82,0.15); color:#ff5252;">{st_}</span></div>' for r, st_ in ind['resistances']]) or '<div style="color:#5a6fa5; font-size:12px;">—</div>'
        st.markdown(f"""<div style="background:#161c2e; border-radius:10px; padding:16px;
                    border:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:12px; color:#8091b8; margin-bottom:12px;">RESISTANCE</div>
            {res}</div>""", unsafe_allow_html=True)

# Risk/Reward
rr = hitung_risk_reward(kode_pilih, ind) if ind else None

if rr:
    st.markdown(section_title("Risk/Reward"), unsafe_allow_html=True)
    rr_ratio = rr['rr_ratio']
    rr_color = "#00d47a" if rr_ratio >= 2 else ("#ffa726" if rr_ratio >= 1.5 else "#ff5252")

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, color in [
        (c1, "ENTRY", f"{rr['entry']:,.0f}", "#ffffff"),
        (c2, "STOP LOSS", f"{rr['stop_loss']:,.0f}", "#ff5252"),
        (c3, "R:R", f"1 : {rr_ratio:.2f}", rr_color),
        (c4, "LOT", f"{rr['lot_disarankan']:,}", "#3b82f6"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:#161c2e; border-radius:10px; padding:14px 16px;
                        margin-bottom:12px; border:1px solid rgba(255,255,255,0.06);">
                <div style="font-size:11px; color:#8091b8;">{label}</div>
                <div style="font-size:20px; font-weight:700; color:{color};">{val}</div>
            </div>""", unsafe_allow_html=True)

# Sinyal
sig = hitung_sinyal_trading(kode_pilih)
if sig:
    st.markdown(section_title("Sinyal Trading"), unsafe_allow_html=True)
    sig_color = "#00d47a" if sig['sinyal'] == 'BUY' else ("#ff5252" if sig['sinyal'] == 'SELL' else "#ffa726")
    sig_icon = "↗" if sig['sinyal'] == 'BUY' else ("↘" if sig['sinyal'] == 'SELL' else "—")

    st.markdown(f"""
    <div style="background:#161c2e; border-radius:12px; padding:20px 24px;
                border:1px solid rgba(255,255,255,0.06); border-left:4px solid {sig_color};">
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:16px;">
            <div style="font-size:36px; color:{sig_color}; font-weight:700;">{sig_icon}</div>
            <div>
                <div style="font-size:22px; font-weight:700; color:{sig_color};">{sig['sinyal']}</div>
                <div style="font-size:12px; color:#8091b8;">Confidence {sig['confidence']}%</div>
            </div>
        </div>
        <div style="font-size:13px; color:#a0aec8; line-height:1.5;">{sig['deskripsi']}</div>
    </div>""", unsafe_allow_html=True)

    # Multi-timeframe
    c1, c2, c3 = st.columns(3)
    for col, label, trend in [
        (c1, "JANGKA PENDEK", sig['trend_short']),
        (c2, "JANGKA MENENGAH", sig['trend_mid']),
        (c3, "JANGKA PANJANG", sig['trend_long']),
    ]:
        color = "#00d47a" if trend == "BULLISH" else ("#ff5252" if trend == "BEARISH" else "#ffa726")
        with col:
            st.markdown(f"""
            <div style="background:#161c2e; border-radius:10px; padding:16px;
                        border:1px solid rgba(255,255,255,0.06); text-align:center;">
                <div style="font-size:10px; color:#8091b8; letter-spacing:0.8px;">{label}</div>
                <div style="font-size:15px; font-weight:700; color:{color}; margin-top:6px;">{trend}</div>
            </div>""", unsafe_allow_html=True)