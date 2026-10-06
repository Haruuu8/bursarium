"""Halaman WATCHLIST — premium, layout vertikal, accent bar cyan."""

import os, sys, json
import pandas as pd
import streamlit as st
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant.theme import (
    load_theme, page_header, section_title, empty_state,
)

st.set_page_config(page_title="Watchlist", page_icon="⭐", layout="wide")
load_theme()

WATCHLIST_FILE = "data_watchlist.json"
OUTPUT_DIR = "quant/output"
OHLCV_DIR = "data_ohlcv_idx"
DETAIL_DIR = "data_fundamental_detail"
SEKTOR_CSV = "data_sektor.csv"


# ============================================================
# PERSISTENCE
# ============================================================
def load_watchlist():
    if not os.path.exists(WATCHLIST_FILE): return {}
    try:
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_watchlist(data):
    with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def remove_from_watchlist(kode):
    wl = load_watchlist()
    if kode in wl:
        del wl[kode]
        save_watchlist(wl)


# ============================================================
# HELPERS
# ============================================================
@st.cache_data(ttl=3600)
def load_nama():
    nama = {}
    if os.path.exists(SEKTOR_CSV):
        try:
            df = pd.read_csv(SEKTOR_CSV)
            if "kode" in df.columns and "nama" in df.columns:
                for _, r in df.iterrows():
                    k = str(r["kode"]).strip().upper()
                    if k:
                        nama[k] = str(r["nama"]).strip()
        except Exception:
            pass
    return nama


def get_prices(kode):
    path = os.path.join(OHLCV_DIR, f"{kode}.csv")
    if not os.path.exists(path): return None, None
    try:
        df = pd.read_csv(path)
        if len(df) < 2: return None, None
        return float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
    except Exception:
        return None, None


def list_files(prefix):
    if not os.path.exists(OUTPUT_DIR): return []
    fs = [f for f in os.listdir(OUTPUT_DIR)
          if f.startswith(prefix) and f.endswith(".csv")]
    fs.sort(reverse=True)
    return fs


@st.cache_data(ttl=300)
def load_csv(path):
    try: return pd.read_csv(path)
    except Exception: return pd.DataFrame()


def wl_row(kode, nama, harga, chg_pct, rating=0, match="-", strength=0,
           return_entry=None):
    """Render 1 baris watchlist. Accent bar kiri 1 warna (cyan)."""
    nama_disp = nama[:50] + "…" if len(nama) > 50 else nama

    if chg_pct > 0:
        chg_color = "#10b981"; chg_sign = "+"
    elif chg_pct < 0:
        chg_color = "#f43f5e"; chg_sign = ""
    else:
        chg_color = "#94a3b8"; chg_sign = ""

    badges = ""
    if rating > 0:
        badges += (
            '<span style="display:inline-flex; align-items:center; '
            'font-size:11px; font-weight:800; padding:4px 10px; '
            'border-radius:6px; letter-spacing:0.3px; margin-right:6px; '
            'background:rgba(99,102,241,0.15); color:#a5b4fc; '
            'border:1px solid rgba(99,102,241,0.35);">'
            f'★ {rating:.0f}</span>'
        )
    if match and match != "-":
        badges += (
            '<span style="display:inline-block; '
            'font-size:11px; font-weight:800; padding:4px 10px; '
            'border-radius:6px; letter-spacing:0.3px; margin-right:6px; '
            'background:rgba(168,85,247,0.15); color:#c4b5fd; '
            'border:1px solid rgba(168,85,247,0.35);">'
            f'{match}</span>'
        )
    if strength > 0:
        badges += (
            '<span style="display:inline-block; '
            'font-size:11px; font-weight:800; padding:4px 10px; '
            'border-radius:6px; letter-spacing:0.3px; '
            'background:rgba(245,158,11,0.15); color:#fbbf24; '
            'border:1px solid rgba(245,158,11,0.35);">'
            f'S{strength}</span>'
        )

    entry_html = ""
    if return_entry is not None:
        re_color = "#10b981" if return_entry >= 0 else "#f43f5e"
        re_sign = "+" if return_entry >= 0 else ""
        entry_html = (
            f'<div style="font-size:12px; color:{re_color}; '
            f'font-weight:700; margin-top:4px;">'
            f'{re_sign}{return_entry:.2f}% entry</div>'
        )

    return (
        '<div style="'
        'background: linear-gradient(135deg, #131a2a 0%, #0f1628 100%); '
        'border: 1px solid #1a2338; border-radius: 14px; '
        'padding: 18px 22px; margin-bottom: 10px; '
        'display: flex; align-items: center; gap: 20px; '
        'box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25); '
        'position: relative; overflow: hidden;">'
        '<div style="position:absolute; left:0; top:0; bottom:0; width:3px; '
        'background: linear-gradient(180deg, #06b6d4, rgba(6,182,212,0.3));"></div>'
        '<div style="flex: 1; min-width: 0; padding-left: 8px;">'
        f'<div style="font-size:17px; font-weight:800; color:#ffffff; '
        f'letter-spacing:-0.02em; line-height:1.2;">{kode}</div>'
        f'<div style="font-size:12px; color:#7d8ba3; margin-top:3px; '
        f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
        f'{nama_disp}</div>'
        f'<div style="margin-top:10px; display:flex; flex-wrap:wrap;">'
        f'{badges}</div>'
        '</div>'
        '<div style="text-align:right; flex-shrink:0; min-width:110px;">'
        f'<div style="font-size:22px; font-weight:800; color:#ffffff; '
        f'letter-spacing:-0.03em; line-height:1;">{harga:,.0f}</div>'
        f'<div style="font-size:13px; font-weight:700; color:{chg_color}; '
        f'margin-top:5px;">{chg_sign}{chg_pct:.2f}%</div>'
        f'{entry_html}'
        '</div>'
        '</div>'
    )


# ============================================================
# HEADER
# ============================================================
st.markdown(page_header("⭐", "Watchlist", "Kandidat & Tracking"),
            unsafe_allow_html=True)

nama_map = load_nama()

tab1, tab2, tab3 = st.tabs(["🎯 Kandidat", "⭐ Watchlist Saya", "📊 Performa"])


# ============================================================
# TAB 1: KANDIDAT
# ============================================================
with tab1:
    files = list_files("kandidat_rated_")
    if not files:
        files = list_files("kandidat_")

    if not files:
        st.markdown(empty_state("📭", "Belum ada kandidat",
                                "Jalankan screener dulu"), unsafe_allow_html=True)
    else:
        c1, c2, c3 = st.columns([3, 2, 2])
        with c1:
            selected = st.selectbox("File", files, label_visibility="collapsed")
        with c2:
            min_str = st.selectbox("Min Strength", [1, 2, 3, 4], index=1,
                                    label_visibility="collapsed")
        with c3:
            min_rating = st.slider("Min Rating", 0, 100, 0, 5,
                                    label_visibility="collapsed")

        df = load_csv(os.path.join(OUTPUT_DIR, selected))

        if df.empty:
            st.error("File kosong.")
        else:
            if "strength" in df.columns and "rating" in df.columns:
                df = df[(df["strength"] >= min_str) &
                        (df["rating"] >= min_rating)]

            st.markdown(
                f'<div style="font-size:12px; color:#7d8ba3; '
                f'margin:8px 0 16px; font-weight:500;">'
                f'{len(df)} kandidat ditampilkan</div>',
                unsafe_allow_html=True
            )

            wl = load_watchlist()

            for idx, r in df.iterrows():
                kode = str(r["kode"]).upper()
                nama = nama_map.get(kode, kode)
                close = float(r["close"])
                rating = float(r.get("rating", 0))
                match = str(r.get("match", "-"))
                strength = int(r.get("strength", 0))
                ret1d = float(r.get("ret_1d", 0)) * 100

                in_wl = kode in wl

                c_left, c_btn = st.columns([12, 1])

                with c_left:
                    st.markdown(
                        wl_row(kode, nama, close, ret1d, rating, match, strength),
                        unsafe_allow_html=True
                    )

                with c_btn:
                    st.markdown('<div style="height: 26px;"></div>',
                                unsafe_allow_html=True)
                    if in_wl:
                        st.markdown(
                            '<div style="text-align:center; font-size:22px; '
                            'padding-top:4px;">✅</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        if st.button("＋", key=f"add_{kode}_{idx}",
                                     use_container_width=True, help="Tambah"):
                            wl[kode] = {
                                "tanggal_tambah": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                "harga_entry": close, "match": match,
                                "strength": strength, "rating": rating,
                                "hist_wr": float(r.get("hist_wr", 0)),
                                "hist_n": int(r.get("hist_n", 0)),
                            }
                            save_watchlist(wl)
                            st.rerun()


# ============================================================
# TAB 2: WATCHLIST SAYA
# ============================================================
with tab2:
    wl = load_watchlist()

    if not wl:
        st.markdown(empty_state("📭", "Watchlist kosong",
                                "Tambah dari tab Kandidat"), unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div style="font-size:12px; color:#7d8ba3; '
            f'margin-bottom:16px; font-weight:500;">'
            f'{len(wl)} saham di watchlist</div>',
            unsafe_allow_html=True
        )

        items = []
        for kode, info in wl.items():
            entry = info.get("harga_entry", 0)
            now, prev = get_prices(kode)
            now = now or entry
            chg = (now / prev - 1) * 100 if prev else 0
            ret = (now / entry - 1) * 100 if entry else 0
            items.append({
                "kode": kode, "nama": nama_map.get(kode, ""),
                "now": now, "chg": chg, "ret": ret,
                "rating": info.get("rating", 0),
                "match": info.get("match", "-"),
                "strength": info.get("strength", 0),
            })

        df = pd.DataFrame(items).sort_values("ret", ascending=False).reset_index(drop=True)

        for idx, r in df.iterrows():
            kode = r["kode"]
            c_left, c_btn = st.columns([12, 1])

            with c_left:
                st.markdown(
                    wl_row(kode, r["nama"], r["now"], r["chg"],
                           r["rating"], r["match"], r["strength"],
                           return_entry=r["ret"]),
                    unsafe_allow_html=True
                )

            with c_btn:
                st.markdown('<div style="height: 26px;"></div>',
                            unsafe_allow_html=True)
                if st.button("🗑", key=f"del_{kode}",
                             use_container_width=True, help="Hapus"):
                    remove_from_watchlist(kode)
                    st.rerun()

        avg = df["ret"].mean()
        ac = "#10b981" if avg >= 0 else "#f43f5e"
        st.markdown(
            '<div style="display:flex; justify-content:space-between; '
            'margin-top:24px; padding:18px 22px; '
            'background:linear-gradient(135deg, #131a2a, #0f1628); '
            'border-radius:14px; border:1px solid #1a2338;">'
            '<span style="font-size:13px; color:#7d8ba3; font-weight:600;">'
            'Rata-rata return</span>'
            f'<span style="font-size:18px; font-weight:800; color:{ac};">'
            f'{avg:+.2f}%</span>'
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# TAB 3: PERFORMA
# ============================================================
with tab3:
    wl = load_watchlist()
    if not wl:
        st.markdown(empty_state("📊", "Belum ada performa",
                                "Tambah saham dulu"), unsafe_allow_html=True)
    else:
        items = []
        for kode, info in wl.items():
            entry = info.get("harga_entry", 0)
            now, _ = get_prices(kode)
            if now is None: continue
            items.append({
                "kode": kode,
                "ret": (now / entry - 1) * 100 if entry else 0
            })

        if not items:
            st.info("Data harga tidak tersedia.")
        else:
            df = pd.DataFrame(items)
            total = len(df)
            win = int((df["ret"] > 0).sum())
            wr = win / total * 100 if total else 0
            avg = df["ret"].mean()
            tot = df["ret"].sum()

            c1, c2, c3 = st.columns(3)
            for col, label, val, color in [
                (c1, "WIN RATE", f"{wr:.1f}%",
                 "#10b981" if wr >= 50 else "#f43f5e"),
                (c2, "AVG RETURN", f"{avg:+.2f}%",
                 "#10b981" if avg >= 0 else "#f43f5e"),
                (c3, "TOTAL RETURN", f"{tot:+.2f}%",
                 "#10b981" if tot >= 0 else "#f43f5e"),
            ]:
                with col:
                    st.markdown(
                        '<div style="background:linear-gradient(180deg, #131a2a, #0f1628); '
                        'border-radius:14px; padding:22px 24px; '
                        'border:1px solid #1a2338; '
                        'box-shadow:0 4px 16px rgba(0,0,0,0.25);">'
                        f'<div style="font-size:11px; color:#7d8ba3; '
                        f'letter-spacing:0.8px; font-weight:600;">{label}</div>'
                        f'<div style="font-size:30px; font-weight:800; '
                        f'color:{color}; margin-top:8px; '
                        f'letter-spacing:-0.03em;">{val}</div>'
                        '</div>',
                        unsafe_allow_html=True
                    )

            st.markdown("<br>", unsafe_allow_html=True)
            best = df.loc[df["ret"].idxmax()]
            worst = df.loc[df["ret"].idxmin()]

            c4, c5 = st.columns(2)
            with c4:
                st.markdown(
                    '<div style="background:linear-gradient(135deg, rgba(16,185,129,0.12), rgba(16,185,129,0.03)); '
                    'border-radius:14px; padding:20px 22px; '
                    'border:1px solid rgba(16,185,129,0.3);">'
                    '<div style="font-size:11px; color:#7d8ba3; '
                    'letter-spacing:1px; font-weight:700;">🏆 BEST</div>'
                    f'<div style="font-size:24px; font-weight:800; color:#10b981; '
                    f'margin-top:8px;">{best["kode"]}</div>'
                    f'<div style="font-size:15px; color:#10b981; '
                    f'margin-top:4px; font-weight:700;">{best["ret"]:+.2f}%</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            with c5:
                st.markdown(
                    '<div style="background:linear-gradient(135deg, rgba(244,63,94,0.12), rgba(244,63,94,0.03)); '
                    'border-radius:14px; padding:20px 22px; '
                    'border:1px solid rgba(244,63,94,0.3);">'
                    '<div style="font-size:11px; color:#7d8ba3; '
                    'letter-spacing:1px; font-weight:700;">📉 WORST</div>'
                    f'<div style="font-size:24px; font-weight:800; color:#f43f5e; '
                    f'margin-top:8px;">{worst["kode"]}</div>'
                    f'<div style="font-size:15px; color:#f43f5e; '
                    f'margin-top:4px; font-weight:700;">{worst["ret"]:+.2f}%</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

            st.markdown(section_title("Ranking Return"), unsafe_allow_html=True)
            ranking = df.sort_values("ret", ascending=False).reset_index(drop=True)
            for i, r in ranking.iterrows():
                rc = "#10b981" if r["ret"] >= 0 else "#f43f5e"
                st.markdown(
                    '<div style="display:flex; justify-content:space-between; '
                    'padding:12px 18px; '
                    'background:linear-gradient(135deg, #131a2a, #0f1628); '
                    'border-radius:10px; margin-bottom:6px; '
                    'border:1px solid #1a2338;">'
                    f'<span style="font-size:14px; color:#e6ebf5; font-weight:700;">'
                    f'{i+1}. {r["kode"]}</span>'
                    f'<span style="font-size:14px; font-weight:800; color:{rc};">'
                    f'{r["ret"]:+.2f}%</span>'
                    '</div>',
                    unsafe_allow_html=True
                )