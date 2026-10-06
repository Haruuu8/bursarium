"""Theme minimalis premium — tanpa animasi, tanpa Google Fonts."""

import streamlit as st


THEME_CSS = """
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.stApp {
    background: #070b16;
    color: #e6ebf5;
}
[data-testid="stAppViewContainer"] { background: #070b16; }
[data-testid="stHeader"] { background: transparent !important; }

.block-container {
    padding-top: 3.5rem !important;
    padding-bottom: 4rem !important;
    max-width: 1150px !important;
}

/* ============================================ */
/* SIDEBAR — hanya warna                         */
/* ============================================ */
[data-testid="stSidebar"] {
    background: #0a0f1e !important;
    border-right: 1px solid #16203a !important;
}
[data-testid="stSidebarContent"] { background: transparent !important; }
[data-testid="stSidebarNav"] a {
    padding: 11px 14px !important;
    border-radius: 10px !important;
    color: #7d8ba3 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    margin: 3px 8px !important;
    border: 1px solid transparent !important;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(99, 102, 241, 0.08) !important;
    color: #e6ebf5 !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: rgba(99, 102, 241, 0.15) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-color: rgba(99, 102, 241, 0.4) !important;
}

/* ============================================ */
/* TYPOGRAPHY                                    */
/* ============================================ */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
}

/* ============================================ */
/* BUTTONS                                       */
/* ============================================ */
.stButton > button {
    background: #1a2338 !important;
    border: 1px solid #24304d !important;
    color: #cbd5e1 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 9px 16px !important;
}
.stButton > button:hover {
    background: #24304d !important;
    border-color: #6366f1 !important;
    color: #ffffff !important;
}

/* ============================================ */
/* INPUTS                                        */
/* ============================================ */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stMultiSelect > div > div {
    background: #131a2a !important;
    border: 1px solid #24304d !important;
    border-radius: 10px !important;
    color: #e6ebf5 !important;
}
.stSelectbox > div > div:focus-within,
.stTextInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

/* ============================================ */
/* TABS — cyan solid dengan fade di ujung        */
/* ============================================ */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid #1a2338 !important;
    padding: 0 !important;
    margin-bottom: 24px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 0 !important;
    color: #7d8ba3 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 10px 16px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    position: relative !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #e6ebf5 !important;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #06b6d4 !important;
    border-bottom: none !important;
    box-shadow: none !important;
}
.stTabs [aria-selected="true"]::after {
    content: "" !important;
    position: absolute !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    height: 2px !important;
    background: linear-gradient(90deg,
        rgba(6, 182, 212, 0.15) 0%,
        #06b6d4 50%,
        rgba(6, 182, 212, 0.15) 100%) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
    height: 0 !important;
    width: 0 !important;
    opacity: 0 !important;
}

/* ============================================ */
/* SCROLLBAR                                     */
/* ============================================ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #070b16; }
::-webkit-scrollbar-thumb { background: #24304d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #344466; }

/* ============================================ */
/* HIDE                                          */
/* ============================================ */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
</style>
"""


def load_theme():
    """Panggil di awal setiap halaman."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


# ============================================================
# KOMPONEN
# ============================================================
def hero(title, subtitle, badge_text="", badge_color="#06b6d4"):
    """Hero section besar."""
    badge_html = ""
    if badge_text:
        badge_html = (
            f'<div style="display: inline-block; padding: 6px 14px; '
            f'background: rgba(99, 102, 241, 0.15); color: {badge_color}; '
            f'border: 1px solid rgba(99, 102, 241, 0.3); '
            f'border-radius: 20px; font-size: 11px; font-weight: 700; '
            f'letter-spacing: 1.2px; margin-bottom: 14px;">'
            f'● {badge_text}</div>'
        )
    return (
        '<div style="'
        'background: linear-gradient(135deg, #0f1628 0%, #0a0f1e 100%); '
        'border: 1px solid #1a2338; border-radius: 20px; '
        'padding: 36px 40px; margin-bottom: 28px;">'
        '<div style="position: relative; z-index: 1;">'
        f'{badge_html}'
        f'<div style="font-size: 36px; font-weight: 900; color: #ffffff; '
        f'letter-spacing: -0.04em; line-height: 1.05; margin-bottom: 10px;">'
        f'{title}</div>'
        f'<div style="font-size: 14px; color: #94a3b8; line-height: 1.5; '
        f'max-width: 600px;">{subtitle}</div>'
        '</div></div>'
    )


def page_header(icon, title, subtitle=""):
    """Header halaman standar."""
    return (
        '<div style="'
        'background: linear-gradient(135deg, #0f1628 0%, #0a0f1e 100%); '
        'border: 1px solid #1a2338; border-radius: 16px; '
        'padding: 24px 28px; margin-bottom: 24px; '
        'display: flex; align-items: center; gap: 18px;">'
        f'<div style="width: 56px; height: 56px; '
        f'background: linear-gradient(135deg, #6366f1, #06b6d4); '
        f'border-radius: 14px; display: flex; align-items: center; '
        f'justify-content: center; font-size: 26px; '
        f'box-shadow: 0 8px 24px rgba(99, 102, 241, 0.3);">{icon}</div>'
        '<div>'
        f'<div style="font-size: 11px; color: #7d8ba3; letter-spacing: 1.3px; '
        f'font-weight: 700; margin-bottom: 5px; text-transform: uppercase;">'
        f'{subtitle}</div>'
        f'<div style="font-size: 26px; font-weight: 800; color: #ffffff; '
        f'letter-spacing: -0.03em; line-height: 1;">{title}</div>'
        '</div></div>'
    )


def section_title(title, subtitle=""):
    """Section title dengan accent bar."""
    sub_html = (
        f'<div style="font-size: 12px; color: #7d8ba3; margin-left: 17px; '
        f'margin-top: 4px;">{subtitle}</div>'
    ) if subtitle else ''
    return (
        '<div style="margin: 32px 0 18px;">'
        '<div style="display: flex; align-items: center; gap: 13px;">'
        '<div style="width: 4px; height: 22px; '
        'background: linear-gradient(180deg, #6366f1, #06b6d4); '
        'border-radius: 2px;"></div>'
        f'<div style="font-size: 18px; font-weight: 800; color: #ffffff; '
        f'letter-spacing: -0.02em;">{title}</div>'
        '</div>'
        f'{sub_html}'
        '</div>'
    )


def card(content, padding="20px 24px"):
    """Card container."""
    return (
        '<div style="'
        'background: linear-gradient(180deg, #131a2a 0%, #0f1628 100%); '
        'border: 1px solid #1a2338; border-radius: 14px; '
        f'padding: {padding}; margin-bottom: 14px;">'
        f'{content}'
        '</div>'
    )


def metric_box(label, value, sub="", accent="#6366f1"):
    """Kotak metric."""
    sub_html = (
        f'<div style="font-size: 11px; color: #7d8ba3; margin-top: 8px;">{sub}</div>'
    ) if sub else ''
    return (
        '<div style="'
        'background: linear-gradient(180deg, #131a2a 0%, #0f1628 100%); '
        'border: 1px solid #1a2338; border-radius: 12px; '
        'padding: 16px 20px;">'
        f'<div style="font-size: 11px; color: #7d8ba3; letter-spacing: 0.07em; '
        f'font-weight: 600; text-transform: uppercase; margin-bottom: 10px;">'
        f'{label}</div>'
        f'<div style="font-size: 24px; font-weight: 800; color: #ffffff; '
        f'letter-spacing: -0.02em;">{value}</div>'
        f'{sub_html}'
        '</div>'
    )


def empty_state(icon, title, subtitle=""):
    """Empty state."""
    return (
        '<div style="text-align: center; padding: 60px 20px; '
        'background: #0f1628; border: 1px dashed #24304d; '
        'border-radius: 14px;">'
        f'<div style="font-size: 48px; margin-bottom: 16px; opacity: 0.5;">{icon}</div>'
        f'<div style="font-size: 16px; font-weight: 700; color: #e6ebf5; '
        f'margin-bottom: 8px;">{title}</div>'
        f'<div style="font-size: 13px; color: #7d8ba3;">{subtitle}</div>'
        '</div>'
    )