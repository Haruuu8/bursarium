"""
Loader & parser fundamental dari data_fundamental/*.json
- Auto-detect struktur (karena tiap endpoint beda format)
- Normalisasi nama metrik (PER, PBV, ROE, DER, dll)
- Hitung composite score
"""

import os
import json
import numpy as np
import pandas as pd


FUND_DIR = "data_fundamental"


# ============================================================
# MAPPING NAMA METRIK (fleksibel)
# ============================================================
METRIC_ALIASES = {
    "per": [
        "per", "pe", "priceearning", "priceearningratio",
        "price_to_earning", "peratio", "rasiopriceearning",
        "priceearningratioannualized",
    ],
    "pbv": [
        "pbv", "pb", "pricebook", "pricebookvalue",
        "price_to_book", "pbvratio", "rasiopricebook",
    ],
    "roe": [
        "roe", "returnonequity", "return_on_equity",
        "rasioReturnOnEquity",
    ],
    "roa": [
        "roa", "returnonasset", "return_on_assets",
    ],
    "der": [
        "der", "debtequity", "debtequityratio",
        "debt_to_equity", "rasioDebtToEquity",
    ],
    "eps": [
        "eps", "earningpershare", "earning_per_share",
    ],
    "bv": [
        "bv", "bookvalue", "book_value", "nilaiBuku",
    ],
    "dividend_yield": [
        "dividendyield", "dividend_yield", "divyield",
        "yielddividen",
    ],
    "net_margin": [
        "netprofitmargin", "net_margin", "npm",
    ],
    "gross_margin": [
        "grossprofitmargin", "gross_margin", "gpm",
    ],
    "revenue": [
        "revenue", "pendapatan", "totRevenue", "sales",
    ],
    "net_income": [
        "netincome", "labaBersih", "net_income", "profit",
    ],
    "market_cap": [
        "marketcap", "market_cap", "kapitalisasi",
    ],
    "total_asset": [
        "totalasset", "total_asset", "jumlahAset",
    ],
    "total_equity": [
        "totalequity", "total_equity", "ekuitas",
    ],
    "current_ratio": [
        "currentratio", "current_ratio", "rasioLancar",
    ],
}


def _normalize_key(k):
    """Lowercase + hapus non-alphanumeric."""
    return "".join(c for c in str(k).lower() if c.isalnum())


def _find_metric_recursive(obj, aliases_set, depth=0):
    """
    Cari value dari key yang match alias, secara rekursif.
    Return list of (key, value) yang match.
    """
    if depth > 6:
        return []

    found = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            nk = _normalize_key(k)
            if nk in aliases_set and not isinstance(v, (dict, list)):
                found.append((k, v))
            elif isinstance(v, (dict, list)):
                found.extend(_find_metric_recursive(v, aliases_set, depth + 1))

    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                found.extend(_find_metric_recursive(item, aliases_set, depth + 1))

    return found


def _pick_best(values):
    """Dari beberapa value yang match, pilih yang paling masuk akal."""
    if not values:
        return None
    parsed = []
    for _, v in values:
        try:
            if v is None or v == "" or v == "-":
                continue
            if isinstance(v, str):
                v_clean = v.replace(",", "").replace("%", "").strip()
                if v_clean == "":
                    continue
                val = float(v_clean)
            else:
                val = float(v)
            if np.isnan(val) or np.isinf(val):
                continue
            parsed.append(val)
        except (ValueError, TypeError):
            continue

    if not parsed:
        return None
    # Pilih median (robust terhadap outlier / multiple values)
    return float(np.median(parsed))


# ============================================================
# LOAD SATU FILE
# ============================================================
def parse_one(json_path):
    """Parse 1 file JSON → dict metrik (support format lama & baru)."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    kode = os.path.basename(json_path).replace(".json", "")

    # Format baru (yfinance) — metrics sudah siap
    if isinstance(data, dict) and "metrics" in data:
        m = data["metrics"].copy()
        m["kode"] = kode
        return m

    # Format lama (IDX scraper) — parse dari raw
    raw = data.get("raw", data) if isinstance(data, dict) else data
    metrics = {"kode": kode}
    for metric_name, aliases in METRIC_ALIASES.items():
        aliases_set = set(_normalize_key(a) for a in aliases)
        found = _find_metric_recursive(raw, aliases_set)
        val = _pick_best(found)
        metrics[metric_name] = val
    return metrics


# ============================================================
# LOAD SEMUA
# ============================================================
def load_all_fundamental(fund_dir=FUND_DIR):
    """Load semua JSON di folder."""
    if not os.path.exists(fund_dir):
        return pd.DataFrame()

    rows = []
    for f in os.listdir(fund_dir):
        if not f.endswith(".json"):
            continue
        path = os.path.join(fund_dir, f)
        m = parse_one(path)
        if m:
            rows.append(m)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ============================================================
# SKOR FUNDAMENTAL
# ============================================================
def score_fundamental(row):
    """
    Composite score 0-100.
    Semakin tinggi = semakin menarik.
    """
    score = 0
    max_score = 0

    # --- Valuasi ---
    if pd.notna(row.get("per")) and row["per"] > 0:
        max_score += 20
        per = row["per"]
        if per < 8:
            score += 20
        elif per < 12:
            score += 16
        elif per < 18:
            score += 11
        elif per < 25:
            score += 6
        elif per < 35:
            score += 2

    if pd.notna(row.get("pbv")) and row["pbv"] > 0:
        max_score += 20
        pbv = row["pbv"]
        if pbv < 0.8:
            score += 20
        elif pbv < 1.5:
            score += 16
        elif pbv < 2.5:
            score += 11
        elif pbv < 4:
            score += 6
        elif pbv < 6:
            score += 2

    # --- Profitabilitas ---
    if pd.notna(row.get("roe")):
        max_score += 20
        roe = row["roe"]
        if abs(roe) < 1:
            roe *= 100
        if roe > 20:
            score += 20
        elif roe > 15:
            score += 16
        elif roe > 10:
            score += 11
        elif roe > 5:
            score += 6
        elif roe > 0:
            score += 2

    # --- Leverage ---
    if pd.notna(row.get("der")):
        max_score += 15
        der = row["der"]
        if der < 0.3:
            score += 15
        elif der < 0.6:
            score += 12
        elif der < 1.0:
            score += 8
        elif der < 1.5:
            score += 4
        elif der < 2.5:
            score += 1

    # --- Dividend ---
    if pd.notna(row.get("dividend_yield")):
        max_score += 15
        dy = row["dividend_yield"]
        if abs(dy) < 1:
            dy *= 100
        if dy > 5:
            score += 15
        elif dy > 3:
            score += 11
        elif dy > 2:
            score += 7
        elif dy > 1:
            score += 3

    # --- Net Margin ---
    if pd.notna(row.get("net_margin")):
        max_score += 10
        nm = row["net_margin"]
        if abs(nm) < 1:
            nm *= 100
        if nm > 20:
            score += 10
        elif nm > 10:
            score += 7
        elif nm > 5:
            score += 4
        elif nm > 0:
            score += 1

    if max_score == 0:
        return None

    final = score / max_score * 100
    return round(final, 1)


def classify(score):
    if score is None or pd.isna(score):
        return "❓ N/A", "#8091b8"
    if score >= 70:
        return "🟢 UNDERVALUED", "#00d47a"
    elif score >= 45:
        return "🟡 FAIR", "#ffa726"
    else:
        return "🔴 OVERVALUED", "#ff5252"


def enrich(df):
    """Tambah kolom skor & klasifikasi."""
    if df.empty:
        return df
    df = df.copy()
    df["score"] = df.apply(score_fundamental, axis=1)
    df["class"] = df["score"].apply(lambda s: classify(s)[0])
    return df.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)


# ============================================================
# DEBUG: INSPECT STRUKTUR JSON
# ============================================================
def inspect_structure(kode="BBCA", fund_dir=FUND_DIR):
    """Print struktur JSON untuk debug."""
    path = os.path.join(fund_dir, f"{kode}.json")
    if not os.path.exists(path):
        print(f"File tidak ada: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"=== Struktur {kode}.json ===")
    _print_tree(data, max_depth=4)


def _print_tree(obj, prefix="", depth=0, max_depth=4):
    if depth > max_depth:
        return
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:30]:
            if isinstance(v, (dict, list)):
                print(f"{prefix}{k}: ({type(v).__name__})")
                _print_tree(v, prefix + "  ", depth + 1, max_depth)
            else:
                val_str = str(v)[:60]
                print(f"{prefix}{k}: {val_str}")
    elif isinstance(obj, list):
        print(f"{prefix}[list len={len(obj)}]")
        if obj and isinstance(obj[0], (dict, list)):
            _print_tree(obj[0], prefix + "  ", depth + 1, max_depth)


if __name__ == "__main__":
    df = load_all_fundamental()
    print(f"Loaded: {len(df)} saham")
    if not df.empty:
        print(df.to_string())
        df2 = enrich(df)
        print("\n=== Setelah scoring ===")
        cols = ["kode", "score", "class", "per", "pbv", "roe", "der"]
        cols = [c for c in cols if c in df2.columns]
        print(df2[cols].to_string())