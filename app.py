# ============================================================================
# HedGEX — Raw Data Collector
# Powered by NYZTrade Analytics Pvt. Ltd.
# Purpose : Collect raw options chain data (OI, IV, GEX, VANNA, Spot)
#           from Dhan Rolling Option API v2 and store in SQLite.
# No algo. No signals. No backtest. Pure data collection only.
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime, timedelta, date
import pytz, requests, time, sqlite3, json, os, warnings
from typing import Dict, List
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HedGEX Raw Data Collector",
    page_icon="🗄️", layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ─────────────────────────────────────────────────────────────────
IST       = pytz.timezone("Asia/Kolkata")
DHAN_BASE = "https://api.dhan.co/v2"
RISK_FREE = 0.07
DB_PATH   = "hedgex_raw.db"
CKPT_PATH = "hedgex_collector_ckpt.json"

# ── MASTER CONFIG — update token here when it expires ─────────────────────────
DHAN_CLIENT_ID    = "1100480354"
DHAN_ACCESS_TOKEN = (
   "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzc2MzY2NTEzLCJhcHBfaWQiOiJhYjYxZmJmOSIsImlhdCI6MTc3NjI4MDExMywidG9rZW5Db25zdW1lclR5cGUiOiJBUFAiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMDQ4MDM1NCJ9.tWhBYSbeNT25V9OJcK3mdV1OHorASWlX_GH-iwNSfmKcY-6PB6hJDHenzZC6-mvrLcOZ3LgVUp8oAtQopwcovQ"
)

DHAN_INDEX_SECURITY_IDS = {
    "NIFTY": 13, "BANKNIFTY": 25, "FINNIFTY": 27,
    "MIDCPNIFTY": 442, "SENSEX": 51,
}
BSE_FNO_SYMBOLS = {"SENSEX"}

INDEX_CONFIG = {
    "NIFTY":      {"contract_size": 25,  "strike_interval": 50},
    "BANKNIFTY":  {"contract_size": 15,  "strike_interval": 100},
    "FINNIFTY":   {"contract_size": 40,  "strike_interval": 50},
    "MIDCPNIFTY": {"contract_size": 75,  "strike_interval": 25},
    "SENSEX":     {"contract_size": 10,  "strike_interval": 200},
}

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;}
.hdr{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);
  border:1px solid rgba(6,182,212,0.3);border-radius:16px;padding:28px 36px;margin-bottom:20px;}
.hdr-title{font-size:2rem;font-weight:800;letter-spacing:-0.02em;
  background:linear-gradient(135deg,#00f5c4,#00d4ff,#a78bfa);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hdr-sub{font-family:"JetBrains Mono",monospace;color:rgba(255,255,255,0.45);font-size:0.82rem;margin-top:4px;}
.info-box{background:rgba(6,182,212,0.08);border-left:3px solid #06b6d4;
  border-radius:6px;padding:10px 14px;font-family:"JetBrains Mono",monospace;
  font-size:0.80rem;line-height:1.8;margin-bottom:8px;}
.warn-box{background:rgba(245,158,11,0.08);border-left:3px solid #f59e0b;
  border-radius:6px;padding:10px 14px;font-family:"JetBrains Mono",monospace;
  font-size:0.80rem;line-height:1.8;margin-bottom:8px;}
.ok-box{background:rgba(16,185,129,0.08);border-left:3px solid #10b981;
  border-radius:6px;padding:10px 14px;font-family:"JetBrains Mono",monospace;
  font-size:0.80rem;line-height:1.8;margin-bottom:8px;}
.metric-card{background:rgba(255,255,255,0.04);border:1px solid rgba(6,182,212,0.2);
  border-radius:12px;padding:14px 18px;text-align:center;}
.metric-val{font-size:1.5rem;font-weight:800;color:#00d4ff;}
.metric-lbl{font-size:0.70rem;color:rgba(255,255,255,0.45);
  font-family:"JetBrains Mono",monospace;margin-top:4px;}
</style>
""", unsafe_allow_html=True)

# ── Black-Scholes Greeks (Gamma + Vanna only) ─────────────────────────────────
class BS:
    @staticmethod
    def _d1(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return 0.0
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def _d2(S, K, T, r, sigma):
        return BS._d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def gamma(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0.0
        try:
            return norm.pdf(BS._d1(S, K, T, r, sigma)) / (S * sigma * np.sqrt(T))
        except: return 0.0

    @staticmethod
    def vanna(S, K, T, r, sigma):
        """∂Δ/∂σ = -d₂ × N'(d₁) / σ"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0.0
        try:
            d1 = BS._d1(S, K, T, r, sigma)
            d2 = BS._d2(S, K, T, r, sigma)
            return -norm.pdf(d1) * d2 / sigma
        except: return 0.0

    @staticmethod
    def charm(S, K, T, r, sigma, is_call=True):
        """∂Δ/∂t — rate of delta change with time (per day).
        Also called delta decay. Sign: positive = delta rises with time.
        charm = -N'(d1) × [2rT - d2·σ√T] / (2T·σ√T)
        """
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0.0
        try:
            d1  = BS._d1(S, K, T, r, sigma)
            d2  = BS._d2(S, K, T, r, sigma)
            num = 2 * r * T - d2 * sigma * np.sqrt(T)
            ch  = -norm.pdf(d1) * num / (2 * T * sigma * np.sqrt(T))
            return ch if is_call else -ch
        except: return 0.0

    @staticmethod
    def delta(S, K, T, r, sigma, is_call=True):
        """Standard Black-Scholes delta."""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return (1.0 if is_call else 0.0) if S >= K else (0.0 if is_call else -1.0)
        try:
            d1 = BS._d1(S, K, T, r, sigma)
            return norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0
        except: return 0.0

# ── API header ────────────────────────────────────────────────────────────────
def get_headers() -> Dict:
    return {
        "access-token": DHAN_ACCESS_TOKEN,
        "client-id":    DHAN_CLIENT_ID,
        "Content-Type": "application/json",
    }

# ── Database ──────────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS raw_chain (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol        TEXT,
            trade_date    TEXT,
            timestamp     TEXT,
            expiry_code   INTEGER,
            expiry_flag   TEXT,
            interval_min  TEXT,
            strike_type   TEXT,
            strike        REAL,
            spot_price    REAL,
            -- OI (Open Interest)
            call_oi       REAL,
            put_oi        REAL,
            -- OI Change (bar-over-bar diff, computed after full day collected)
            call_oi_chg   REAL,
            put_oi_chg    REAL,
            -- Volume
            call_vol      REAL,
            put_vol       REAL,
            -- Implied Volatility (%)
            call_iv       REAL,
            put_iv        REAL,
            -- Option LTP (Last Traded Price)
            call_ltp      REAL,
            put_ltp       REAL,
            -- GEX: (OI × Gamma × Spot² × Lot) / 1e9  [Billions]
            call_gex      REAL,
            put_gex       REAL,
            net_gex       REAL,
            -- VANNA: (OI × Vanna × Spot × Lot) / 1e9  [Billions]
            call_vanna    REAL,
            put_vanna     REAL,
            net_vanna     REAL,
            -- Raw greeks for reference
            call_gamma    REAL,
            put_gamma     REAL,
            call_vanna_greek REAL,
            put_vanna_greek  REAL,
            UNIQUE(symbol, trade_date, timestamp, strike_type, expiry_code, expiry_flag)
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS derived_snapshots (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol                TEXT,
            trade_date            TEXT,
            timestamp             TEXT,
            expiry_code           INTEGER,
            expiry_flag           TEXT,
            interval_min          TEXT,
            spot_price            REAL,
            -- IV metrics
            avg_iv                REAL,
            atm_iv                REAL,
            iv_skew               REAL,
            iv_change             REAL,
            iv_regime             TEXT,
            iv_term_structure     REAL,
            -- OI metrics
            total_call_oi         REAL,
            total_put_oi          REAL,
            pcr_oi                REAL,
            pcr_volume            REAL,
            max_pain              REAL,
            call_oi_concentration REAL,
            put_oi_concentration  REAL,
            oi_buildup_signal     TEXT,
            -- GEX derivatives
            net_gex_total         REAL,
            cumulative_gex_above  REAL,
            cumulative_gex_below  REAL,
            gex_flip_level        REAL,
            gex_skew              REAL,
            largest_gex_strike    REAL,
            -- VANNA derivatives
            net_vanna_total       REAL,
            vacuum_zone_level     REAL,
            trap_door_level       REAL,
            support_floor_level   REAL,
            resistance_ceil_level REAL,
            vanna_skew            REAL,
            -- Enhanced OI VANNA (flow)
            net_flow_vanna_total  REAL,
            -- Cascade mathematics
            bear_fuel_pts         REAL,
            bear_absorb_pts       REAL,
            bull_fuel_pts         REAL,
            bull_absorb_pts       REAL,
            bear_quality          REAL,
            bull_quality          REAL,
            cascade_direction     TEXT,
            estimated_cascade_pts REAL,
            -- Charm
            net_charm_total       REAL,
            UNIQUE(symbol, trade_date, timestamp, expiry_code, expiry_flag, interval_min)
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            symbol        TEXT,
            trade_date    TEXT,
            expiry_code   INTEGER,
            expiry_flag   TEXT,
            interval_min  TEXT,
            status        TEXT,
            rows_fetched  INTEGER,
            fetched_at    TEXT,
            PRIMARY KEY(symbol, trade_date, expiry_code, expiry_flag, interval_min)
        )""")
    con.commit()
    con.close()

def get_fetched_dates(symbol, expiry_code, expiry_flag, interval_min):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""SELECT trade_date FROM fetch_log
                   WHERE symbol=? AND expiry_code=? AND expiry_flag=?
                   AND interval_min=? AND status='ok'""",
                (symbol, expiry_code, expiry_flag, interval_min))
    done = {r[0] for r in cur.fetchall()}
    con.close()
    return done

def log_fetch(symbol, trade_date, expiry_code, expiry_flag, interval_min, status, rows):
    con = sqlite3.connect(DB_PATH)
    con.execute("""INSERT OR REPLACE INTO fetch_log VALUES(?,?,?,?,?,?,?,?)""",
                (symbol, trade_date, expiry_code, expiry_flag, interval_min,
                 status, rows, datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")))
    con.commit()
    con.close()

def save_raw_rows(rows: List[Dict]):
    if not rows: return
    con = sqlite3.connect(DB_PATH)
    con.executemany("""
        INSERT OR IGNORE INTO raw_chain (
            symbol, trade_date, timestamp, expiry_code, expiry_flag, interval_min,
            strike_type, strike, spot_price,
            call_oi, put_oi, call_oi_chg, put_oi_chg,
            call_vol, put_vol,
            call_iv, put_iv,
            call_ltp, put_ltp,
            call_gex, put_gex, net_gex,
            call_vanna, put_vanna, net_vanna,
            call_gamma, put_gamma, call_vanna_greek, put_vanna_greek
        ) VALUES (
            :symbol, :trade_date, :timestamp, :expiry_code, :expiry_flag, :interval_min,
            :strike_type, :strike, :spot_price,
            :call_oi, :put_oi, :call_oi_chg, :put_oi_chg,
            :call_vol, :put_vol,
            :call_iv, :put_iv,
            :call_ltp, :put_ltp,
            :call_gex, :put_gex, :net_gex,
            :call_vanna, :put_vanna, :net_vanna,
            :call_gamma, :put_gamma, :call_vanna_greek, :put_vanna_greek
        )""", rows)
    con.commit()
    con.close()

def db_stats():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM raw_chain")
    total_rows = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT trade_date), COUNT(DISTINCT symbol) FROM raw_chain")
    days, syms = cur.fetchone()
    cur.execute("SELECT COUNT(DISTINCT trade_date) FROM raw_chain")
    uniq_days = cur.fetchone()[0]
    cur.execute("SELECT SUM(rows_fetched) FROM fetch_log WHERE status='ok'")
    logged_rows = cur.fetchone()[0] or 0
    con.close()
    return {
        "total_rows": total_rows, "days": days,
        "symbols": syms, "logged_rows": logged_rows,
    }

def load_raw_chain(symbol, trade_date, expiry_code, expiry_flag, interval_min):
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT * FROM raw_chain
        WHERE symbol=? AND trade_date=? AND expiry_code=?
          AND expiry_flag=? AND interval_min=?
        ORDER BY timestamp, strike""",
        con, params=(symbol, trade_date, expiry_code, expiry_flag, interval_min))
    con.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# ── Checkpoint ────────────────────────────────────────────────────────────────
def save_checkpoint(symbol, trade_date, expiry_code, expiry_flag,
                    interval_min, completed, partial_rows):
    try:
        with open(CKPT_PATH, "w") as f:
            json.dump({
                "symbol": symbol, "trade_date": trade_date,
                "expiry_code": expiry_code, "expiry_flag": expiry_flag,
                "interval_min": interval_min,
                "completed_strikes": completed,
                "partial_rows": partial_rows,
                "saved_at": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            }, f)
    except: pass

def load_checkpoint(symbol, trade_date, expiry_code, expiry_flag, interval_min):
    if not os.path.exists(CKPT_PATH): return [], []
    try:
        with open(CKPT_PATH) as f:
            ckpt = json.load(f)
        if (ckpt.get("symbol") == symbol
                and ckpt.get("trade_date") == trade_date
                and ckpt.get("expiry_code") == expiry_code
                and ckpt.get("expiry_flag") == expiry_flag
                and ckpt.get("interval_min") == interval_min):
            return ckpt.get("completed_strikes", []), ckpt.get("partial_rows", [])
    except: pass
    return [], []

def clear_checkpoint():
    try:
        if os.path.exists(CKPT_PATH): os.remove(CKPT_PATH)
    except: pass

def checkpoint_status():
    if not os.path.exists(CKPT_PATH): return None
    try:
        with open(CKPT_PATH) as f: return json.load(f)
    except: return None

# ── Dhan API ──────────────────────────────────────────────────────────────────
def fetch_rolling_option(symbol, from_date, to_date, strike_type,
                          option_type, interval, expiry_code, expiry_flag,
                          silent=True):
    sec_id   = DHAN_INDEX_SECURITY_IDS.get(symbol)
    if not sec_id:
        if not silent: st.error(f"Unknown symbol: {symbol}")
        return None
    exchange = "BSE_FNO" if symbol in BSE_FNO_SYMBOLS else "NSE_FNO"
    payload  = {
        "exchangeSegment": exchange,
        "interval":        interval,
        "securityId":      sec_id,
        "instrument":      "OPTIDX",
        "expiryFlag":      expiry_flag,
        "expiryCode":      expiry_code,
        "strike":          strike_type,
        "drvOptionType":   option_type,
        "requiredData":    ["open", "high", "low", "close",
                            "volume", "oi", "iv", "strike", "spot"],
        "fromDate":        from_date,
        "toDate":          to_date,
    }
    try:
        resp = requests.post(
            f"{DHAN_BASE}/charts/rollingoption",
            headers=get_headers(), json=payload, timeout=30
        )
        if resp.status_code == 200:
            return resp.json().get("data", {}) or None
        if not silent:
            st.error(f"HTTP {resp.status_code}: {resp.text[:400]}")
        return None
    except Exception as e:
        if not silent: st.error(f"Exception: {e}")
        return None

def fetch_one_day(symbol, trade_date, strikes, interval_min,
                  expiry_code, expiry_flag,
                  progress_bar=None, status_text=None):
    """
    Fetch all strikes for one trading day.
    Returns number of rows collected and stored.
    """
    cfg           = INDEX_CONFIG.get(symbol, {})
    contract_size = cfg.get("contract_size", 25)
    scaling       = 1e9   # store GEX/VANNA in Billions
    tte           = 7 / 365 if expiry_flag == "WEEK" else 30 / 365
    target_dt     = datetime.strptime(trade_date, "%Y-%m-%d")

    # ±2-day window — same pattern as working GEX dashboard
    from_date = (target_dt - timedelta(days=2)).strftime("%Y-%m-%d")
    to_date   = (target_dt + timedelta(days=2)).strftime("%Y-%m-%d")

    # Resume from checkpoint if available
    completed, all_rows = load_checkpoint(
        symbol, trade_date, expiry_code, expiry_flag, interval_min)
    remaining = [s for s in strikes if s not in completed]
    total     = len(strikes) * 2
    done      = len(completed) * 2

    for stype in remaining:
        if status_text:
            status_text.text(
                f"  {symbol} | {trade_date} | {stype} "
                f"({len(completed)+1}/{len(strikes)})")

        call_data = fetch_rolling_option(
            symbol, from_date, to_date, stype, "CALL",
            interval_min, expiry_code, expiry_flag)
        done += 1
        if progress_bar: progress_bar.progress(min(done / total, 1.0))
        time.sleep(0.3)

        put_data = fetch_rolling_option(
            symbol, from_date, to_date, stype, "PUT",
            interval_min, expiry_code, expiry_flag)
        done += 1
        if progress_bar: progress_bar.progress(min(done / total, 1.0))
        time.sleep(0.3)

        if not call_data or not put_data:
            completed.append(stype)
            save_checkpoint(symbol, trade_date, expiry_code, expiry_flag,
                            interval_min, completed, all_rows)
            continue

        ce = call_data.get("ce", {})
        pe = put_data.get("pe", {})
        if not ce:
            completed.append(stype)
            save_checkpoint(symbol, trade_date, expiry_code, expiry_flag,
                            interval_min, completed, all_rows)
            continue

        ts_list = ce.get("timestamp", [])
        for i, ts in enumerate(ts_list):
            try:
                dt_ist = datetime.fromtimestamp(ts, tz=pytz.UTC).astimezone(IST)
                if dt_ist.date() != target_dt.date():
                    continue

                def _g(src, key, default=0.0):
                    arr = src.get(key, [])
                    return arr[i] if i < len(arr) else default

                spot   = float(_g(ce, "spot",   0) or 0)
                strike = float(_g(ce, "strike", 0) or 0)
                if spot == 0 or strike == 0:
                    continue

                # Raw market data
                c_oi  = float(_g(ce, "oi",     0) or 0)
                p_oi  = float(_g(pe, "oi",     0) or 0)
                c_vol = float(_g(ce, "volume", 0) or 0)
                p_vol = float(_g(pe, "volume", 0) or 0)
                c_iv  = float(_g(ce, "iv",    15) or 15)
                p_iv  = float(_g(pe, "iv",    15) or 15)
                c_ltp = float(_g(ce, "close",  0) or 0)   # close = LTP for options
                p_ltp = float(_g(pe, "close",  0) or 0)

                # Normalise IV: API returns % (e.g. 15.2 means 15.2%)
                civ = max(c_iv / 100 if c_iv > 1 else float(c_iv), 0.01)
                piv = max(p_iv / 100 if p_iv > 1 else float(p_iv), 0.01)

                # Greeks
                cg = BS.gamma(spot, strike, tte, RISK_FREE, civ)
                pg = BS.gamma(spot, strike, tte, RISK_FREE, piv)
                cv = BS.vanna(spot, strike, tte, RISK_FREE, civ)
                pv = BS.vanna(spot, strike, tte, RISK_FREE, piv)

                # GEX = (OI × Gamma × Spot² × Lot) / 1e9
                c_gex  = (c_oi * cg * spot ** 2 * contract_size) / scaling
                p_gex  = -(p_oi * pg * spot ** 2 * contract_size) / scaling
                n_gex  = (c_oi * cg - p_oi * pg) * spot ** 2 * contract_size / scaling

                # VANNA = (OI × Vanna × Spot × Lot) / 1e9
                c_van  = (c_oi * cv * spot * contract_size) / scaling
                p_van  = (p_oi * pv * spot * contract_size) / scaling
                n_van  = (c_oi * cv + p_oi * pv) * spot * contract_size / scaling

                all_rows.append({
                    "symbol":       symbol,
                    "trade_date":   trade_date,
                    "timestamp":    dt_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "expiry_code":  expiry_code,
                    "expiry_flag":  expiry_flag,
                    "interval_min": interval_min,
                    "strike_type":  stype,
                    "strike":       strike,
                    "spot_price":   spot,
                    # OI
                    "call_oi":      c_oi,
                    "put_oi":       p_oi,
                    "call_oi_chg":  0.0,   # computed after full day
                    "put_oi_chg":   0.0,
                    # Volume
                    "call_vol":     c_vol,
                    "put_vol":      p_vol,
                    # IV (store as % — e.g. 15.2)
                    "call_iv":      c_iv,
                    "put_iv":       p_iv,
                    # LTP
                    "call_ltp":     c_ltp,
                    "put_ltp":      p_ltp,
                    # GEX (Billions)
                    "call_gex":     round(c_gex, 6),
                    "put_gex":      round(p_gex, 6),
                    "net_gex":      round(n_gex, 6),
                    # VANNA (Billions)
                    "call_vanna":   round(c_van, 6),
                    "put_vanna":    round(p_van, 6),
                    "net_vanna":    round(n_van, 6),
                    # Raw greeks
                    "call_gamma":       round(cg, 8),
                    "put_gamma":        round(pg, 8),
                    "call_vanna_greek": round(cv, 8),
                    "put_vanna_greek":  round(pv, 8),
                })
            except:
                continue

        completed.append(stype)
        save_checkpoint(symbol, trade_date, expiry_code, expiry_flag,
                        interval_min, completed, all_rows)

    # ── Compute OI Change bar-over-bar after full day is collected ────────────
    if all_rows:
        df = pd.DataFrame(all_rows).sort_values(["strike", "timestamp"])
        for sv in df["strike"].unique():
            m = df["strike"] == sv
            df.loc[m, "call_oi_chg"] = df.loc[m, "call_oi"].diff().fillna(0)
            df.loc[m, "put_oi_chg"]  = df.loc[m, "put_oi"].diff().fillna(0)
        all_rows = df.to_dict("records")

    save_raw_rows(all_rows)
    clear_checkpoint()
    return len(all_rows)

# ── Utility ───────────────────────────────────────────────────────────────────
def get_trading_dates(start: date, end: date) -> List[str]:
    dates, cur = [], start
    while cur <= end:
        if cur.weekday() < 5:   # Mon–Fri
            dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return dates

# ── Main ──────────────────────────────────────────────────────────────────────
# ── Derivatives computation ───────────────────────────────────────────────────

def _vanna_flip_zones(df_ts, spot):
    """Identify all VANNA flip zones from net_vanna sign changes."""
    df_s = df_ts.sort_values("strike").reset_index(drop=True)
    zones = []
    for i in range(len(df_s) - 1):
        cv = df_s.iloc[i]["net_vanna"]
        nv = df_s.iloc[i+1]["net_vanna"]
        ck = df_s.iloc[i]["strike"]
        nk = df_s.iloc[i+1]["strike"]
        if cv * nv < 0:
            w      = abs(cv) / (abs(cv) + abs(nv) + 1e-12)
            flip_k = ck + (nk - ck) * w
            ftype  = "NEG_TO_POS" if cv < 0 else "POS_TO_NEG"
            above  = flip_k > spot
            if   above  and ftype == "NEG_TO_POS": role = "VACUUM_ZONE"
            elif above  and ftype == "POS_TO_NEG": role = "RESISTANCE_CEILING"
            elif not above and ftype == "POS_TO_NEG": role = "TRAP_DOOR"
            else:                                   role = "SUPPORT_FLOOR"
            zones.append({"strike": round(flip_k, 2), "role": role,
                          "magnitude": round((abs(cv)+abs(nv))/2, 6)})
    return zones


def _gex_flip_level(df_ts, spot):
    """Linear interpolation of the gamma flip level (net_gex = 0 crossing)."""
    df_s = df_ts.sort_values("strike").reset_index(drop=True)
    for i in range(len(df_s) - 1):
        g1 = df_s.iloc[i]["net_gex"]
        g2 = df_s.iloc[i+1]["net_gex"]
        k1 = df_s.iloc[i]["strike"]
        k2 = df_s.iloc[i+1]["strike"]
        if g1 * g2 < 0:
            w = abs(g1) / (abs(g1) + abs(g2) + 1e-12)
            return round(k1 + (k2 - k1) * w, 2)
    return None


def _max_pain(df_ts):
    """
    Max pain = strike that minimises total option dollar value at expiry.
    For each candidate strike S:
      call_pain = Σ max(S - K, 0) × call_oi   for all K < S
      put_pain  = Σ max(K - S, 0) × put_oi    for all K > S
      total_pain = call_pain + put_pain
    Return S with minimum total_pain.
    """
    strikes = sorted(df_ts["strike"].unique())
    if len(strikes) < 2:
        return None
    min_pain = float("inf")
    max_pain_strike = strikes[0]
    for s in strikes:
        call_pain = df_ts[df_ts["strike"] < s].apply(
            lambda r: max(s - r["strike"], 0) * r["call_oi"], axis=1).sum()
        put_pain  = df_ts[df_ts["strike"] > s].apply(
            lambda r: max(r["strike"] - s, 0) * r["put_oi"],  axis=1).sum()
        total = call_pain + put_pain
        if total < min_pain:
            min_pain = total
            max_pain_strike = s
    return float(max_pain_strike)


def _enhanced_oi_vanna(df_ts, spot, contract_size, scaling, tte):
    """
    Flow VANNA = call_oi_chg × call_vanna_greek × vol_weight × iv_adj × dist_weight × spot × lot / scale
    Uses OI CHANGE (not total OI) — measures intraday dealer flow.
    """
    total_vol = max(df_ts["call_vol"].sum() + df_ts["put_vol"].sum(), 1.0)
    vals = []
    for _, row in df_ts.iterrows():
        try:
            strike = row["strike"]
            civ = max(row["call_iv"]/100 if row["call_iv"] > 1 else row["call_iv"], 0.01)
            piv = max(row["put_iv"] /100 if row["put_iv"]  > 1 else row["put_iv"],  0.01)
            cv  = BS.vanna(spot, strike, tte, RISK_FREE, civ)
            pv  = BS.vanna(spot, strike, tte, RISK_FREE, piv)
            vw  = 1.0 + (row["call_vol"] + row["put_vol"]) / total_vol
            iv_adj = 1.0 + ((civ + piv) / 2 * 3)
            dw  = 1.0 / (1 + abs(strike - spot) / spot * 1.5)
            eov = (
                (row["call_oi_chg"] * cv * 2.0 * vw * iv_adj * dw * spot * contract_size) / scaling +
                (row["put_oi_chg"]  * pv * 2.0 * vw * iv_adj * dw * spot * contract_size) / scaling
            )
            vals.append(eov)
        except:
            vals.append(0.0)
    return vals


def _cascade_pts(df_ts, spot, symbol):
    """Fuel/Absorption cascade mathematics per snapshot."""
    ppu_map = {
        "NIFTY": 0.010, "BANKNIFTY": 0.033,
        "FINNIFTY": 0.050, "MIDCPNIFTY": 0.050, "SENSEX": 0.025,
    }
    cap_map = {
        "NIFTY": 150, "BANKNIFTY": 300,
        "FINNIFTY": 150, "MIDCPNIFTY": 75, "SENSEX": 500,
    }
    ppu = ppu_map.get(symbol, 0.010)
    cap = cap_map.get(symbol, 150)
    bf = ba = uf = ua = 0.0
    for _, row in df_ts.iterrows():
        s   = row["strike"]
        gex = row["net_gex"]
        rp  = min(abs(gex) * ppu, cap)
        if s < spot:
            if gex < 0: bf += rp
            else:        ba += rp
        else:
            if gex < 0: uf += rp
            else:        ua += rp
    bq   = bf / max(ba, 1.0)
    uq   = uf / max(ua, 1.0)
    net_bear = max(0, bf - ba * 0.5)
    net_bull = max(0, uf - ua * 0.5)
    if bq >= uq and max(bq, uq) >= 0.5:
        direction = "BEAR"
        est_pts   = -net_bear
    elif uq > bq and max(bq, uq) >= 0.5:
        direction = "BULL"
        est_pts   = net_bull
    else:
        direction = "NONE"
        est_pts   = 0.0
    return {
        "bear_fuel_pts":     round(bf,  2),
        "bear_absorb_pts":   round(ba,  2),
        "bull_fuel_pts":     round(uf,  2),
        "bull_absorb_pts":   round(ua,  2),
        "bear_quality":      round(bq,  4),
        "bull_quality":      round(uq,  4),
        "cascade_direction": direction,
        "estimated_cascade_pts": round(est_pts, 2),
    }


def compute_derivatives_for_day(df_day, symbol, trade_date,
                                  expiry_code, expiry_flag, interval_min):
    """
    Compute all derived metrics from raw chain data for one trading day.
    Returns list of dicts — one dict per timestamp (bar-level snapshot).
    """
    cfg           = INDEX_CONFIG.get(symbol, {})
    contract_size = cfg.get("contract_size", 25)
    scaling       = 1e9
    tte           = 7 / 365 if expiry_flag == "WEEK" else 30 / 365

    timestamps  = sorted(df_day["timestamp"].unique())
    prev_avg_iv = None
    results     = []

    for ts in timestamps:
        df_ts = df_day[df_day["timestamp"] == ts].copy()
        if df_ts.empty:
            continue

        spot = float(df_ts["spot_price"].mean())

        # ── IV metrics ────────────────────────────────────────────────────────
        avg_iv   = float((df_ts["call_iv"].mean() + df_ts["put_iv"].mean()) / 2)
        iv_change = float(avg_iv - prev_avg_iv) if prev_avg_iv is not None else 0.0
        prev_avg_iv = avg_iv

        # IV regime
        thr = 0.15  # % IV change threshold
        if   iv_change >  thr: iv_regime = "EXPANDING"
        elif iv_change < -thr: iv_regime = "COMPRESSING"
        else:                   iv_regime = "FLAT"

        # ATM IV — closest strike to spot
        df_ts["dist"] = (df_ts["strike"] - spot).abs()
        atm_row = df_ts.loc[df_ts["dist"].idxmin()]
        atm_iv  = float((atm_row["call_iv"] + atm_row["put_iv"]) / 2)
        iv_skew = float(atm_row["put_iv"] - atm_row["call_iv"])

        # IV term structure: ATM vs 2-strikes-OTM average
        otm_rows = df_ts[df_ts["dist"] >= 2 * cfg.get("strike_interval", 50)]
        iv_term_structure = float(
            (otm_rows["call_iv"].mean() + otm_rows["put_iv"].mean()) / 2 - atm_iv
        ) if len(otm_rows) > 0 else 0.0

        # ── OI metrics ────────────────────────────────────────────────────────
        total_call_oi   = float(df_ts["call_oi"].sum())
        total_put_oi    = float(df_ts["put_oi"].sum())
        pcr_oi          = round(total_put_oi / max(total_call_oi, 1), 4)
        total_call_vol  = float(df_ts["call_vol"].sum())
        total_put_vol   = float(df_ts["put_vol"].sum())
        pcr_volume      = round(total_put_vol / max(total_call_vol, 1), 4)

        # Max pain
        max_pain_strike = _max_pain(df_ts)

        # OI concentration
        call_oi_conc = float(df_ts["call_oi"].max() / max(total_call_oi, 1) * 100)
        put_oi_conc  = float(df_ts["put_oi"].max()  / max(total_put_oi,  1) * 100)

        # OI buildup signal — compare spot move to total OI change direction
        total_oi_chg = float(
            df_ts["call_oi_chg"].sum() + df_ts["put_oi_chg"].sum())
        if   total_oi_chg > 0 and spot >= (prev_avg_iv or spot): oi_buildup = "LONG_BUILDUP"
        elif total_oi_chg > 0 and spot <  (prev_avg_iv or spot): oi_buildup = "SHORT_BUILDUP"
        elif total_oi_chg < 0 and spot >= (prev_avg_iv or spot): oi_buildup = "SHORT_COVERING"
        elif total_oi_chg < 0 and spot <  (prev_avg_iv or spot): oi_buildup = "LONG_UNWINDING"
        else:                                                       oi_buildup = "NEUTRAL"

        # ── GEX derivatives ───────────────────────────────────────────────────
        net_gex_total        = float(df_ts["net_gex"].sum())
        cumulative_gex_above = float(df_ts[df_ts["strike"] > spot]["net_gex"].sum())
        cumulative_gex_below = float(df_ts[df_ts["strike"] < spot]["net_gex"].sum())
        gex_skew             = round(
            cumulative_gex_above / max(abs(cumulative_gex_below), 1e-9), 4)
        gex_flip_lvl         = _gex_flip_level(df_ts, spot)
        largest_gex_row      = df_ts.loc[df_ts["net_gex"].abs().idxmax()]
        largest_gex_strike   = float(largest_gex_row["strike"])

        # ── VANNA derivatives ─────────────────────────────────────────────────
        net_vanna_total = float(df_ts["net_vanna"].sum())
        vanna_above     = float(df_ts[df_ts["strike"] > spot]["net_vanna"].sum())
        vanna_below     = float(df_ts[df_ts["strike"] < spot]["net_vanna"].sum())
        vanna_skew      = round(vanna_above / max(abs(vanna_below), 1e-9), 4)

        vz = _vanna_flip_zones(df_ts, spot)
        vacuum_zone_lvl   = next((z["strike"] for z in vz if z["role"] == "VACUUM_ZONE"),      None)
        trap_door_lvl     = next((z["strike"] for z in vz if z["role"] == "TRAP_DOOR"),         None)
        support_floor_lvl = next((z["strike"] for z in vz if z["role"] == "SUPPORT_FLOOR"),     None)
        resistance_lvl    = next((z["strike"] for z in vz if z["role"] == "RESISTANCE_CEILING"),None)

        # ── Enhanced OI VANNA (flow) ──────────────────────────────────────────
        eov_vals         = _enhanced_oi_vanna(df_ts, spot, contract_size, scaling, tte)
        net_flow_vanna   = float(sum(eov_vals))

        # ── Cascade mathematics ───────────────────────────────────────────────
        cas = _cascade_pts(df_ts, spot, symbol)

        # ── Charm ─────────────────────────────────────────────────────────────
        net_charm = 0.0
        for _, row in df_ts.iterrows():
            try:
                k   = row["strike"]
                civ = max(row["call_iv"]/100 if row["call_iv"] > 1 else row["call_iv"], 0.01)
                piv = max(row["put_iv"] /100 if row["put_iv"]  > 1 else row["put_iv"],  0.01)
                cc  = BS.charm(spot, k, tte, RISK_FREE, civ, is_call=True)
                pc  = BS.charm(spot, k, tte, RISK_FREE, piv, is_call=False)
                net_charm += (row["call_oi"] * cc + row["put_oi"] * pc) * spot * contract_size / scaling
            except:
                pass

        results.append({
            "symbol":        symbol,
            "trade_date":    trade_date,
            "timestamp":     ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts),
            "expiry_code":   expiry_code,
            "expiry_flag":   expiry_flag,
            "interval_min":  interval_min,
            "spot_price":    round(spot, 2),
            # IV
            "avg_iv":                round(avg_iv, 4),
            "atm_iv":                round(atm_iv, 4),
            "iv_skew":               round(iv_skew, 4),
            "iv_change":             round(iv_change, 4),
            "iv_regime":             iv_regime,
            "iv_term_structure":     round(iv_term_structure, 4),
            # OI
            "total_call_oi":         round(total_call_oi, 0),
            "total_put_oi":          round(total_put_oi,  0),
            "pcr_oi":                pcr_oi,
            "pcr_volume":            pcr_volume,
            "max_pain":              max_pain_strike,
            "call_oi_concentration": round(call_oi_conc, 2),
            "put_oi_concentration":  round(put_oi_conc,  2),
            "oi_buildup_signal":     oi_buildup,
            # GEX
            "net_gex_total":         round(net_gex_total, 6),
            "cumulative_gex_above":  round(cumulative_gex_above, 6),
            "cumulative_gex_below":  round(cumulative_gex_below, 6),
            "gex_flip_level":        gex_flip_lvl,
            "gex_skew":              gex_skew,
            "largest_gex_strike":    largest_gex_strike,
            # VANNA
            "net_vanna_total":       round(net_vanna_total, 6),
            "vacuum_zone_level":     vacuum_zone_lvl,
            "trap_door_level":       trap_door_lvl,
            "support_floor_level":   support_floor_lvl,
            "resistance_ceil_level": resistance_lvl,
            "vanna_skew":            vanna_skew,
            # Flow VANNA
            "net_flow_vanna_total":  round(net_flow_vanna, 6),
            # Cascade
            **cas,
            # Charm
            "net_charm_total":       round(net_charm, 6),
        })

    return results


def save_derived(rows: List[Dict]):
    if not rows: return
    con = sqlite3.connect(DB_PATH)
    con.executemany("""
        INSERT OR REPLACE INTO derived_snapshots (
            symbol, trade_date, timestamp, expiry_code, expiry_flag, interval_min,
            spot_price,
            avg_iv, atm_iv, iv_skew, iv_change, iv_regime, iv_term_structure,
            total_call_oi, total_put_oi, pcr_oi, pcr_volume, max_pain,
            call_oi_concentration, put_oi_concentration, oi_buildup_signal,
            net_gex_total, cumulative_gex_above, cumulative_gex_below,
            gex_flip_level, gex_skew, largest_gex_strike,
            net_vanna_total, vacuum_zone_level, trap_door_level,
            support_floor_level, resistance_ceil_level, vanna_skew,
            net_flow_vanna_total,
            bear_fuel_pts, bear_absorb_pts, bull_fuel_pts, bull_absorb_pts,
            bear_quality, bull_quality, cascade_direction, estimated_cascade_pts,
            net_charm_total
        ) VALUES (
            :symbol, :trade_date, :timestamp, :expiry_code, :expiry_flag, :interval_min,
            :spot_price,
            :avg_iv, :atm_iv, :iv_skew, :iv_change, :iv_regime, :iv_term_structure,
            :total_call_oi, :total_put_oi, :pcr_oi, :pcr_volume, :max_pain,
            :call_oi_concentration, :put_oi_concentration, :oi_buildup_signal,
            :net_gex_total, :cumulative_gex_above, :cumulative_gex_below,
            :gex_flip_level, :gex_skew, :largest_gex_strike,
            :net_vanna_total, :vacuum_zone_level, :trap_door_level,
            :support_floor_level, :resistance_ceil_level, :vanna_skew,
            :net_flow_vanna_total,
            :bear_fuel_pts, :bear_absorb_pts, :bull_fuel_pts, :bull_absorb_pts,
            :bear_quality, :bull_quality, :cascade_direction, :estimated_cascade_pts,
            :net_charm_total
        )""", rows)
    con.commit()
    con.close()


def get_derived_dates(symbol, expiry_code, expiry_flag, interval_min):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""SELECT DISTINCT trade_date FROM derived_snapshots
                   WHERE symbol=? AND expiry_code=? AND expiry_flag=? AND interval_min=?""",
                (symbol, expiry_code, expiry_flag, interval_min))
    done = {r[0] for r in cur.fetchall()}
    con.close()
    return done


def load_derived(symbol, trade_date, expiry_code, expiry_flag, interval_min):
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT * FROM derived_snapshots
        WHERE symbol=? AND trade_date=? AND expiry_code=? AND expiry_flag=? AND interval_min=?
        ORDER BY timestamp""",
        con, params=(symbol, trade_date, expiry_code, expiry_flag, interval_min))
    con.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def main():
    init_db()

    # Header
    st.markdown("""
    <div class="hdr">
        <div class="hdr-title">HedGEX — Raw Data Collector</div>
        <div class="hdr-sub">
            Collect OI · OI Change · Volume · IV · LTP · GEX · VANNA · Spot
            &nbsp;·&nbsp; Dhan Rolling Option API v2
            &nbsp;·&nbsp; NYZTrade Analytics
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Collection Config")

        symbol = st.selectbox(
            "Index", list(INDEX_CONFIG.keys()), index=0)

        expiry_flag = st.selectbox(
            "Expiry Type", ["WEEK", "MONTH"], index=0)

        expiry_code = st.selectbox(
            "Expiry Code", [1, 2, 3], index=0,
            format_func=lambda x: {1: "1 — Current", 2: "2 — Next", 3: "3 — Far"}[x])

        interval_min = st.selectbox(
            "Bar Interval", ["5", "15", "60"], index=0,
            format_func=lambda x: f"{x} min")

        st.markdown("---")
        st.markdown("### 📅 Date Range")
        today   = date.today()
        d_end   = st.date_input("End Date",   value=today - timedelta(days=1))
        d_start = st.date_input("Start Date", value=today - timedelta(days=365))

        st.markdown("---")
        st.markdown("### ⚡ Strike Range")
        n_strikes = st.slider("ATM ± N strikes", 3, 15, 7)
        all_strikes = (
            ["ATM"]
            + [f"ATM+{i}" for i in range(1, n_strikes + 1)]
            + [f"ATM-{i}" for i in range(1, n_strikes + 1)]
        )
        st.caption(f"Total: {len(all_strikes)} strikes × 2 legs = "
                   f"{len(all_strikes)*2} API calls per bar")

        st.markdown("---")
        st.markdown("### 🗄️ Database")
        stats = db_stats()
        st.markdown(
            '<div class="info-box">'
            f'Rows in DB: <b>{stats["total_rows"]:,}</b><br>'
            f'Days stored: <b>{stats["days"]}</b><br>'
            f'Symbols:     <b>{stats["symbols"]}</b>'
            '</div>', unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_collect, tab_view, tab_deriv, tab_export, tab_inspect = st.tabs([
        "🚀 Collect Data",
        "🔍 View Data",
        "⚗️ Derivatives",
        "📥 Export",
        "🔬 API Inspector",
    ])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 — Collect Data
    # ═════════════════════════════════════════════════════════════════════════
    with tab_collect:
        st.markdown("### 📡 Collect Raw Options Chain Data")

        st.markdown("""
        <div class="info-box">
        <b>What is collected per bar per strike:</b><br>
        • <b>Spot price</b> — underlying index level<br>
        • <b>Call OI / Put OI</b> — open interest in contracts<br>
        • <b>Call OI Change / Put OI Change</b> — bar-over-bar OI delta<br>
        • <b>Call Volume / Put Volume</b> — traded volume<br>
        • <b>Call IV / Put IV</b> — implied volatility (%)<br>
        • <b>Call LTP / Put LTP</b> — last traded price (option price)<br>
        • <b>Call GEX / Put GEX / Net GEX</b> — gamma exposure (Billions)<br>
        • <b>Call VANNA / Put VANNA / Net VANNA</b> — vanna exposure (Billions)<br>
        • <b>Raw greeks</b> — gamma, vanna per strike (for verification)
        </div>""", unsafe_allow_html=True)

        # Checkpoint check
        ckpt = checkpoint_status()
        if ckpt:
            st.markdown(
                '<div class="warn-box">⚡ <b>Checkpoint detected</b> — '
                f'{ckpt.get("trade_date","?")} | '
                f'{len(ckpt.get("completed_strikes",[]))} strikes done | '
                f'{len(ckpt.get("partial_rows",[]))} rows buffered | '
                f'Saved at {ckpt.get("saved_at","?")}. '
                'Next fetch will resume automatically.</div>',
                unsafe_allow_html=True)
            if st.button("🗑️ Discard Checkpoint & Start Fresh"):
                clear_checkpoint()
                st.rerun()

        # Date planning
        trading_dates = get_trading_dates(d_start, d_end)
        done_dates    = get_fetched_dates(symbol, expiry_code, expiry_flag, interval_min)
        pending       = [d for d in trading_dates if d not in done_dates]

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-val">{len(trading_dates)}</div>'
                    '<div class="metric-lbl">Trading Days in Range</div></div>',
                    unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#10b981">'
                    f'{len(done_dates)}</div>'
                    '<div class="metric-lbl">Already Collected</div></div>',
                    unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#f59e0b">'
                    f'{len(pending)}</div>'
                    '<div class="metric-lbl">Pending</div></div>',
                    unsafe_allow_html=True)
        calls_per_day = len(all_strikes) * 2
        c4.markdown(f'<div class="metric-card"><div class="metric-val">'
                    f'{calls_per_day}</div>'
                    '<div class="metric-lbl">API Calls / Day</div></div>',
                    unsafe_allow_html=True)

        st.markdown("")

        if pending:
            est_min = len(pending) * calls_per_day * 0.35 / 60
            st.markdown(
                f'<div class="warn-box">⏱️ Estimated collection time: '
                f'<b>~{est_min:.1f} minutes</b> for {len(pending)} days × '
                f'{calls_per_day} API calls/day (0.35s delay per call)</div>',
                unsafe_allow_html=True)

        fetch_btn = st.button(
            f"🚀 Start Collection — {len(pending)} Days",
            type="primary", use_container_width=True,
            disabled=(len(pending) == 0))

        if fetch_btn:
            overall_bar = st.progress(0)
            day_bar     = st.progress(0)
            day_status  = st.empty()
            strike_status = st.empty()
            log_box     = st.empty()
            log_lines   = []

            for idx, trade_date in enumerate(pending):
                day_status.markdown(
                    f"**Day {idx+1} / {len(pending)}** — `{trade_date}`")

                # If checkpoint is for a different date, clear it
                ckpt_now = checkpoint_status()
                if ckpt_now and ckpt_now.get("trade_date") != trade_date:
                    clear_checkpoint()

                try:
                    n = fetch_one_day(
                        symbol, trade_date, all_strikes, interval_min,
                        expiry_code, expiry_flag,
                        progress_bar=day_bar,
                        status_text=strike_status)

                    log_fetch(symbol, trade_date, expiry_code, expiry_flag,
                              interval_min, "ok", n)
                    log_lines.append(f"✅  {trade_date}  →  {n:,} rows saved")
                    if n == 0:
                        log_lines.append(
                            f"   ⚠️  0 rows on {trade_date} — verify expiry_code "
                            f"or date range (holiday / no data?)")

                except Exception as e:
                    log_fetch(symbol, trade_date, expiry_code, expiry_flag,
                              interval_min, "error", 0)
                    log_lines.append(f"⚠️  {trade_date}  →  Error: {e} (checkpoint saved)")
                    log_box.text("\n".join(log_lines[-20:]))
                    st.warning(
                        f"Interrupted at {trade_date}. "
                        "Click **Start Collection** again to resume from checkpoint.")
                    break

                overall_bar.progress((idx + 1) / len(pending))
                log_box.text("\n".join(log_lines[-20:]))

            overall_bar.empty()
            day_bar.empty()
            day_status.empty()
            strike_status.empty()

            if log_lines and "⚠️" not in log_lines[-1]:
                st.markdown(
                    '<div class="ok-box">✅ <b>Collection complete!</b> '
                    'Go to <b>View Data</b> or <b>Export</b> tabs.</div>',
                    unsafe_allow_html=True)

        if done_dates:
            st.markdown("---")
            st.markdown("#### ✅ Already Collected Days")
            con = sqlite3.connect(DB_PATH)
            log_df = pd.read_sql_query("""
                SELECT trade_date, rows_fetched, fetched_at
                FROM fetch_log
                WHERE symbol=? AND expiry_code=? AND expiry_flag=?
                  AND interval_min=? AND status='ok'
                ORDER BY trade_date DESC""",
                con, params=(symbol, expiry_code, expiry_flag, interval_min))
            con.close()
            st.dataframe(log_df, use_container_width=True,
                         height=300, hide_index=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 — View Data
    # ═════════════════════════════════════════════════════════════════════════
    with tab_view:
        st.markdown("### 🔍 View Collected Data")

        # Date picker from what's available
        con = sqlite3.connect(DB_PATH)
        avail = pd.read_sql_query(
            "SELECT DISTINCT trade_date FROM raw_chain WHERE symbol=? ORDER BY trade_date DESC",
            con, params=(symbol,))
        con.close()

        if avail.empty:
            st.info("No data collected yet. Go to **Collect Data** first.")
        else:
            view_date = st.selectbox(
                "Select Date", avail["trade_date"].tolist())

            df = load_raw_chain(
                symbol, view_date, expiry_code, expiry_flag, interval_min)

            if df.empty:
                st.warning(
                    f"No data for {view_date} with these settings. "
                    "Try different expiry_code or interval.")
            else:
                timestamps = sorted(df["timestamp"].unique())
                ts_sel = st.selectbox(
                    "Timestamp", timestamps,
                    index=len(timestamps) - 1,
                    format_func=lambda x: pd.to_datetime(x).strftime("%H:%M:%S"))

                df_ts = df[df["timestamp"] == ts_sel].copy()
                spot  = df_ts["spot_price"].mean()

                st.markdown(
                    f'<div class="info-box">'
                    f'Symbol: <b>{symbol}</b> &nbsp;|&nbsp; '
                    f'Date: <b>{view_date}</b> &nbsp;|&nbsp; '
                    f'Time: <b>{pd.to_datetime(ts_sel).strftime("%H:%M")}</b> &nbsp;|&nbsp; '
                    f'Spot: <b>₹{spot:,.2f}</b> &nbsp;|&nbsp; '
                    f'Strikes: <b>{len(df_ts)}</b> &nbsp;|&nbsp; '
                    f'Bars: <b>{len(timestamps)}</b>'
                    f'</div>', unsafe_allow_html=True)

                # Column selector
                display_groups = {
                    "All columns": df_ts.columns.tolist(),
                    "OI & Volume": ["strike", "spot_price", "call_oi", "put_oi",
                                    "call_oi_chg", "put_oi_chg", "call_vol", "put_vol"],
                    "IV & LTP":    ["strike", "spot_price", "call_iv", "put_iv",
                                    "call_ltp", "put_ltp"],
                    "GEX":         ["strike", "spot_price", "call_gex", "put_gex", "net_gex"],
                    "VANNA":       ["strike", "spot_price", "call_vanna", "put_vanna", "net_vanna"],
                    "Raw Greeks":  ["strike", "spot_price",
                                    "call_gamma", "put_gamma",
                                    "call_vanna_greek", "put_vanna_greek"],
                }
                col_group = st.selectbox("Show columns", list(display_groups.keys()))
                cols_show = [c for c in display_groups[col_group]
                             if c in df_ts.columns]

                st.dataframe(
                    df_ts[cols_show].sort_values("strike"),
                    use_container_width=True, height=400, hide_index=True)

                # Quick charts
                st.markdown("#### Net GEX by Strike")
                fig_gex = go.Figure()
                df_plot = df_ts.sort_values("strike")
                fig_gex.add_trace(go.Bar(
                    x=df_plot["strike"], y=df_plot["net_gex"],
                    marker_color=df_plot["net_gex"].apply(
                        lambda x: "#a78bfa" if x >= 0 else "#fbbf24"),
                    name="Net GEX"))
                fig_gex.add_vline(x=spot, line_dash="dash",
                                  line_color="#00d4ff", annotation_text="Spot")
                fig_gex.update_layout(
                    template="plotly_dark", height=320,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10,10,20,0.95)",
                    yaxis_title="GEX (B)",
                    margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_gex, use_container_width=True)

                st.markdown("#### Net VANNA by Strike")
                fig_van = go.Figure()
                fig_van.add_trace(go.Bar(
                    x=df_plot["strike"], y=df_plot["net_vanna"],
                    marker_color=df_plot["net_vanna"].apply(
                        lambda x: "#ec4899" if x >= 0 else "#be185d"),
                    name="Net VANNA"))
                fig_van.add_vline(x=spot, line_dash="dash",
                                  line_color="#00d4ff", annotation_text="Spot")
                fig_van.update_layout(
                    template="plotly_dark", height=320,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10,10,20,0.95)",
                    yaxis_title="VANNA (B)",
                    margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_van, use_container_width=True)

                st.markdown("#### OI — Call vs Put by Strike")
                fig_oi = go.Figure()
                fig_oi.add_trace(go.Bar(
                    x=df_plot["strike"], y=df_plot["call_oi"],
                    name="Call OI", marker_color="#10b981", opacity=0.8))
                fig_oi.add_trace(go.Bar(
                    x=df_plot["strike"], y=-df_plot["put_oi"],
                    name="Put OI", marker_color="#ef4444", opacity=0.8))
                fig_oi.add_vline(x=spot, line_dash="dash",
                                 line_color="#00d4ff", annotation_text="Spot")
                fig_oi.update_layout(
                    template="plotly_dark", height=320, barmode="overlay",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10,10,20,0.95)",
                    yaxis_title="OI (contracts)",
                    margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_oi, use_container_width=True)


    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3 — Derivatives
    # ═════════════════════════════════════════════════════════════════════════
    with tab_deriv:
        st.markdown("### ⚗️ Compute Derivatives from Raw Chain Data")

        st.markdown("""
        <div class="info-box">
        Computes <b>40 derived metrics</b> from the stored raw chain data — one row per bar
        (timestamp) per trading day. All metrics are written to the
        <code>derived_snapshots</code> table and can be exported to CSV.<br><br>
        <b>IV:</b> avg_iv · atm_iv · iv_skew · iv_change · iv_regime · iv_term_structure<br>
        <b>OI:</b> pcr_oi · pcr_volume · max_pain · oi_concentration · oi_buildup_signal<br>
        <b>GEX:</b> net_gex_total · gex_flip_level · cumulative_above/below · gex_skew<br>
        <b>VANNA:</b> net_vanna_total · vacuum_zone · trap_door · support_floor · resistance · vanna_skew<br>
        <b>Flow VANNA:</b> enhanced_oi_vanna (OI change weighted by greeks + IV + distance)<br>
        <b>Cascade:</b> bear/bull fuel · absorb · quality · direction · estimated_pts<br>
        <b>Charm:</b> net_charm_total (dealer delta decay — expiry pin force)
        </div>""", unsafe_allow_html=True)

        # Date status
        raw_dates     = get_fetched_dates(symbol, expiry_code, expiry_flag, interval_min)
        derived_dates = get_derived_dates(symbol, expiry_code, expiry_flag, interval_min)
        pending_deriv = sorted(raw_dates - derived_dates)

        dc1, dc2, dc3 = st.columns(3)
        dc1.markdown(f'<div class="metric-card"><div class="metric-val">{len(raw_dates)}</div>'
                     '<div class="metric-lbl">Raw Days Available</div></div>',
                     unsafe_allow_html=True)
        dc2.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#10b981">'
                     f'{len(derived_dates)}</div>'
                     '<div class="metric-lbl">Derivatives Computed</div></div>',
                     unsafe_allow_html=True)
        dc3.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#f59e0b">'
                     f'{len(pending_deriv)}</div>'
                     '<div class="metric-lbl">Pending</div></div>',
                     unsafe_allow_html=True)

        st.markdown("")

        col_run, col_recomp = st.columns([3, 1])
        run_btn     = col_run.button(
            f"⚗️ Compute Derivatives — {len(pending_deriv)} Pending Days",
            type="primary", use_container_width=True,
            disabled=(len(pending_deriv) == 0))
        recomp_btn  = col_recomp.button(
            "🔄 Recompute All", use_container_width=True,
            help="Recomputes derivatives for ALL collected days (overwrites existing)")

        if recomp_btn:
            con = sqlite3.connect(DB_PATH)
            con.execute(
                "DELETE FROM derived_snapshots WHERE symbol=? AND expiry_code=? "
                "AND expiry_flag=? AND interval_min=?",
                (symbol, expiry_code, expiry_flag, interval_min))
            con.commit(); con.close()
            pending_deriv = sorted(raw_dates)
            st.info(f"Cleared — will recompute {len(pending_deriv)} days.")

        if run_btn or recomp_btn:
            prog   = st.progress(0)
            status = st.empty()
            log_bx = st.empty()
            logs   = []
            for idx, td in enumerate(pending_deriv):
                status.text(f"Computing {td}  ({idx+1}/{len(pending_deriv)})")
                df_day = load_raw_chain(symbol, td, expiry_code, expiry_flag, interval_min)
                if df_day.empty:
                    logs.append(f"⚠️  {td}  — no raw data found")
                else:
                    rows = compute_derivatives_for_day(
                        df_day, symbol, td, expiry_code, expiry_flag, interval_min)
                    save_derived(rows)
                    logs.append(f"✅  {td}  →  {len(rows)} snapshots")
                prog.progress((idx + 1) / max(len(pending_deriv), 1))
                log_bx.text("\n".join(logs[-20:]))

            prog.empty(); status.empty()
            st.markdown(
                '<div class="ok-box">✅ <b>Derivatives computation complete!</b> '
                'Go to <b>Preview</b> below or <b>Export</b> tab.</div>',
                unsafe_allow_html=True)
            st.rerun()

        # ── Preview derived snapshots ─────────────────────────────────────────
        if derived_dates:
            st.markdown("---")
            st.markdown("#### 🔍 Preview Derived Snapshots")

            prev_date = st.selectbox(
                "Select Date", sorted(derived_dates, reverse=True),
                key="deriv_prev_date")

            df_deriv = load_derived(
                symbol, prev_date, expiry_code, expiry_flag, interval_min)

            if df_deriv.empty:
                st.warning("No derived data for this date/settings.")
            else:
                # Metric group selector
                groups = {
                    "IV Metrics":     ["timestamp", "spot_price", "avg_iv", "atm_iv",
                                       "iv_skew", "iv_change", "iv_regime",
                                       "iv_term_structure"],
                    "OI Metrics":     ["timestamp", "spot_price", "total_call_oi",
                                       "total_put_oi", "pcr_oi", "pcr_volume",
                                       "max_pain", "call_oi_concentration",
                                       "put_oi_concentration", "oi_buildup_signal"],
                    "GEX Derivatives":["timestamp", "spot_price", "net_gex_total",
                                       "cumulative_gex_above", "cumulative_gex_below",
                                       "gex_flip_level", "gex_skew",
                                       "largest_gex_strike"],
                    "VANNA Zones":    ["timestamp", "spot_price", "net_vanna_total",
                                       "vacuum_zone_level", "trap_door_level",
                                       "support_floor_level", "resistance_ceil_level",
                                       "vanna_skew", "net_flow_vanna_total"],
                    "Cascade Math":   ["timestamp", "spot_price",
                                       "bear_fuel_pts", "bear_absorb_pts",
                                       "bull_fuel_pts", "bull_absorb_pts",
                                       "bear_quality", "bull_quality",
                                       "cascade_direction", "estimated_cascade_pts"],
                    "Charm":          ["timestamp", "spot_price", "net_charm_total",
                                       "iv_regime", "cascade_direction"],
                    "All":            [c for c in df_deriv.columns
                                       if c not in ("id",)],
                }
                grp_sel = st.selectbox("Metric group", list(groups.keys()))
                cols_sel = [c for c in groups[grp_sel] if c in df_deriv.columns]

                st.dataframe(
                    df_deriv[cols_sel],
                    use_container_width=True, height=350, hide_index=True)

                # ── Intraday charts ───────────────────────────────────────────
                st.markdown("#### 📈 Intraday Charts")

                chart_tabs = st.tabs([
                    "IV Regime", "GEX Flip", "VANNA Zones",
                    "PCR", "Cascade", "Charm"])

                with chart_tabs[0]:
                    fig_iv = go.Figure()
                    fig_iv.add_trace(go.Scatter(
                        x=df_deriv["timestamp"], y=df_deriv["avg_iv"],
                        name="Avg IV", line=dict(color="#00d4ff", width=2)))
                    fig_iv.add_trace(go.Scatter(
                        x=df_deriv["timestamp"], y=df_deriv["atm_iv"],
                        name="ATM IV", line=dict(color="#a78bfa", width=2,
                                                  dash="dash")))
                    # Colour background by iv_regime using a bar trace (avoids vrect rgba issues)
                    _regime_color_map = {
                        "EXPANDING":   "rgba(239,68,68,0.13)",
                        "COMPRESSING": "rgba(16,185,129,0.13)",
                        "FLAT":        "rgba(148,163,184,0.06)",
                    }
                    for regime, rcolor in _regime_color_map.items():
                        _mask = df_deriv["iv_regime"] == regime
                        if _mask.any():
                            fig_iv.add_trace(go.Bar(
                                x=df_deriv.loc[_mask, "timestamp"],
                                y=[df_deriv["avg_iv"].max() * 1.5] * _mask.sum(),
                                marker_color=rcolor,
                                marker_line_width=0,
                                width=[pd.Timedelta(minutes=4).total_seconds() * 1000] * _mask.sum(),
                                showlegend=False,
                                hoverinfo="skip",
                                yaxis="y",
                            ))
                    fig_iv.update_layout(barmode="overlay")
                    fig_iv.update_layout(
                        template="plotly_dark", height=350,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(10,10,20,0.95)",
                        yaxis_title="IV (%)",
                        legend=dict(orientation="h"),
                        margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_iv, use_container_width=True)
                    st.caption("🟥 EXPANDING  🟩 COMPRESSING  ⬜ FLAT")

                with chart_tabs[1]:
                    fig_gfl = go.Figure()
                    fig_gfl.add_trace(go.Scatter(
                        x=df_deriv["timestamp"], y=df_deriv["spot_price"],
                        name="Spot", line=dict(color="#00d4ff", width=2.5)))
                    if df_deriv["gex_flip_level"].notna().any():
                        fig_gfl.add_trace(go.Scatter(
                            x=df_deriv["timestamp"],
                            y=df_deriv["gex_flip_level"],
                            name="GEX Flip Level",
                            line=dict(color="#f59e0b", width=1.5, dash="dot")))
                    fig_gfl.add_trace(go.Scatter(
                        x=df_deriv["timestamp"], y=df_deriv["net_gex_total"],
                        name="Net GEX Total", yaxis="y2",
                        line=dict(color="#a78bfa", width=1.5),
                        opacity=0.7))
                    fig_gfl.update_layout(
                        template="plotly_dark", height=380,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(10,10,20,0.95)",
                        yaxis=dict(title="Price"),
                        yaxis2=dict(title="Net GEX (B)", overlaying="y",
                                    side="right", showgrid=False),
                        legend=dict(orientation="h"),
                        margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_gfl, use_container_width=True)

                with chart_tabs[2]:
                    fig_vz = go.Figure()
                    fig_vz.add_trace(go.Scatter(
                        x=df_deriv["timestamp"], y=df_deriv["spot_price"],
                        name="Spot", line=dict(color="#00d4ff", width=2.5)))
                    zone_cfg = {
                        "vacuum_zone_level":    ("Vacuum Zone (LOC)",   "#10b981"),
                        "trap_door_level":      ("Trap Door",           "#f59e0b"),
                        "support_floor_level":  ("Support Floor",       "#06b6d4"),
                        "resistance_ceil_level":("Resistance Ceiling",  "#ef4444"),
                    }
                    for col, (label, color) in zone_cfg.items():
                        if df_deriv[col].notna().any():
                            fig_vz.add_trace(go.Scatter(
                                x=df_deriv["timestamp"],
                                y=df_deriv[col],
                                name=label,
                                line=dict(color=color, width=1.5, dash="dash"),
                                mode="lines+markers",
                                marker=dict(size=5)))
                    fig_vz.update_layout(
                        template="plotly_dark", height=420,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(10,10,20,0.95)",
                        yaxis_title="Strike Level",
                        legend=dict(orientation="h"),
                        margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_vz, use_container_width=True)

                with chart_tabs[3]:
                    fig_pcr = go.Figure()
                    fig_pcr.add_trace(go.Scatter(
                        x=df_deriv["timestamp"], y=df_deriv["pcr_oi"],
                        name="PCR (OI)", line=dict(color="#ec4899", width=2)))
                    fig_pcr.add_trace(go.Scatter(
                        x=df_deriv["timestamp"], y=df_deriv["pcr_volume"],
                        name="PCR (Volume)",
                        line=dict(color="#f59e0b", width=1.5, dash="dash")))
                    fig_pcr.add_hline(y=1.0, line_dash="dot",
                                      line_color="#94a3b8",
                                      annotation_text="PCR = 1.0 (neutral)")
                    if df_deriv["max_pain"].notna().any():
                        fig_pcr.add_trace(go.Scatter(
                            x=df_deriv["timestamp"], y=df_deriv["max_pain"],
                            name="Max Pain", yaxis="y2",
                            line=dict(color="#00d4ff", width=1.5, dash="dot")))
                    fig_pcr.update_layout(
                        template="plotly_dark", height=350,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(10,10,20,0.95)",
                        yaxis=dict(title="PCR"),
                        yaxis2=dict(title="Max Pain Strike", overlaying="y",
                                    side="right", showgrid=False),
                        legend=dict(orientation="h"),
                        margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_pcr, use_container_width=True)

                with chart_tabs[4]:
                    clrs = df_deriv["cascade_direction"].map(
                        {"BEAR": "#ef4444", "BULL": "#10b981", "NONE": "#94a3b8"})
                    fig_cas = go.Figure()
                    fig_cas.add_trace(go.Bar(
                        x=df_deriv["timestamp"],
                        y=df_deriv["bear_fuel_pts"],
                        name="Bear Fuel", marker_color="#ef4444", opacity=0.8))
                    fig_cas.add_trace(go.Bar(
                        x=df_deriv["timestamp"],
                        y=df_deriv["bull_fuel_pts"],
                        name="Bull Fuel", marker_color="#10b981", opacity=0.8))
                    fig_cas.add_trace(go.Scatter(
                        x=df_deriv["timestamp"],
                        y=df_deriv["bear_quality"],
                        name="Bear Quality", yaxis="y2",
                        line=dict(color="#fbbf24", width=2)))
                    fig_cas.add_trace(go.Scatter(
                        x=df_deriv["timestamp"],
                        y=df_deriv["bull_quality"],
                        name="Bull Quality", yaxis="y2",
                        line=dict(color="#00f5c4", width=2)))
                    fig_cas.add_hline(y=0, line_color="#475569", line_width=1)
                    fig_cas.update_layout(
                        template="plotly_dark", height=380, barmode="group",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(10,10,20,0.95)",
                        yaxis=dict(title="Cascade Pts"),
                        yaxis2=dict(title="Quality Ratio", overlaying="y",
                                    side="right", showgrid=False),
                        legend=dict(orientation="h"),
                        margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_cas, use_container_width=True)

                with chart_tabs[5]:
                    fig_ch = go.Figure()
                    fig_ch.add_trace(go.Scatter(
                        x=df_deriv["timestamp"],
                        y=df_deriv["net_charm_total"],
                        name="Net Charm",
                        fill="tozeroy",
                        fillcolor="rgba(168,85,247,0.12)",
                        line=dict(color="#a78bfa", width=2)))
                    fig_ch.add_hline(y=0, line_dash="dot",
                                     line_color="#94a3b8")
                    fig_ch.update_layout(
                        template="plotly_dark", height=320,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(10,10,20,0.95)",
                        yaxis_title="Net Charm (B/day)",
                        margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_ch, use_container_width=True)
                    st.caption(
                        "Net Charm = total dealer delta-decay per day. "
                        "Large positive → expiry pin above spot. "
                        "Large negative → pin below spot.")

                # ── Export derived ────────────────────────────────────────────
                st.markdown("---")
                csv_d = df_deriv.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"⬇️ Download Derivatives CSV — {prev_date}",
                    data=csv_d,
                    file_name=f"hedgex_derived_{symbol}_{prev_date}.csv",
                    mime="text/csv",
                    use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 4 — Export
    # ═════════════════════════════════════════════════════════════════════════
    with tab_export:
        st.markdown("### 📥 Export Raw Data")

        st.markdown("""
        <div class="info-box">
        Export raw data to CSV for offline analysis, strategy backtesting,
        or feeding into the overnight strategy engine.
        </div>""", unsafe_allow_html=True)

        con = sqlite3.connect(DB_PATH)
        avail_exp = pd.read_sql_query(
            "SELECT DISTINCT symbol, trade_date FROM raw_chain ORDER BY symbol, trade_date",
            con)
        con.close()

        if avail_exp.empty:
            st.info("No data to export yet.")
        else:
            exp_cols = st.columns(3)
            exp_symbol = exp_cols[0].selectbox(
                "Symbol", sorted(avail_exp["symbol"].unique()), key="exp_sym")
            dates_for_sym = sorted(
                avail_exp[avail_exp["symbol"] == exp_symbol]["trade_date"].unique())
            exp_from = exp_cols[1].selectbox(
                "From Date", dates_for_sym, key="exp_from")
            exp_to   = exp_cols[2].selectbox(
                "To Date", dates_for_sym,
                index=len(dates_for_sym) - 1, key="exp_to")

            export_cols = st.multiselect(
                "Columns to export (leave empty = all)",
                options=[
                    "symbol", "trade_date", "timestamp",
                    "strike_type", "strike", "spot_price",
                    "call_oi", "put_oi", "call_oi_chg", "put_oi_chg",
                    "call_vol", "put_vol",
                    "call_iv", "put_iv",
                    "call_ltp", "put_ltp",
                    "call_gex", "put_gex", "net_gex",
                    "call_vanna", "put_vanna", "net_vanna",
                    "call_gamma", "put_gamma",
                    "call_vanna_greek", "put_vanna_greek",
                ],
                default=[])

            if st.button("📦 Prepare Export", type="primary",
                         use_container_width=True):
                con = sqlite3.connect(DB_PATH)
                df_exp = pd.read_sql_query("""
                    SELECT * FROM raw_chain
                    WHERE symbol=? AND trade_date>=? AND trade_date<=?
                    ORDER BY trade_date, timestamp, strike""",
                    con, params=(exp_symbol, exp_from, exp_to))
                con.close()

                if export_cols:
                    avail_ec = [c for c in export_cols if c in df_exp.columns]
                    df_exp   = df_exp[avail_ec]

                n_rows = len(df_exp)
                n_days = df_exp["trade_date"].nunique() if "trade_date" in df_exp else "?"
                st.markdown(
                    f'<div class="ok-box">✅ Ready: <b>{n_rows:,} rows</b> '
                    f'across <b>{n_days} days</b></div>',
                    unsafe_allow_html=True)

                csv_bytes = df_exp.to_csv(index=False).encode("utf-8")
                fname     = (f"hedgex_raw_{exp_symbol}_"
                             f"{exp_from}_to_{exp_to}.csv")
                st.download_button(
                    label=f"⬇️ Download {fname}",
                    data=csv_bytes,
                    file_name=fname,
                    mime="text/csv",
                    use_container_width=True)

                st.markdown("#### Preview (first 50 rows)")
                st.dataframe(df_exp.head(50),
                             use_container_width=True, hide_index=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 5 — API Inspector
    # ═════════════════════════════════════════════════════════════════════════
    with tab_inspect:
        st.markdown("### 🔬 API Response Inspector")
        st.markdown("""
        <div class="info-box">
        Test a single API call to verify token, expiry_code, and data availability
        before running a full collection.
        </div>""", unsafe_allow_html=True)

        ic1, ic2, ic3, ic4 = st.columns(4)
        dbg_date   = ic1.text_input(
            "Test Date (YYYY-MM-DD)",
            value=(date.today() - timedelta(days=5)).strftime("%Y-%m-%d"))
        dbg_strike = ic2.selectbox("Strike", ["ATM", "ATM+1", "ATM-1", "ATM+2", "ATM-2"],
                                   key="dbg_s")
        dbg_otype  = ic3.selectbox("Option Type", ["CALL", "PUT"], key="dbg_o")
        dbg_intv   = ic4.selectbox("Interval", ["5", "15", "60"],
                                   key="dbg_i",
                                   format_func=lambda x: f"{x} min")

        if st.button("🔍 Run API Test", type="primary"):
            try:
                dbg_dt   = datetime.strptime(dbg_date, "%Y-%m-%d").date()
                dbg_from = (dbg_dt - timedelta(days=2)).strftime("%Y-%m-%d")
                dbg_to   = (dbg_dt + timedelta(days=2)).strftime("%Y-%m-%d")

                with st.spinner(f"Calling API for {symbol} {dbg_strike} "
                                f"{dbg_otype} | {dbg_date}..."):
                    raw = fetch_rolling_option(
                        symbol, dbg_from, dbg_to,
                        dbg_strike, dbg_otype, dbg_intv,
                        expiry_code, expiry_flag, silent=False)

                if raw:
                    ce = raw.get("ce", {}) if dbg_otype == "CALL" else raw.get("pe", {})
                    key = "ce" if dbg_otype == "CALL" else "pe"
                    ts_list  = ce.get("timestamp", [])
                    all_keys = list(ce.keys())

                    # Filter to target date
                    match = [
                        t for t in ts_list
                        if datetime.fromtimestamp(t, tz=pytz.UTC)
                           .astimezone(IST).date() == dbg_dt
                    ]

                    st.markdown(
                        f'<div class="ok-box">'
                        f'✅ Response received<br>'
                        f'Keys in <code>{key}</code>: <b>{all_keys}</b><br>'
                        f'Total timestamps: <b>{len(ts_list)}</b><br>'
                        f'Matching <b>{dbg_date}</b>: <b>{len(match)}</b> bars'
                        f'</div>', unsafe_allow_html=True)

                    if ts_list:
                        # Show first 10 bars matching target date
                        show_ts = match[:10] if match else ts_list[:10]
                        preview_rows = []
                        for i, ts_ep in enumerate(show_ts):
                            orig_i = ts_list.index(ts_ep)
                            dt_ist = datetime.fromtimestamp(
                                ts_ep, tz=pytz.UTC).astimezone(IST)
                            row = {
                                "timestamp_IST": dt_ist.strftime("%Y-%m-%d %H:%M"),
                                "spot":   (ce.get("spot",   [])[orig_i]
                                           if orig_i < len(ce.get("spot", [])) else None),
                                "strike": (ce.get("strike", [])[orig_i]
                                           if orig_i < len(ce.get("strike", [])) else None),
                                "oi":     (ce.get("oi",     [])[orig_i]
                                           if orig_i < len(ce.get("oi", [])) else None),
                                "volume": (ce.get("volume", [])[orig_i]
                                           if orig_i < len(ce.get("volume", [])) else None),
                                "iv":     (ce.get("iv",     [])[orig_i]
                                           if orig_i < len(ce.get("iv", [])) else None),
                                "close":  (ce.get("close",  [])[orig_i]
                                           if orig_i < len(ce.get("close", [])) else None),
                            }
                            preview_rows.append(row)

                        st.dataframe(pd.DataFrame(preview_rows),
                                     use_container_width=True,
                                     hide_index=True)
                    else:
                        st.warning("No timestamps in response. "
                                   "Try a different date or expiry_code.")
                else:
                    st.error("Empty response. Check token validity or try "
                             "expiry_code=2 for a different expiry window.")

            except Exception as e:
                st.error(f"Error: {e}")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="text-align:center;padding:16px;'
        'font-family:JetBrains Mono,monospace;font-size:0.68rem;'
        'color:rgba(255,255,255,0.2);">'
        'HedGEX Raw Data Collector · NYZTrade Analytics · '
        'Research & Strategy Development Use Only'
        '</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
