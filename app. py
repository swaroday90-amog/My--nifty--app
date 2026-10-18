
import sqlite3
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="NT Pro Analytics", page_icon="📈", layout="wide")

DB_FILE = "oi_timeseries.db"

class NiftyTraderEngine:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/option-chain"
        }
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trending_oi (
                    timestamp TEXT, symbol TEXT, spot_price REAL,
                    sum_ce_chg_oi INTEGER, sum_pe_chg_oi INTEGER,
                    net_oi_diff INTEGER, pcr REAL, max_pain REAL
                )
            """)

    def fetch_live_nse_chain(self, symbol="NIFTY", allow_demo=True):
        try:
            self.session.get("https://www.nseindia.com", headers=self.headers, timeout=5)
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
            response = self.session.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                df, summary = self._parse_nse_json(response.json(), symbol)
                summary["is_live"] = True
                return df, summary
        except Exception:
            pass
        
        if not allow_demo:
            return None, {"is_live": False}
            
        df, summary = self._generate_simulated_chain(symbol)
        summary["is_live"] = False
        return df, summary

    def _parse_nse_json(self, data, symbol):
        records = data["records"]["data"]
        nearest_expiry = data["records"]["expiryDates"][0]
        spot_price = data["records"]["underlyingValue"]
        rows = []
        for item in records:
            if item.get("expiryDate") == nearest_expiry:
                ce, pe = item.get("CE", {}), item.get("PE", {})
                rows.append({
                    "strike": item["strikePrice"],
                    "ce_oi": ce.get("openInterest", 0),
                    "ce_chg_oi": ce.get("changeinOpenInterest", 0),
                    "ce_ltp": ce.get("lastPrice", 0.0),
                    "ce_chg_pts": ce.get("change", 0.0),
                    "pe_oi": pe.get("openInterest", 0),
                    "pe_chg_oi": pe.get("changeinOpenInterest", 0),
                    "pe_ltp": pe.get("lastPrice", 0.0),
                    "pe_chg_pts": pe.get("change", 0.0),
                })
        df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
        return self._enrich_quant_metrics(df, spot_price, symbol)

    def _generate_simulated_chain(self, symbol="NIFTY"):
        base_spot = 24850.0 if symbol == "NIFTY" else 52400.0
        step = 50 if symbol == "NIFTY" else 100
        spot_price = round(base_spot + np.random.normal(0, step * 0.4), 2)
        atm = round(spot_price / step) * step
        strikes = [atm + i * step for i in range(-12, 13)]
        rows = []
        for k in strikes:
            dist = (k - spot_price) / step
            ce_ltp = max(1.5, round(max(0, spot_price - k) + 120 * np.exp(-abs(dist)*0.25), 2))
            pe_ltp = max(1.5, round(max(0, k - spot_price) + 120 * np.exp(-abs(dist)*0.25), 2))
            ce_oi = int(max(10000, np.random.normal(150000, 40000) * (1.4 if k >= atm else 0.5)))
            pe_oi = int(max(10000, np.random.normal(165000, 40000) * (1.4 if k <= atm else 0.5)))
            rows.append({
                "strike": k,
                "ce_oi": ce_oi, "ce_chg_oi": int(ce_oi * np.random.uniform(-0.15, 0.35)),
                "ce_ltp": ce_ltp, "ce_chg_pts": round(np.random.uniform(-25, 25), 2),
                "pe_oi": pe_oi, "pe_chg_oi": int(pe_oi * np.random.uniform(-0.12, 0.40)),
                "pe_ltp": pe_ltp, "pe_chg_pts": round(np.random.uniform(-25, 25), 2),
            })
        return self._enrich_quant_metrics(pd.DataFrame(rows), spot_price, symbol)

    @staticmethod
    def classify_buildup(p, o):
        if p >= 0 and o >= 0: return "Long Build-up"
        if p < 0 and o >= 0: return "Short Build-up"
        if p >= 0 and o < 0: return "Short Covering"
        return "Long Unwinding"

    @staticmethod
    def calculate_max_pain(df):
        strikes, ce_oi, pe_oi = df["strike"].values, df["ce_oi"].values, df["pe_oi"].values
        payouts = [np.sum(np.maximum(0, s - strikes) * ce_oi) + np.sum(np.maximum(0, strikes - s) * pe_oi) for s in strikes]
        df["total_pain_payout"] = payouts
        return strikes[np.argmin(payouts)]

    def _enrich_quant_metrics(self, df, spot_price, symbol):
        df["ce_buildup"] = [self.classify_buildup(p, o) for p, o in zip(df["ce_chg_pts"], df["ce_chg_oi"])]
        df["pe_buildup"] = [self.classify_buildup(p, o) for p, o in zip(df["pe_chg_pts"], df["pe_chg_oi"])]
        df["strike_pcr"] = (df["pe_oi"] / df["ce_oi"].replace(0, np.nan)).round(2).fillna(0)
        max_pain = self.calculate_max_pain(df)
        total_pcr = round(df["pe_oi"].sum() / max(1, df["ce_oi"].sum()), 2)

        atm_idx = (df["strike"] - spot_price).abs().idxmin()
        near_atm = df.iloc[max(0, atm_idx - 5) : min(len(df), atm_idx + 6)]
        sum_ce_chg, sum_pe_chg = int(near_atm["ce_chg_oi"].sum()), int(near_atm["pe_chg_oi"].sum())
        net_diff = sum_pe_chg - sum_ce_chg
        now_str = datetime.now().strftime("%H:%M:%S")

        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("INSERT INTO trending_oi VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (now_str, symbol, spot_price, sum_ce_chg, sum_pe_chg, net_diff, total_pcr, max_pain))

        return df, {
            "symbol": symbol, "spot_price": spot_price, "pcr": total_pcr,
            "max_pain": max_pain, "net_oi_diff": net_diff, "timestamp": now_str
        }

    def get_trending_oi_history(self, symbol="NIFTY"):
        with sqlite3.connect(DB_FILE) as conn:
            return pd.read_sql_query(
                "SELECT * FROM trending_oi WHERE symbol = ? ORDER BY rowid DESC LIMIT 60",
                conn, params=(symbol,)
            ).iloc[::-1]

@st.cache_resource
def get_engine():
    return NiftyTraderEngine()

engine = get_engine()

st.subheader("📈 NT Pro Mobile Analytics")

c1, c2 = st.columns([2, 1])
symbol = c1.selectbox("Index", ["NIFTY", "BANKNIFTY"], label_visibility="collapsed")
if c2.button("🔄 Refresh", use_container_width=True):
    pass

# Safety toggle to allow/block simulated fallback data
allow_demo = st.checkbox("Allow Demo Mode if NSE blocks connection / Market Closed", value=True)

df, summary = engine.fetch_live_nse_chain(symbol, allow_demo=allow_demo)

# --- SAFETY STATUS BANNER ---
if summary.get("is_live"):
    st.success("🟢 **LIVE NSE DATA CONNECTED** — Showing real market numbers.")
else:
    st.error("🔴 **DEMO MODE (SIMULATED DATA - DO NOT TRADE)** — NSE blocked the cloud request or market is closed. Numbers below are randomly generated for UI testing!")
    if df is None:
        st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Spot", f"₹{summary['spot_price']:,.0f}")
m2.metric("PCR", f"{summary['pcr']}", "Bullish" if summary['pcr'] > 1 else "Bearish")
m3.metric("Max Pain", f"{summary['max_pain']:,.0f}")
m4.metric("Net OI (PE-CE)", f"{summary['net_oi_diff']:+,}")

tab1, tab2, tab3 = st.tabs(["📊 Chain", "📈 Trending OI", "🎯 Max Pain"])

with tab1:
    cols = ["ce_buildup", "ce_chg_oi", "ce_ltp", "strike", "pe_ltp", "pe_chg_oi", "pe_buildup"]
    st.dataframe(df[cols], use_container_width=True, height=450)

with tab2:
    hist_df = engine.get_trending_oi_history(symbol)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df["sum_ce_chg_oi"], name="Call Chg OI", line=dict(color="#ef4444", width=2)))
    fig.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df["sum_pe_chg_oi"], name="Put Chg OI", line=dict(color="#22c55e", width=2)))
    fig.update_layout(template="plotly_dark", height=360, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig_oi = go.Figure()
    fig_oi.add_trace(go.Bar(x=df["strike"], y=df["ce_oi"], name="Call OI", marker_color="#ef4444"))
    fig_oi.add_trace(go.Bar(x=df["strike"], y=df["pe_oi"], name="Put OI", marker_color="#22c55e"))
    fig_oi.update_layout(barmode="group", template="plotly_dark", height=360, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_oi, use_container_width=True)
