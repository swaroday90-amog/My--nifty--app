import sqlite3
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="NT Pro F&O Scanner", page_icon="⚡", layout="wide")

DB_FILE = "oi_timeseries.db"

# Full Universe: NSE Indices, Top NSE F&O Stocks, and MCX Commodities
INSTRUMENTS = {
    "NSE Indices": {
        "NIFTY": {"step": 50, "base": 24850.0, "type": "index"},
        "BANKNIFTY": {"step": 100, "base": 52400.0, "type": "index"},
        "FINNIFTY": {"step": 50, "base": 23900.0, "type": "index"},
        "MIDCPNIFTY": {"step": 25, "base": 13100.0, "type": "index"},
    },
    "NSE F&O Stocks": {
        "RELIANCE": {"step": 20, "base": 2950.0, "type": "equities"},
        "HDFCBANK": {"step": 20, "base": 1720.0, "type": "equities"},
        "ICICIBANK": {"step": 20, "base": 1280.0, "type": "equities"},
        "INFY": {"step": 20, "base": 1920.0, "type": "equities"},
        "TCS": {"step": 50, "base": 4150.0, "type": "equities"},
        "SBIN": {"step": 10, "base": 820.0, "type": "equities"},
        "TATAMOTORS": {"step": 10, "base": 940.0, "type": "equities"},
        "BAJFINANCE": {"step": 100, "base": 7200.0, "type": "equities"},
        "AXISBANK": {"step": 20, "base": 1190.0, "type": "equities"},
        "LT": {"step": 50, "base": 3650.0, "type": "equities"},
        "BHARTIARTL": {"step": 20, "base": 1680.0, "type": "equities"},
        "ITC": {"step": 5, "base": 505.0, "type": "equities"},
    },
    "MCX Commodities": {
        "CRUDEOIL": {"step": 50, "base": 6150.0, "type": "mcx"},
        "NATURALGAS": {"step": 5, "base": 235.0, "type": "mcx"},
        "GOLD": {"step": 500, "base": 76500.0, "type": "mcx"},
        "SILVER": {"step": 1000, "base": 92000.0, "type": "mcx"},
        "COPPER": {"step": 5, "base": 845.0, "type": "mcx"},
        "ZINC": {"step": 2.5, "base": 285.0, "type": "mcx"},
        "ALUMINIUM": {"step": 2.5, "base": 240.0, "type": "mcx"},
        "LEAD": {"step": 2.5, "base": 185.0, "type": "mcx"},
    }
}

class NiftyTraderEngine:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
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

    def fetch_chain(self, segment, symbol, allow_demo=True):
        inst_type = INSTRUMENTS[segment][symbol]["type"]
        if inst_type in ["index", "equities"]:
            try:
                headers = {**self.headers, "Referer": "https://www.nseindia.com/option-chain"}
                self.session.get("https://www.nseindia.com", headers=headers, timeout=4)
                endpoint = "option-chain-indices" if inst_type == "index" else "option-chain-equities"
                url = f"https://www.nseindia.com/api/{endpoint}?symbol={symbol}"
                response = self.session.get(url, headers=headers, timeout=4)
                if response.status_code == 200:
                    df, summary = self._parse_nse_json(response.json(), segment, symbol)
                    summary["is_live"] = True
                    return df, summary
            except Exception:
                pass
        elif inst_type == "mcx":
            try:
                headers = {
                    **self.headers,
                    "Referer": "https://www.mcxindia.com/market-data/option-chain",
                    "Origin": "https://www.mcxindia.com",
                    "X-Requested-With": "XMLHttpRequest"
                }
                self.session.get("https://www.mcxindia.com/market-data/option-chain", headers=headers, timeout=4)
                url = "https://www.mcxindia.com/backpage.aspx/GetOptionChain"
                response = self.session.post(url, json={"Commodity": symbol, "Expiry": ""}, headers=headers, timeout=4)
                if response.status_code == 200 and "d" in response.json():
                    df, summary = self._parse_mcx_json(response.json()["d"], segment, symbol)
                    if not df.empty:
                        summary["is_live"] = True
                        return df, summary
            except Exception:
                pass

        if not allow_demo:
            return None, {"is_live": False}

        df, summary = self._generate_simulated_chain(segment, symbol)
        summary["is_live"] = False
        return df, summary

    def _parse_nse_json(self, data, segment, symbol):
        records = data["records"]["data"]
        nearest_expiry = data["records"]["expiryDates"][0]
        spot_price = float(data["records"]["underlyingValue"])
        rows = []
        for item in records:
            if item.get("expiryDate") == nearest_expiry:
                ce, pe = item.get("CE", {}), item.get("PE", {})
                rows.append({
                    "strike": float(item["strikePrice"]),
                    "ce_oi": int(ce.get("openInterest", 0)),
                    "ce_chg_oi": int(ce.get("changeinOpenInterest", 0)),
                    "ce_ltp": float(ce.get("lastPrice", 0.0)),
                    "ce_chg_pts": float(ce.get("change", 0.0)),
                    "pe_oi": int(pe.get("openInterest", 0)),
                    "pe_chg_oi": int(pe.get("changeinOpenInterest", 0)),
                    "pe_ltp": float(pe.get("lastPrice", 0.0)),
                    "pe_chg_pts": float(pe.get("change", 0.0)),
                })
        df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
        return self._enrich_quant_metrics(df, spot_price, segment, symbol)

    def _parse_mcx_json(self, data, segment, symbol):
        records = data.get("Data", [])
        if not records:
            return pd.DataFrame(), {}
        spot_price = float(records[0].get("UnderlyingValue", 0.0))
        rows = []
        for item in records:
            rows.append({
                "strike": float(item.get("StrikePrice", 0)),
                "ce_oi": int(item.get("CE_OpenInterest", 0) or 0),
                "ce_chg_oi": int(item.get("CE_ChangeInOI", 0) or 0),
                "ce_ltp": float(item.get("CE_LTP", 0.0) or 0.0),
                "ce_chg_pts": float(item.get("CE_NetChange", 0.0) or 0.0),
                "pe_oi": int(item.get("PE_OpenInterest", 0) or 0),
                "pe_chg_oi": int(item.get("PE_ChangeInOI", 0) or 0),
                "pe_ltp": float(item.get("PE_LTP", 0.0) or 0.0),
                "pe_chg_pts": float(item.get("PE_NetChange", 0.0) or 0.0),
            })
        df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
        return self._enrich_quant_metrics(df, spot_price, segment, symbol)

    def _generate_simulated_chain(self, segment, symbol):
        cfg = INSTRUMENTS[segment][symbol]
        base_spot, step = cfg["base"], cfg["step"]
        spot_price = round(base_spot + np.random.normal(0, step * 0.4), 2)
        atm = round(spot_price / step) * step
        strikes = [round(atm + i * step, 2) for i in range(-10, 11)]
        rows = []
        for k in strikes:
            dist = (k - spot_price) / step
            base_prem = step * 1.5
            ce_ltp = max(0.5, round(max(0, spot_price - k) + base_prem * np.exp(-abs(dist)*0.25), 2))
            pe_ltp = max(0.5, round(max(0, k - spot_price) + base_prem * np.exp(-abs(dist)*0.25), 2))
            base_oi = 25000 if segment != "NSE Indices" else 150000
            ce_oi = int(max(500, np.random.normal(base_oi, base_oi*0.25) * (1.4 if k >= atm else 0.5)))
            pe_oi = int(max(500, np.random.normal(base_oi*1.05, base_oi*0.25) * (1.4 if k <= atm else 0.5)))
            rows.append({
                "strike": k,
                "ce_oi": ce_oi, "ce_chg_oi": int(ce_oi * np.random.uniform(-0.18, 0.35)),
                "ce_ltp": ce_ltp, "ce_chg_pts": round(np.random.uniform(-step*0.25, step*0.25), 2),
                "pe_oi": pe_oi, "pe_chg_oi": int(pe_oi * np.random.uniform(-0.18, 0.35)),
                "pe_ltp": pe_ltp, "pe_chg_pts": round(np.random.uniform(-step*0.25, step*0.25), 2),
            })
        return self._enrich_quant_metrics(pd.DataFrame(rows), spot_price, segment, symbol)

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

    def _enrich_quant_metrics(self, df, spot_price, segment, symbol):
        df["ce_buildup"] = [self.classify_buildup(p, o) for p, o in zip(df["ce_chg_pts"], df["ce_chg_oi"])]
        df["pe_buildup"] = [self.classify_buildup(p, o) for p, o in zip(df["pe_chg_pts"], df["pe_chg_oi"])]
        df["strike_pcr"] = (df["pe_oi"] / df["ce_oi"].replace(0, np.nan)).round(2).fillna(0)
        max_pain = self.calculate_max_pain(df)
        total_pcr = round(df["pe_oi"].sum() / max(1, df["ce_oi"].sum()), 2)

        atm_idx = (df["strike"] - spot_price).abs().idxmin()
        near_atm = df.iloc[max(0, atm_idx - 5) : min(len(df), atm_idx + 6)]
        sum_ce_chg, sum_pe_chg = int(near_atm["ce_chg_oi"].sum()), int(near_atm["pe_chg_oi"].sum())
        net_diff = sum_pe_chg - sum_ce_chg
        
        # Synthetic Futures Price & OI Change Proxy from ATM + Net Option Flow
        base_spot = INSTRUMENTS[segment][symbol]["base"]
        fut_price_chg_pct = round(((spot_price - base_spot) / base_spot) * 100, 2)
        fut_oi_chg_pct = round((net_diff / max(1, df["ce_oi"].sum() + df["pe_oi"].sum())) * 100, 2)
        fut_buildup = self.classify_buildup(fut_price_chg_pct, fut_oi_chg_pct)

        now_str = datetime.now().strftime("%H:%M:%S")
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("INSERT INTO trending_oi VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (now_str, symbol, spot_price, sum_ce_chg, sum_pe_chg, net_diff, total_pcr, max_pain))

        return df, {
            "segment": segment, "symbol": symbol, "spot_price": spot_price,
            "fut_chg_pct": fut_price_chg_pct, "fut_oi_chg_pct": fut_oi_chg_pct,
            "fut_buildup": fut_buildup, "pcr": total_pcr, "max_pain": max_pain,
            "net_oi_diff": net_diff, "option_signal": "Bullish" if total_pcr >= 1.05 and net_diff > 0 else ("Bearish" if total_pcr <= 0.95 and net_diff < 0 else "Neutral"),
            "timestamp": now_str
        }

    def scan_market(self, selected_segments, allow_demo=True):
        results = []
        for seg in selected_segments:
            for sym in INSTRUMENTS[seg].keys():
                _, summary = self.fetch_chain(seg, sym, allow_demo=allow_demo)
                if summary and "symbol" in summary:
                    results.append(summary)
        return pd.DataFrame(results)

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

st.subheader("⚡ NT Pro: All-in-One NSE & MCX F&O Scanner")
allow_demo = st.checkbox("Allow Demo Mode if Exchange blocks cloud IP / Market Closed", value=True)

main_tab1, main_tab2 = st.tabs(["🔍 Full Market F&O Scanner", "📊 Deep Option Chain & Trending OI"])

# === MAIN TAB 1: MULTI-ASSET SCANNER ===
with main_tab1:
    st.markdown("**Scan Futures & Options across NSE Indices, NSE Stocks & MCX Commodities**")
    seg_filter = st.multiselect(
        "Select Segments to Scan",
        ["NSE Indices", "NSE F&O Stocks", "MCX Commodities"],
        default=["NSE Indices", "NSE F&O Stocks", "MCX Commodities"]
    )
    buildup_filter = st.selectbox(
        "Filter by Futures & OI Build-Up",
        ["All", "Long Build-up", "Short Build-up", "Short Covering", "Long Unwinding"]
    )
    
    if st.button("🚀 Run Full Market Scan", use_container_width=True):
        pass

    if seg_filter:
        scan_df = engine.scan_market(seg_filter, allow_demo=allow_demo)
        if not scan_df.empty:
            if scan_df["is_live"].any():
                st.success("🟢 **LIVE EXCHANGE DATA CONNECTED**")
            else:
                st.error("🔴 **DEMO MODE (SIMULATED DATA - DO NOT TRADE)** — Exchange blocked cloud IP or market is closed.")

            if buildup_filter != "All":
                scan_df = scan_df[scan_df["fut_buildup"] == buildup_filter]

            display_scan = scan_df[[
                "segment", "symbol", "spot_price", "fut_chg_pct",
                "fut_buildup", "pcr", "max_pain", "net_oi_diff", "option_signal"
            ]].rename(columns={
                "segment": "Segment", "symbol": "Symbol", "spot_price": "Spot/Fut",
                "fut_chg_pct": "Price %", "fut_buildup": "F&O Build-Up",
                "pcr": "PCR", "max_pain": "Max Pain", "net_oi_diff": "Net OI (PE-CE)",
                "option_signal": "OI Sentiment"
            })
            st.dataframe(display_scan, use_container_width=True, height=480)

# === MAIN TAB 2: DEEP OPTION CHAIN & TRENDING OI ===
with main_tab2:
    c1, c2, c3 = st.columns([2, 2, 1])
    segment = c1.selectbox("Segment", list(INSTRUMENTS.keys()))
    symbol = c2.selectbox("Symbol", list(INSTRUMENTS[segment].keys()))
    if c3.button("🔄 Refresh", use_container_width=True):
        pass

    df, summary = engine.fetch_chain(segment=segment, symbol=symbol, allow_demo=allow_demo)

    if df is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{symbol} Spot", f"₹{summary['spot_price']:,.1f}", f"{summary['fut_chg_pct']:+.2f}%")
        m2.metric("PCR", f"{summary['pcr']}", summary["option_signal"])
        m3.metric("Max Pain", f"{summary['max_pain']:,.1f}")
        m4.metric("F&O Build-Up", summary["fut_buildup"])

        sub1, sub2, sub3 = st.tabs(["📊 Option Chain", "📈 Trending OI", "🎯 Max Pain"])
        with sub1:
            cols = ["ce_buildup", "ce_chg_oi", "ce_ltp", "strike", "pe_ltp", "pe_chg_oi", "pe_buildup"]
            st.dataframe(df[cols], use_container_width=True, height=420)
        with sub2:
            hist_df = engine.get_trending_oi_history(symbol)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df["sum_ce_chg_oi"], name="Call Chg OI", line=dict(color="#ef4444", width=2)))
            fig.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df["sum_pe_chg_oi"], name="Put Chg OI", line=dict(color="#22c55e", width=2)))
            fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with sub3:
            fig_oi = go.Figure()
            fig_oi.add_trace(go.Bar(x=df["strike"], y=df["ce_oi"], name="Call OI", marker_color="#ef4444"))
            fig_oi.add_trace(go.Bar(x=df["strike"], y=df["pe_oi"], name="Put OI", marker_color="#22c55e"))
            fig_oi.update_layout(barmode="group", template="plotly_dark", height=350, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_oi, use_container_width=True)
