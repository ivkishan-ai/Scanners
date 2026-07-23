import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Safe Platform Check for Audio Alerts
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

st.set_page_config(page_title="Independent F&O Screener", layout="wide")

# Complete Cleaned NSE F&O Universe Watchlist + Nifty/Sensex
TICKERS = [
    "^NSEI", "^BSESN", 
    "AARTIIND.NS", "ABB.NS", "ABBOTINDIA.NS", "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS", 
    "ADANIENT.NS", "ADANIPORTS.NS", "ALKEM.NS", "AMBUJACEM.NS", "APOLLOHOSP.NS", 
    "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASIANPAINT.NS", "ASTRAL.NS", "ATUL.NS", 
    "AUBANK.NS", "AUROPHARMA.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", 
    "BAJFINANCE.NS", "BALKRISIND.NS", "BALRAMCHIN.NS", "BANDHANBNK.NS", "BANKBARODA.NS", 
    "BATAINDIA.NS", "BEL.NS", "BERGEPAINT.NS", "BHARATFORG.NS", "BHARTIARTL.NS", 
    "BHEL.NS", "BIOCON.NS", "BOSCHLTD.NS", "BPCL.NS", "BRITANNIA.NS", "CANBK.NS", 
    "CANFINHOME.NS", "CHAMBLFERT.NS", "CHOLAFIN.NS", "CIPLA.NS", "COALINDIA.NS", 
    "COFORGE.NS", "COLPAL.NS", "CONCOR.NS", "COROMANDEL.NS", "CROMPTON.NS", "CUB.NS", 
    "CUMMINSIND.NS", "DABUR.NS", "DALBHARAT.NS", "DEEPAKNTR.NS", "DIVISLAB.NS", 
    "DIXON.NS", "DLF.NS", "DRREDDY.NS", "EICHERMOT.NS", "ESCORTS.NS", "EXIDEIND.NS", 
    "FEDERALBNK.NS", "GAIL.NS", "GLENMARK.NS", "GNFC.NS", "GODREJCP.NS", 
    "GODREJPROP.NS", "GRANULES.NS", "GRASIM.NS", "HAL.NS", "HAVELLS.NS", 
    "HCLTECH.NS", "HDFCAMC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", 
    "HINDALCO.NS", "HINDCOPPER.NS", "HINDPETRO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", 
    "ICICIGI.NS", "ICICIPRULI.NS", "IDEA.NS", "IDFCFIRSTB.NS", "IEX.NS", "IGL.NS", 
    "INDHOTEL.NS", "INDIACEM.NS", "INDIAMART.NS", "INDIGO.NS", "INDUSINDBK.NS", 
    "INFY.NS", "INTELLECT.NS", "IOC.NS", "IPCALAB.NS", "IRCTC.NS", "ITC.NS", 
    "JINDALSTEL.NS", "JSWSTEEL.NS", "JUBLFOOD.NS", "KOTAKBANK.NS", 
    "LALPATHLAB.NS", "LAURUSLABS.NS", "LICHSGFIN.NS", "LT.NS", "LTTS.NS", 
    "LUPIN.NS", "M&M.NS", "M&MFIN.NS", "MANAPPURAM.NS", "MARICO.NS", "MARUTI.NS", 
    "MCX.NS", "METROPOLIS.NS", "MFSL.NS", "MGL.NS", "MOTHERSON.NS", 
    "MPHASIS.NS", "MRF.NS", "MUTHOOTFIN.NS", "NATIONALUM.NS", "NAUKRI.NS", 
    "NAVINFLUOR.NS", "NESTLEIND.NS", "NMDC.NS", "NTPC.NS", "OBEROIRLTY.NS", "OFSS.NS", 
    "ONGC.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PFC.NS", 
    "PIDILITIND.NS", "PIIND.NS", "PNB.NS", "POLYCAB.NS", "POWERGRID.NS", "PVRINOX.NS", 
    "RAMCOCEM.NS", "RBLBANK.NS", "RECLTD.NS", "RELIANCE.NS", "SAIL.NS", "SBICARD.NS", 
    "SBILIFE.NS", "SBIN.NS", "SHREECEM.NS", "SIEMENS.NS", "SRF.NS", "SUNTV.NS", 
    "SUNPHARMA.NS", "SYNGENE.NS", "TATACHEM.NS", "TATACOMM.NS", "TATACONSUM.NS", 
    "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", 
    "TORNTPHARM.NS", "TORNTPOWER.NS", "TRENT.NS", "TVSMOTOR.NS", "UBL.NS", 
    "ULTRACEMCO.NS", "UPL.NS", "VEDL.NS", "VOLTAS.NS", "WIPRO.NS", "ZEEL.NS", "ZYDUSLIFE.NS"
]

@st.cache_data(show_spinner=False, ttl=300)
def fetch_daily_data(tickers):
    return yf.download(tickers, period="40d", interval="1d", group_by="ticker", progress=False)

@st.cache_data(show_spinner=False, ttl=300)
def fetch_market_data(tickers, timeframe):
    return yf.download(tickers, period="20d", interval=timeframe, group_by="ticker", progress=False)

def calculate_rsi(data, length=14):
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=length-1, adjust=False).mean()
    ema_down = down.ewm(com=length-1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def run_scanners(data_intra, data_daily, tickers, min_daily_vol, min_daily_turnover_cr, strict_trend):
    ema_rows, rsi_pullback_rows, new_strat_rows = [], [], []
    has_multiple = len(tickers) > 1
    
    for ticker in tickers:
        try:
            if has_multiple:
                if (ticker not in data_daily.columns.levels[0] or 
                    ticker not in data_intra.columns.levels[0]):
                    continue
            
            # ==========================================
            # 1. GATEKEEPERS (Bypass for Indices)
            # ==========================================
            df_daily = data_daily.dropna() if not has_multiple else data_daily[ticker].dropna()
            if df_daily.empty or len(df_daily) < 20: continue
            
            is_index = ticker.startswith("^")
            if not is_index:
                avg_daily_vol = df_daily['Volume'].tail(20).mean()
                daily_close = df_daily['Close'].iloc[-1]
                avg_daily_turnover_cr = (avg_daily_vol * daily_close) / 10000000
                if avg_daily_vol < min_daily_vol or avg_daily_turnover_cr < min_daily_turnover_cr: 
                    continue 

            # ==========================================
            # MICRO SETUP 
            # ==========================================
            df = data_intra.dropna() if not has_multiple else data_intra[ticker].dropna()
            if df.empty or len(df) < 50: continue
                
            clean_name = ticker.replace(".NS", "").replace("^", "")
            c = df['Close'].iloc[-1]

            df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
            for ema_period in [9, 21, 50, 100, 200]:
                df[f'EMA{ema_period}'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
                
            df['RSI'] = calculate_rsi(df['Close'], 14)
            current_rsi = df['RSI'].iloc[-1]
            
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            
            macd_ascending = (df['MACD'].iloc[-1] > df['MACD'].iloc[-2]) and (df['MACD'].iloc[-2] > df['MACD'].iloc[-3])
            macd_descending = (df['MACD'].iloc[-1] < df['MACD'].iloc[-2]) and (df['MACD'].iloc[-2] < df['MACD'].iloc[-3])
            macd_trend_str = "Ascending 📈" if macd_ascending else ("Descending 📉" if macd_descending else "Flat ➡️")

            # ==========================================
            # SCREENER 1: RSI 50 & 5 EMA WITH EXITS
            # ==========================================
            strat_signal, strat_action, bars_ago = "Neutral", "-", 999
            
            # Locate all RSI crossovers across the 50 line
            rsi_bull_crosses = np.where((df['RSI'] > 50) & (df['RSI'].shift(1) <= 50))[0]
            rsi_bear_crosses = np.where((df['RSI'] < 50) & (df['RSI'].shift(1) >= 50))[0]
            
            # Calculate bars since the most recent crosses
            bars_since_rsi_bull = len(df) - 1 - rsi_bull_crosses[-1] if len(rsi_bull_crosses) > 0 else 999
            bars_since_rsi_bear = len(df) - 1 - rsi_bear_crosses[-1] if len(rsi_bear_crosses) > 0 else 999
            
            # Check previous bar state to detect live Exits
            prev_rsi = df['RSI'].iloc[-2]
            prev_c = df['Close'].iloc[-2]
            prev_ema5 = df['EMA5'].iloc[-2]
            
            was_long_valid = (prev_rsi >= 50) and (prev_c >= prev_ema5)
            was_short_valid = (prev_rsi <= 50) and (prev_c <= prev_ema5)
            
            # Trigger: Fresh Buy/Sell Entries
            if (current_rsi > 50) and (c > df['EMA5'].iloc[-1]) and (bars_since_rsi_bull <= 5):
                strat_signal = "🟢 RSI > 50 & Price > 5 EMA"
                strat_action = "BUY 🟢"
                bars_ago = bars_since_rsi_bull
                
            elif (current_rsi < 50) and (c < df['EMA5'].iloc[-1]) and (bars_since_rsi_bear <= 5):
                strat_signal = "🔴 RSI < 50 & Price < 5 EMA"
                strat_action = "SELL 🔴"
                bars_ago = bars_since_rsi_bear
                
            # Trigger: Exit Signals (If no new entry is firing, check if an old setup just broke)
            elif was_long_valid and ((current_rsi < 50) or (c < df['EMA5'].iloc[-1])):
                strat_signal = "⚠️ Price fell below 5 EMA or RSI < 50"
                strat_action = "EXIT LONG ⚠️"
                bars_ago = 0
                
            elif was_short_valid and ((current_rsi > 50) or (c > df['EMA5'].iloc[-1])):
                strat_signal = "⚠️ Price broke above 5 EMA or RSI > 50"
                strat_action = "EXIT SHORT ⚠️"
                bars_ago = 0
                
            if strat_signal != "Neutral":
                new_strat_rows.append({
                    "Symbol": clean_name, "LTP": round(c, 2), "Action": strat_action,
                    "Signal": strat_signal, "Bars Since RSI Cross": bars_ago,
                    "RSI": round(current_rsi, 1), "5 EMA": round(df['EMA5'].iloc[-1], 2)
                })

            # ==========================================
            # SCREENER 2: EMA TREND CROSSOVER 
            # ==========================================
            bull_crosses = np.where((df['EMA9'] > df['EMA21']) & (df['EMA9'].shift(1) <= df['EMA21'].shift(1)))[0]
            bear_crosses = np.where((df['EMA9'] < df['EMA21']) & (df['EMA9'].shift(1) >= df['EMA21'].shift(1)))[0]
            
            bars_since_bull_cross = len(df) - 1 - bull_crosses[-1] if len(bull_crosses) > 0 else 999
            bars_since_bear_cross = len(df) - 1 - bear_crosses[-1] if len(bear_crosses) > 0 else 999
            
            e9, e21, e50, e200 = df['EMA9'].iloc[-1], df['EMA21'].iloc[-1], df['EMA50'].iloc[-1], df['EMA200'].iloc[-1]
            ema_signal, cross_age_int = "Neutral", 999
            
            if strict_trend:
                if c > e200 and e9 > e21 and bars_since_bull_cross <= 5:
                    ema_signal, cross_age_int = "🟢 Bullish Setup", bars_since_bull_cross
                elif c < e200 and e9 < e21 and bars_since_bear_cross <= 5:
                    ema_signal, cross_age_int = "🔴 Bearish Setup", bars_since_bear_cross
            else:
                if e9 > e21 and bars_since_bull_cross <= 5:
                    ema_signal, cross_age_int = "🟢 Bullish Setup", bars_since_bull_cross
                elif e9 < e21 and bars_since_bear_cross <= 5:
                    ema_signal, cross_age_int = "🔴 Bearish Setup", bars_since_bear_cross

            if ema_signal != "Neutral":
                ema_rows.append({
                    "Symbol": clean_name, "LTP": round(c, 2), "Signal": ema_signal,
                    "Bars Since Cross": cross_age_int, "MACD Trend": macd_trend_str
                })

            # ==========================================
            # SCREENER 3: RSI TREND PULLBACK (30/70)
            # ==========================================
            rsi_pb_signal, rsi_pb_action = "Neutral", "-"
            if not pd.isna(current_rsi) and not pd.isna(e50):
                if c > e50 and current_rsi < 30:
                    rsi_pb_signal = "🔥 Bullish Pullback (Above 50 EMA & RSI Oversold)"
                    rsi_pb_action = "BUY 🟢"
                elif c < e50 and current_rsi > 70:
                    rsi_pb_signal = "🚨 Bearish Pullback (Below 50 EMA & RSI Overbought)"
                    rsi_pb_action = "SELL 🔴"
                
            if rsi_pb_signal != "Neutral":
                rsi_pullback_rows.append({
                    "Symbol": clean_name, "LTP": round(c, 2), "Action": rsi_pb_action,
                    "Signal": rsi_pb_signal, "RSI": round(current_rsi, 1),
                    "50 EMA": round(e50, 2), "MACD Trend": macd_trend_str
                })

        except Exception as e:
            continue
            
    return pd.DataFrame(ema_rows), pd.DataFrame(rsi_pullback_rows), pd.DataFrame(new_strat_rows)


# --- UI Dashboard Setup ---
st.title("⚡ Pro Options Live Execution Dashboard")

st.sidebar.header("⚙️ Global Settings")
selected_tf = st.sidebar.selectbox("Select Timeframe", options=["5m", "15m"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("Liquidity Gatekeepers (F&O Stocks only)")
min_daily_vol = st.sidebar.number_input("Min Average Daily Volume", min_value=100000, value=1000000, step=100000)
min_daily_turnover_cr = st.sidebar.number_input("Min Daily Turnover (₹ Cr)", min_value=1.0, value=250.0, step=50.0)

st.sidebar.markdown("---")
st.sidebar.header("EMA Trend Settings")
strict_trend = st.sidebar.checkbox("Require 200 EMA Trend Alignment (for EMA Cross)", value=True)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Force Data Refresh"):
    fetch_daily_data.clear()
    fetch_market_data.clear() 
    st.rerun()

with st.spinner(f"Loading Live Market Data..."):
    raw_daily = fetch_daily_data(TICKERS)
    raw_intra = fetch_market_data(TICKERS, selected_tf)
    
    ema_df, rsi_pb_df, new_strat_df = run_scanners(
        raw_intra, raw_daily, TICKERS, 
        min_daily_vol, min_daily_turnover_cr, strict_trend
    )
    
    if not ema_df.empty or not rsi_pb_df.empty or not new_strat_df.empty:
        if HAS_WINSOUND:
            try:
                winsound.Beep(1000, 600) 
                winsound.Beep(1500, 400)
            except:
                pass 
        st.toast("🚨 Active setups found on current timeframe.", icon="🔔")

    tab1, tab2, tab3 = st.tabs([
        "🎯 RSI 50 & 5 EMA Logic",
        "📈 9/21 EMA Trend", 
        "🔄 50 EMA Trend Pullback"
    ])
    
    with tab1:
        st.subheader("RSI 50 Crossover + 5 EMA Validation & Exits")
        if not new_strat_df.empty:
            st.dataframe(new_strat_df.sort_values(by="Bars Since RSI Cross"), width="stretch")
        else:
            st.info("No fresh triggers or exit signals found. Waiting for conditions to align or break.")

    with tab2:
        st.subheader("Recent Momentum Crossovers (9 & 21 EMA)")
        if not ema_df.empty:
            st.dataframe(ema_df.sort_values(by="Bars Since Cross"), width="stretch")
        else:
            st.info("No fresh EMA setups matching criteria.")
            
    with tab3:
        st.subheader("RSI Trend Pullback (Price vs 50 EMA + 30/70 Exhaustion)")
        if not rsi_pb_df.empty:
            st.dataframe(rsi_pb_df.sort_values(by="Action"), width="stretch")
        else:
            st.info("No symbols currently resting inside the 50 EMA boundary exhaustion zones.")
