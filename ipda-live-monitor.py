# ipda_live_monitor.py
import time
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
import ccxt
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
from app import push_update          # <-- import the socket helper
from ipda_utils import engineer_ipda_features

# ----------------------------------------------------------------------
# CONFIG – keep the same keys you used before (pair, interval, etc.)
# ----------------------------------------------------------------------
CONFIG = {
    "data_source": "bitget",   # Options: "yahoo", "bitget", "metatrader"
    "pair": "BTC/USDT",       # e.g., "EURUSD=X" (yahoo), "BTC/USDT" (bitget), "EURUSD" (metatrader)
    "interval": "1d",
    "threshold": 0.35,
    "check_interval_sec": 60,
}
PAIR_LABEL = CONFIG["pair"].replace("=X", "").replace("/", "")

# ----------------------------------------------------------------------
# Load the trained model & feature list (produced by ipda_reversal_predictor.py)
# ----------------------------------------------------------------------
model = xgb.XGBClassifier()
model.load_model("ipda_model.json")
FEATURE_COLS = joblib.load("ipda_features.pkl")

# ----------------------------------------------------------------------
# Helper: fetch the most recent OHLCV bar (using yfinance for demo)
# ----------------------------------------------------------------------
def fetch_latest():
    source = CONFIG["data_source"]
    pair = CONFIG["pair"]
    interval = CONFIG["interval"]
    
    if source == "yahoo":
        df = yf.download(pair, period="5d", interval=interval, progress=False)
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df.dropna(inplace=True)
        return df
        
    elif source == "bitget":
        exchange = ccxt.bitget({'enableRateLimit': True})
        timeframe = interval.replace("d", "d").replace("h", "h").replace("m", "m")
        bars = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=200)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
        
    elif source == "metatrader":
        if not mt5.initialize():
            raise Exception("MT5 initialization failed")
            
        tf_map = {
            "1d": mt5.TIMEFRAME_D1,
            "1h": mt5.TIMEFRAME_H1,
            "15m": mt5.TIMEFRAME_M15,
            "5m": mt5.TIMEFRAME_M5,
            "1m": mt5.TIMEFRAME_M1
        }
        tf = tf_map.get(interval, mt5.TIMEFRAME_D1)
        
        rates = mt5.copy_rates_from_pos(pair, tf, 0, 200)
        if rates is None or len(rates) == 0:
            mt5.shutdown()
            raise Exception(f"Failed to fetch {pair} from MT5, or no data available.")
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        mt5.shutdown()
        return df
        
    else:
        raise ValueError(f"Unknown data source: {source}")

# ----------------------------------------------------------------------
# Main monitoring loop
# ----------------------------------------------------------------------
def monitor():
    print(f"🚀 Starting live monitor for {PAIR_LABEL} (threshold={CONFIG['threshold']})")
    while True:
        try:
            # 1️⃣ Get recent data
            df = fetch_latest()
            if len(df) < 70:          # need enough rows for the longest rolling window
                print("⏳ Not enough data yet – sleeping")
                time.sleep(CONFIG["check_interval_sec"])
                continue

            # 2️⃣ Engineer features (exact same windows as training)
            df_feat = engineer_ipda_features(df)
            latest = df_feat[FEATURE_COLS].iloc[[-1]].dropna()

            if latest.empty:
                print("⚠️ Feature mismatch – skipping this cycle")
                time.sleep(CONFIG["check_interval_sec"])
                continue

            # 3️⃣ Predict probability
            prob = model.predict_proba(latest.values)[0][1]

            # 4️⃣ Build payload for the dashboard
            payload = {
                "pair": PAIR_LABEL,
                "probability": float(prob),               # JSON‑serialisable
                "threshold": CONFIG["threshold"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # 5️⃣ Emit via SocketIO
            push_update(payload)

            # 6️⃣ (Optional) webhook or desktop notification – see previous answer
            # send_alert(...)

        except Exception as exc:
            print(f"❌ Monitor error: {exc}")

        # Wait until the next poll
        time.sleep(CONFIG["check_interval_sec"])

if __name__ == "__main__":
    monitor()