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
import socketio
from datetime import datetime, timezone, timedelta
from ipda_utils import engineer_ipda_features

# ----------------------------------------------------------------------
# CONFIG – will be updated dynamically via SocketIO
# ----------------------------------------------------------------------
CONFIG = {
    "data_source": "bitget",
    "pair": "BTC/USDT",
    "interval": "1d",
    "threshold": 0.35,
    "check_interval_sec": 60,
    "model_path": "ipda_model.json"
}

# ----------------------------------------------------------------------
# SocketIO Client Setup
# ----------------------------------------------------------------------
sio = socketio.Client()

@sio.on('config_updated')
def on_config_updated(data):
    global CONFIG, model, FEATURE_COLS
    print(f"⚙️ Config update received: {data}")

    # Reload model if path changed
    old_model_path = CONFIG.get("model_path")
    CONFIG.update(data)

    if CONFIG.get("model_path") != old_model_path:
        load_model_and_features()

def load_model_and_features():
    global model, FEATURE_COLS
    try:
        model = xgb.XGBClassifier()
        model.load_model(CONFIG["model_path"])
        FEATURE_COLS = joblib.load("ipda_features.pkl")
        print(f"✅ Model loaded from {CONFIG['model_path']}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

# Initial load
load_model_and_features()

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
    # Connect to the local Flask server
    try:
        sio.connect('http://localhost:5000')
        print("🔌 Connected to dashboard server")
    except Exception as e:
        print(f"❌ Failed to connect to dashboard server: {e}")
        # Continue anyway, will just print to console

    print(f"🚀 Starting live monitor")

    while True:
        pair_label = CONFIG["pair"].replace("=X", "").replace("/", "")
        try:
            # 1️⃣ Get recent data
            df = fetch_latest()
            if len(df) < 70:          # need enough rows for the longest rolling window
                print(f"⏳ Not enough data yet for {pair_label} – sleeping")
                time.sleep(CONFIG["check_interval_sec"])
                continue

            # 2️⃣ Engineer features (exact same windows as training)
            df_feat = engineer_ipda_features(df)
            latest = df_feat[FEATURE_COLS].iloc[[-1]].dropna()

            if latest.empty:
                print(f"⚠️ Feature mismatch for {pair_label} – skipping this cycle")
                time.sleep(CONFIG["check_interval_sec"])
                continue

            # 3️⃣ Predict probability
            prob = model.predict_proba(latest.values)[0][1]

            # Enrich payload with latest indicator values for the detailed dashboard
            latest_row = df_feat.iloc[-1]

            # 4️⃣ Build payload for the dashboard
            payload = {
                "pair": pair_label,
                "probability": float(prob),
                "threshold": CONFIG["threshold"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "price": float(latest_row['close']),
                "rsi": float(latest_row['rsi_14']),
                "atr_pct": float(latest_row['atr_pct']),
                "mom5": float(latest_row['momentum_5']),
                "h20": float(latest_row['ipda_20d_high']),
                "l20": float(latest_row['ipda_20d_low']),
                "pos20": float(latest_row['ipda_20d_pos']),
                "bull_fvg": int(latest_row['bull_fvg']),
                "bear_fvg": int(latest_row['bear_fvg']),
                "mss_bullish": int(latest_row['mss_bullish']),
                "mss_bearish": int(latest_row['mss_bearish']),
                "near_bull_ob": int(latest_row['near_bull_ob']),
                "swing_high": int(latest_row['swing_high']),
                "swing_low": int(latest_row['swing_low']),
                "conf20": int(latest_row['confluence_20d']),
                "conf40": int(latest_row['confluence_40d']),
                "conf60": int(latest_row['confluence_60d']),
            }

            # 5️⃣ Emit via SocketIO
            if sio.connected:
                sio.emit('monitor_data', payload)
                print(f"📊 [{pair_label}] Prob: {prob*100:5.1f}% | Price: {payload['price']:.5f} (Sent)")
            else:
                print(f"📊 [{pair_label}] Prob: {prob*100:5.1f}% | Price: {payload['price']:.5f} (Offline)")

        except Exception as exc:
            print(f"❌ Monitor error: {exc}")

        # Wait until the next poll
        time.sleep(CONFIG["check_interval_sec"])

if __name__ == "__main__":
    monitor()