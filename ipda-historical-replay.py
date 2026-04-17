# ipda-historical-replay.py
import time
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import ccxt
import MetaTrader5 as mt5
import yfinance as yf
from datetime import datetime, timezone, timedelta
from app import push_update  # socket helper
from ipda_utils import engineer_ipda_features

# ----------------------------------------------------------------------
# REPLAY CONFIGURATION
# ----------------------------------------------------------------------
CONFIG = {
    "data_source": "bitget",    # "yahoo", "bitget", "metatrader"
    "pair": "BTC/USDT",         # e.g. "BTC/USDT", "EURUSD"
    "start_date": "2023-01-01", # Where to start the historical download
    "end_date": datetime.today().strftime("%Y-%m-%d"),
    "interval": "1d",
    "threshold": 0.35,
    "replay_speed_sec": 0.5     # Wait time between pushing each bar (seconds)
}

PAIR_LABEL = CONFIG["pair"].replace("=X", "").replace("/", "")

# ----------------------------------------------------------------------
# HISTORICAL DATA FETCHING
# ----------------------------------------------------------------------
def fetch_historical_data(source, pair, start, end, interval):
    if source == "yahoo":
        df = yf.download(pair, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)
        return df
        
    elif source == "bitget":
        exchange = ccxt.bitget({'enableRateLimit': True})
        since = exchange.parse8601(f"{start}T00:00:00Z")
        timeframe = interval.replace("d", "d").replace("h", "h").replace("m", "m")
        all_bars = []
        while True:
            bars = exchange.fetch_ohlcv(pair, timeframe=timeframe, since=since, limit=1000)
            if not bars: break
            all_bars.extend(bars)
            since = bars[-1][0] + 1
            if len(all_bars) > 10000: break
        df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
        
    elif source == "metatrader":
        if not mt5.initialize():
            raise Exception("MT5 initialization failed")
        tf_map = { "1d": mt5.TIMEFRAME_D1, "1h": mt5.TIMEFRAME_H1, "15m": mt5.TIMEFRAME_M15, "5m": mt5.TIMEFRAME_M5, "1m": mt5.TIMEFRAME_M1 }
        tf = tf_map.get(interval, mt5.TIMEFRAME_D1)
        s_dt = datetime.strptime(start, '%Y-%m-%d')
        e_dt = datetime.strptime(end, '%Y-%m-%d') + timedelta(days=1)
        rates = mt5.copy_rates_range(pair, tf, s_dt, e_dt)
        if rates is None or len(rates) == 0:
            mt5.shutdown()
            raise Exception(f"Failed to fetch {pair} from MT5")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        mt5.shutdown()
        return df[['open', 'high', 'low', 'close', 'volume']]
        
    else:
        raise ValueError(f"Unknown data source: {source}")

# ----------------------------------------------------------------------
# REPLAY EXECUTION
# ----------------------------------------------------------------------
def run_replay():
    print(f"🎬 Initializing Historical Replay for {PAIR_LABEL}")
    print(f"Data Source: {CONFIG['data_source']}")
    
    # 1. Load Model & Features
    try:
        model = xgb.XGBClassifier()
        model.load_model("ipda_model.json")
        FEATURE_COLS = joblib.load("ipda_features.pkl")
    except Exception as e:
        print(f"❌ Failed to load model. Did you run the predictor first? Error: {e}")
        return

    # 2. Fetch Historical Blocks
    print(f"📥 Loading historical timeline from {CONFIG['start_date']} to {CONFIG['end_date']}...")
    df = fetch_historical_data(CONFIG["data_source"], CONFIG["pair"], CONFIG["start_date"], CONFIG["end_date"], CONFIG["interval"])
    
    if len(df) < 70:
        print(f"⚠️ Not enough historical data. Loaded {len(df)} bars.")
        return
        
    print(f"⚙️ Engineering features for {len(df)} historical bars...")
    df_feat = engineer_ipda_features(df, windows=[20, 40, 60])
    
    # Drop rows at the beginning that produce NaN from rolling features
    df_valid = df_feat.dropna(subset=FEATURE_COLS)
    
    print(f"▶️ Starting Replay! (Speed: 1 update every {CONFIG['replay_speed_sec']} sec)")
    
    for i in range(len(df_valid)):
        row = df_valid.iloc[i : i+1]
        dt = row.index[0]
        
        prob = model.predict_proba(row[FEATURE_COLS].values)[0][1]
        
        payload = {
            "pair": PAIR_LABEL,
            "probability": float(prob),
            "threshold": CONFIG["threshold"],
            "timestamp": dt.isoformat(),
            "close": float(row["close"].iloc[0]),
            "is_replay": True
        }
        
        # 3. Output logic
        signal_char = "⚠️" if prob >= CONFIG["threshold"] else "✅"
        print(f"[{dt.strftime('%Y-%m-%d')}] {PAIR_LABEL} Price: {payload['close']:.5f} | Prob: {prob*100:5.1f}% | {signal_char}")
        
        # Emit over socket if available
        try:
            push_update(payload)
        except Exception:
            pass # Fails silently if no socket server running
            
        time.sleep(CONFIG["replay_speed_sec"])
        
    print("🏁 Replay Completed.")

if __name__ == "__main__":
    run_replay()
