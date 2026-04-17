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
# ENGINEERING FUNCTION (Identical to predictor)
# ----------------------------------------------------------------------
def engineer_ipda_features(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    f = df.copy()
    close  = f["close"]
    high   = f["high"]
    low    = f["low"]
    open_  = f["open"]

    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    f["atr_14"]  = tr.rolling(14).mean()
    f["atr_pct"] = f["atr_14"] / close

    for w in windows:
        roll_high = high.rolling(w).max()
        roll_low  = low.rolling(w).min()
        roll_range = (roll_high - roll_low).replace(0, np.nan)

        f[f"ipda_{w}d_high"]  = roll_high
        f[f"ipda_{w}d_low"]   = roll_low
        f[f"ipda_{w}d_range"] = roll_range
        f[f"ipda_{w}d_pos"]   = (close - roll_low) / roll_range
        f[f"dist_from_{w}d_high"] = (roll_high - close) / f["atr_14"]
        f[f"dist_from_{w}d_low"]  = (close - roll_low)  / f["atr_14"]
        f[f"breach_high_{w}d"] = (high >= roll_high).astype(int)
        f[f"breach_low_{w}d"]  = (low <= roll_low).astype(int)
        equil = roll_low + roll_range * 0.5
        f[f"above_equil_{w}d"] = (close > equil).astype(int)

    bull_fvg = (low > high.shift(2))
    bear_fvg = (high < low.shift(2))
    f["bull_fvg"] = bull_fvg.astype(int)
    f["bear_fvg"] = bear_fvg.astype(int)
    f["fvg_any"]  = ((bull_fvg) | (bear_fvg)).astype(int)

    f["swing_high"] = ((high > high.shift(1)) & (high > high.shift(-1))).astype(int)
    f["swing_low"]  = ((low < low.shift(1)) & (low < low.shift(-1))).astype(int)

    lookback = 5
    f["recent_hh"]  = (high == high.rolling(lookback).max()).astype(int)
    f["recent_ll"]  = (low == low.rolling(lookback).min()).astype(int)
    f["mss_bearish"] = ((f["recent_hh"].shift(1) == 1) & (close < low.shift(1))).astype(int)
    f["mss_bullish"] = ((f["recent_ll"].shift(1) == 1) & (close > high.shift(1))).astype(int)

    strong_bull = (close - close.shift(1)) > (1.5 * f["atr_14"])
    strong_bear = (close - close.shift(1)) < -(1.5 * f["atr_14"])
    f["near_bull_ob"] = strong_bear.shift(1).fillna(False).astype(int)
    f["near_bear_ob"] = strong_bull.shift(1).fillna(False).astype(int)

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    f["rsi_14"] = 100 - (100 / (1 + rs))
    f["rsi_ob"] = (f["rsi_14"] >= 70).astype(int)
    f["rsi_os"] = (f["rsi_14"] <= 30).astype(int)

    f["momentum_5"]  = close.pct_change(5)
    f["momentum_10"] = close.pct_change(10)
    f["momentum_20"] = close.pct_change(20)

    body       = (close - open_).abs()
    full_range = (high - low).replace(0, np.nan)
    f["body_ratio"]       = body / full_range
    f["upper_wick_ratio"] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / full_range
    f["lower_wick_ratio"] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / full_range
    f["bearish_candle"]   = (close < open_).astype(int)
    f["bullish_candle"]   = (close > open_).astype(int)

    f["trading_day_num"]   = np.arange(len(f))
    f["quarter_cycle_pos"] = f["trading_day_num"] % 63
    f["near_quarterly_shift"] = ((f["quarter_cycle_pos"] <= 5) | (f["quarter_cycle_pos"] >= 58)).astype(int)

    f["is_monday"]   = (f.index.dayofweek == 0).astype(int)
    f["is_friday"]   = (f.index.dayofweek == 4).astype(int)
    f["week_of_month"] = (f.index.day - 1) // 7 + 1

    for w in windows:
        f[f"confluence_{w}d"] = (
            f[f"breach_high_{w}d"] + f[f"breach_low_{w}d"] +
            f["mss_bearish"] + f["mss_bullish"] + f["fvg_any"]
        )

    return f

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
