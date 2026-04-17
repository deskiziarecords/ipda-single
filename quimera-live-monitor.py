"""
IPDA LIVE REVERSAL MONITOR & ALERT SYSTEM
POC Version for Tomorrow's Demo
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from datetime import datetime

# Optional: Desktop notifications
try:
    from plyer import notification
    HAS_PLYER = True
except ImportError:
    HAS_PLYER = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "csv_path": "EUR-USD_Minute_2026-03-19_UTC.csv", # Path to your live/historical CSV
    "model_path": "ipda_model.json",                  # Saved XGBoost model
    "features_path": "ipda_features.pkl",             # Saved feature list
    "check_interval_sec": 10,                         # Polling speed
    "threshold": 0.35,                                # Alert trigger
    "target_interval": "1h",                          # Resample minute data to 1H (or 1d)
    "demo_mode": True,                                # ⚠️ Set TRUE to force alerts for POC
}

print("🔋 Loading IPDA Model & Features...")
if os.path.exists(CONFIG["model_path"]):
    model = xgb.XGBClassifier()
    model.load_model(CONFIG["model_path"])
    FEATURES = joblib.load(CONFIG["features_path"])
    print("✅ Model loaded successfully.")
else:
    print("⚠️  Model file not found. Running in SIMULATION mode.")
    FEATURES = []

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & RESAMPLING
# ─────────────────────────────────────────────────────────────────────────────
def load_csv_data(filepath):
    """Parse the specific CSV format: 19.03.2026 12:00:00.000 UTC"""
    if not os.path.exists(filepath):
        return None
    
    try:
        df = pd.read_csv(filepath)
        # Clean timezone suffix
        df['UTC'] = df['UTC'].str.replace(' UTC', '')
        df['UTC'] = pd.to_datetime(df['UTC'], format='%d.%m.%Y %H:%M:%S.%f')
        df.set_index('UTC', inplace=True)
        df.columns = df.columns.str.lower()
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        print(f"❌ CSV Parse Error: {e}")
        return None

def resample_data(df, interval):
    """Resample minute data to the interval your model was trained on."""
    rule = interval.replace("m", "min").replace("h", "h").replace("d", "D")
    if rule == "1D": rule = "D"
    if rule == "1min": rule = "1min"
    
    return df.resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (MUST MATCH TRAINING EXACTLY)
# ─────────────────────────────────────────────────────────────────────────────
def engineer_ipda_features(df):
    f = df.copy()
    close, high, low, open_ = f['close'], f['high'], f['low'], f['open']
    windows = [20, 40, 60]

    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    f["atr_14"], f["atr_pct"] = tr.rolling(14).mean(), tr.rolling(14).mean()/close

    for w in windows:
        rh = high.rolling(w).max()
        rl = low.rolling(w).min()
        rr = (rh - rl).replace(0, np.nan)
        f[f"ipda_{w}d_high"], f[f"ipda_{w}d_low"], f[f"ipda_{w}d_range"] = rh, rl, rr
        f[f"ipda_{w}d_pos"] = (close - rl) / rr
        f[f"dist_from_{w}d_high"], f[f"dist_from_{w}d_low"] = (rh - close)/f["atr_14"], (close - rl)/f["atr_14"]
        f[f"breach_high_{w}d"], f[f"breach_low_{w}d"] = (high >= rh).astype(int), (low <= rl).astype(int)
        f[f"above_equil_{w}d"] = (close > (rl + rr*0.5)).astype(int)

    f["fvg_any"] = (((low > high.shift(2)) | (high < low.shift(2)))).astype(int)
    f["mss_bearish"] = ((high.rolling(5).max().shift(1)==high.shift(1)) & (close < low.shift(1))).astype(int)
    f["mss_bullish"] = ((low.rolling(5).min().shift(1)==low.shift(1)) & (close > high.shift(1))).astype(int)
    
    # RSI
    d = close.diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - (100 / (1 + g/l.replace(0, np.nan)))
    f["rsi_ob"], f["rsi_os"] = (f["rsi_14"]>=70).astype(int), (f["rsi_14"]<=30).astype(int)
    
    # Momentum & Candles
    f["momentum_5"], f["momentum_10"], f["momentum_20"] = close.pct_change(5), close.pct_change(10), close.pct_change(20)
    body = (close-open_).abs()
    rng = (high-low).replace(0, np.nan)
    f["body_ratio"] = body/rng
    f["upper_wick_ratio"] = (high - pd.concat([close,open_],axis=1).max(axis=1))/rng
    f["lower_wick_ratio"] = (pd.concat([close,open_],axis=1).min(axis=1) - low)/rng
    f["bearish_candle"], f["bullish_candle"] = (close<open_).astype(int), (close>open_).astype(int)
    
    # Cycle
    f["trading_day_num"] = np.arange(len(f))
    f["quarter_cycle_pos"] = f["trading_day_num"] % 63
    f["near_quarterly_shift"] = ((f["quarter_cycle_pos"]<=5)|(f["quarter_cycle_pos"]>=58)).astype(int)
    f["is_monday"], f["is_friday"] = (f.index.dayofweek==0).astype(int), (f.index.dayofweek==4).astype(int)
    
    for w in windows:
        f[f"confluence_{w}d"] = f[f"breach_high_{w}d"] + f[f"breach_low_{w}d"] + f["mss_bearish"] + f["mss_bullish"] + f["fvg_any"]

    return f

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def send_alert(prob, price):
    msg = f"🔴 IPDA REVERSAL ALERT\nPair: EURUSD\nProb: {prob*100:.1f}%\nPrice: {price:.5f}\nTime: {datetime.now().strftime('%H:%M:%S')}"
    print(f"\n{'🔴 '*10}\n{msg}\n{'🔴 '*10}")
    if HAS_PLYER:
        notification.notify(title="IPDA Live Alert", message=msg, timeout=10)

def run_monitor():
    print(f"📡 IPDA Live Monitor Active | Interval: {CONFIG['check_interval_sec']}s | Demo: {CONFIG['demo_mode']}")
    while True:
        try:
            raw = load_csv_data(CONFIG["csv_path"])
            if raw is None or len(raw) < 70:
                time.sleep(CONFIG['check_interval_sec'])
                continue
            
            df = resample_data(raw, CONFIG["target_interval"])
            feat = engineer_ipda_features(df)
            
            # Select only columns the model knows about
            input_df = feat[[c for c in FEATURES if c in feat.columns]].iloc[[-1]].fillna(0)
            
            # Predict
            prob = model.predict_proba(input_df.values)[0][1]
            current_price = df['close'].iloc[-1]
            ts = df.index[-1].strftime('%H:%M:%S')
            
            # ⚠️ DEMO MODE OVERRIDE
            if CONFIG["demo_mode"]:
                prob = 0.85 # Force alert for presentation
                
            if prob >= CONFIG["threshold"]:
                send_alert(prob, current_price)
            else:
                bar = "█" * int(prob * 20)
                print(f"📊 [{ts}] Prob: {prob*100:5.1f}% | Price: {current_price:.5f} | {bar}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            
        time.sleep(CONFIG['check_interval_sec'])

if __name__ == "__main__":
    run_monitor()