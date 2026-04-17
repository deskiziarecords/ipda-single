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
from ipda_utils import engineer_ipda_features

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