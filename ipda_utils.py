import numpy as np
import pandas as pd

def engineer_ipda_features(df: pd.DataFrame, windows: list = [20, 40, 60]) -> pd.DataFrame:
    f = df.copy()
    close  = f["close"]
    high   = f["high"]
    low    = f["low"]
    open_  = f["open"]

    # ATR
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    f["atr_14"]  = tr.rolling(14).mean()
    f["atr_pct"] = f["atr_14"] / close

    # IPDA Ranges
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

    # FVG Detection
    bull_fvg = (low > high.shift(2))
    bear_fvg = (high < low.shift(2))
    f["bull_fvg"] = bull_fvg.astype(int)
    f["bear_fvg"] = bear_fvg.astype(int)
    f["fvg_any"]  = ((bull_fvg) | (bear_fvg)).astype(int)

    # Swing High/Low
    f["swing_high"] = ((high > high.shift(1)) & (high > high.shift(-1))).astype(int)
    f["swing_low"]  = ((low < low.shift(1)) & (low < low.shift(-1))).astype(int)

    # MSS
    lookback = 5
    f["recent_hh"]  = (high == high.rolling(lookback).max()).astype(int)
    f["recent_ll"]  = (low == low.rolling(lookback).min()).astype(int)
    f["mss_bearish"] = ((f["recent_hh"].shift(1) == 1) & (close < low.shift(1))).astype(int)
    f["mss_bullish"] = ((f["recent_ll"].shift(1) == 1) & (close > high.shift(1))).astype(int)

    # Order Block Proximity
    strong_bull = (close - close.shift(1)) > (1.5 * f["atr_14"])
    strong_bear = (close - close.shift(1)) < -(1.5 * f["atr_14"])
    f["near_bull_ob"] = strong_bear.shift(1).fillna(False).astype(int)
    f["near_bear_ob"] = strong_bull.shift(1).fillna(False).astype(int)

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    f["rsi_14"] = 100 - (100 / (1 + rs))
    f["rsi_ob"] = (f["rsi_14"] >= 70).astype(int)
    f["rsi_os"] = (f["rsi_14"] <= 30).astype(int)

    # Momentum
    f["momentum_5"]  = close.pct_change(5)
    f["momentum_10"] = close.pct_change(10)
    f["momentum_20"] = close.pct_change(20)

    # Candle Body/Wick
    body       = (close - open_).abs()
    full_range = (high - low).replace(0, np.nan)
    f["body_ratio"]       = body / full_range
    f["upper_wick_ratio"] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / full_range
    f["lower_wick_ratio"] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / full_range
    f["bearish_candle"]   = (close < open_).astype(int)
    f["bullish_candle"]   = (close > open_).astype(int)

    # Quarterly Cycle
    f["trading_day_num"]   = np.arange(len(f))
    f["quarter_cycle_pos"] = f["trading_day_num"] % 63
    f["near_quarterly_shift"] = ((f["quarter_cycle_pos"] <= 5) | (f["quarter_cycle_pos"] >= 58)).astype(int)

    # Day of Week
    f["is_monday"]   = (f.index.dayofweek == 0).astype(int)
    f["is_friday"]   = (f.index.dayofweek == 4).astype(int)
    f["week_of_month"] = (f.index.day - 1) // 7 + 1

    # Confluence
    for w in windows:
        f[f"confluence_{w}d"] = (
            f[f"breach_high_{w}d"] + f[f"breach_low_{w}d"] +
            f["mss_bearish"] + f["mss_bullish"] + f["fvg_any"]
        )

    return f
