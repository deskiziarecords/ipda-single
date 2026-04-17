import pandas as pd
import numpy as np

# Load your OHLCV data
df = pd.read_csv(EUR-USD_Minute_2026-4-2_UTC.csvimport pandas as pd
import numpy as np

# Load your OHLCV data
df = pd.read_csv("your_data.csv", parse_dates=["UTC"], dayfirst=True)

# --- Step 1: Window the data ---
window_size = 10

# Add a temporary integer index for windowing
df["row_num"] = range(len(df))  # Assign row numbers: 0, 1, 2, ...
df["window_id"] = f"window_{df['row_num'] // window_size}"  # Now safe to divide

# --- Step 2: Label windows as fracture/stable ---
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
df["volatility"] = df["log_return"].rolling(window_size).std()
df["is_fracture"] = (df["volatility"] > df["volatility"].quantile(0.9)).astype(int)
df["is_fracture"] = df.groupby("window_id")["is_fracture"].transform("first")

# --- Step 3: Compute missing features ---
df["ofi"] = np.where(df["Close"] > df["Open"], 1, -1)
df["liquidity_grad"] = df["Volume"].pct_change().rolling(5).mean()
df["vol_idx"] = df["log_return"].rolling(window_size).std()

# --- Step 4: Select and save required columns ---
result = df[["window_id", "is_fracture", "Close", "ofi", "liquidity_grad", "vol_idx"]].copy()
result = result.rename(columns={"Close": "price"})

# Save to CSV
result.to_csv("windows.csv", index=False)
print("Converted data saved to windows.csv")
print("Example rows:")
print(result.head(15))", parse_dates=["UTC"], dayfirst=True)
df = df.set_index("UTC")

# --- Step 1: Window the data ---
# Define window size (e.g., 10 bars per window)
window_size = 10
df["window_id"] = f"window_{df.index // window_size}"

# --- Step 2: Label windows as fracture/stable ---
# Example: Label windows with high volatility as fractures (customize this!)
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
df["volatility"] = df["log_return"].rolling(window_size).std()
df["is_fracture"] = (df["volatility"] > df["volatility"].quantile(0.9)).astype(int)  # Top 10% volatility = fracture

# Forward-fill labels to all bars in the window
df["is_fracture"] = df.groupby("window_id")["is_fracture"].transform("first")

# --- Step 3: Compute missing features ---
# Order Flow Imbalance (OFI): Approximate using tick rule (Close > Open = buy pressure)
df["ofi"] = np.where(df["Close"] > df["Open"], 1, -1)  # Simplified OFI

# Liquidity gradient: Approximate using volume changes (customize!)
df["liquidity_grad"] = df["Volume"].pct_change().rolling(5).mean()

# Volatility index: Use rolling std of log returns
df["vol_idx"] = df["log_return"].rolling(window_size).std()

# --- Step 4: Select and save required columns ---
result = df[["window_id", "is_fracture", "Close", "ofi", "liquidity_grad", "vol_idx"]].copy()
result = result.rename(columns={"Close": "price"})  # Rename to match IPDA format

# Save to CSV
result.to_csv("windows.csv", index=False)
print("Converted data saved to windows.csv")
print("Example rows:")
print(result.head(15))