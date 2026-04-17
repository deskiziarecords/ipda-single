# IPDA Reversal Prediction & Adelic Market Oracle

A high-performance algorithmic trading system based on the **ICT Interbank Price Delivery Algorithm (IPDA)** framework, featuring real-time reversal prediction, HFT simulations, and a live monitoring dashboard.

## 🚀 System Overview

This repository integrates three core components:
1.  **IPDA Reversal Suite**: An XGBoost-based prediction system that identifies high-probability price reversal windows using institutional order flow concepts (FVGs, MSS, IPDA Ranges).
2.  **Adelic Oracle**: A Non-Archimedean HFT superstructure powered by **JAX**, focusing on "Truth Bridge" news filtering and dark liquidity routing.
3.  **Live Monitoring Dashboard**: A real-time web interface powered by **Flask-SocketIO** and **Chart.js**.

---

## 🏗️ Architecture

### 1. IPDA Prediction & Monitoring
*   **`ipda-predictor.py`**: Fetches historical data (Yahoo Finance, Bitget, MT5), engineers IPDA features, and trains the XGBoost classification model.
*   **`ipda-live-monitor.py`**: Polls live data and emits real-time reversal probabilities to the dashboard.
*   **`ipda-historical-replay.py`**: Simulates historical periods to verify model performance visually on the dashboard.
*   **`ipda_utils.py`**: The "Source of Truth" for feature engineering, ensuring consistency across training and live execution.

### 2. Adelic Oracle (`/adelic`)
*   **`adelic_oracle_app.py`**: A Streamlit application for HFT simulation.
*   **`adelic_causal_force_generalizer.py`**: Uses Weierstrass smoothing and FORCE constraints for causal refinement.
*   **`adelic_choco_schur_router.py`**: Implements a dark liquidity router using Schur complement allocation and gossip-compressed consensus.

### 3. Real-Time Dashboard
*   **`app.py`**: Flask server with Eventlet monkey patching for high-concurrency SocketIO connections.
*   **`dashboard/index.html`**: Responsive Tailwind CSS frontend with a real-time probability gauge and historical line charts.

### 4. Calibration (`/ipda-calibration`)
*   A Rust-based crate for ultra-fast threshold calibration using spectral phase inversion and persistent homology.

---

## 📦 Installation

Ensure you have Python 3.12+ installed.

```bash
pip install pandas numpy xgboost yfinance ccxt scikit-learn matplotlib seaborn joblib plotly jax jaxlib streamlit flask flask-socketio eventlet
```

*Note: For the Rust calibration module, you will need the Rust toolchain installed to run `cargo build`.*

---

## 🛠️ Usage

### 1. Training the Model
First, generate the model and feature metadata:
```bash
python3 ipda-predictor.py
```

### 2. Running the Live Dashboard
Start the SocketIO server:
```bash
python3 app.py
```
Then, start the monitor (in a separate terminal):
```bash
python3 ipda-live-monitor.py
```
Open `http://localhost:5000` to view the live dashboard.

### 3. Adelic Oracle Simulation
```bash
cd adelic
streamlit run adelic_oracle_app.py
```

---

## 🧠 Key Features
- **Institutional Indicators**: Automated detection of Fair Value Gaps (FVG), Market Structure Shifts (MSS), and Order Blocks (OB).
- **Multi-Source Data**: Native support for Crypto (Bitget via CCXT), Forex (MetaTrader 5), and general equities (Yahoo Finance).
- **JAX Acceleration**: Sub-millisecond execution simulations for HFT environments.
- **Robust Feature Engineering**: Centralized pipeline in `ipda_utils.py` to prevent train-serve skew.

---

## ⚖️ Disclaimer
This system is for educational and research purposes only. Algorithmic trading involves significant risk. Past performance is not indicative of future results.
