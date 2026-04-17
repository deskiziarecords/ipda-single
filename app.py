# app.py
import os
from flask import Flask, send_from_directory, render_template
from flask_socketio import SocketIO, emit
import eventlet
eventlet.monkey_patch()   # required for Flask‑SocketIO to work with eventlet

# ----------------------------------------------------------------------
# Flask app configuration
# ----------------------------------------------------------------------
app = Flask(__name__, static_folder="dashboard", static_url_path="/dashboard")
socketio = SocketIO(app, cors_allowed_origins="*")   # allow any origin for demo

# Shared system configuration
SYSTEM_CONFIG = {
    "data_source": "bitget",
    "pair": "BTC/USDT",
    "interval": "1d",
    "threshold": 0.35,
    "check_interval_sec": 60,
    "demo_mode": True,
    "model_path": "ipda_model.json"
}

# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/")
def index():
    # Serve the dashboard HTML (Tailwind + Chart.js already inside dashboard/)
    return send_from_directory("dashboard", "index.html")

# ----------------------------------------------------------------------
# SocketIO events
# ----------------------------------------------------------------------
@socketio.on("connect")
def on_connect():
    print("🔌 Client connected")
    emit("status", {"msg": "connected"})
    # Send current config to newly connected client
    emit("config_updated", SYSTEM_CONFIG)

@socketio.on("disconnect")
def on_disconnect():
    print("🔌 Client disconnected")

@socketio.on("update_config")
def on_update_config(data):
    print(f"⚙️ Config update received: {data}")
    SYSTEM_CONFIG.update(data)
    # Broadcast update to everyone (dashboard and monitor)
    socketio.emit("config_updated", SYSTEM_CONFIG)

@socketio.on("monitor_data")
def on_monitor_data(payload):
    # This comes from the live monitor script
    payload['is_live'] = True
    socketio.emit("probability", payload)

# ----------------------------------------------------------------------
# Helper that the monitor script will import to push data
# (Kept for backward compatibility if needed, but monitor should use SocketIO Client)
# ----------------------------------------------------------------------
def push_update(payload: dict):
    socketio.emit("probability", payload)

# ----------------------------------------------------------------------
# Run the server
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Use eventlet's WSGI server – it handles long‑running connections efficiently
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)