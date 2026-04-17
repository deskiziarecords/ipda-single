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

@socketio.on("disconnect")
def on_disconnect():
    print("🔌 Client disconnected")

# ----------------------------------------------------------------------
# Helper that the monitor script will import to push data
# ----------------------------------------------------------------------
def push_update(payload: dict):
    """
    Emit a `probability` event to **all** connected browsers.
    payload must contain:
        - pair (str)
        - probability (float, 0‑1)
        - threshold (float)
        - timestamp (ISO‑8601 string)
    """
    socketio.emit("probability", payload)

# ----------------------------------------------------------------------
# Run the server
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Use eventlet's WSGI server – it handles long‑running connections efficiently
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)