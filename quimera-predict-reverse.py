"""
IPDA FOREX REVERSAL PREDICTION SYSTEM
Based on the ICT Interbank Price Delivery Algorithm (IPDA) framework.
"""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from ipda_utils import engineer_ipda_features

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "pair":               "EURUSD=X",
    "start_date":         "2018-01-01",
    "end_date":           datetime.today().strftime("%Y-%m-%d"),
    "interval":           "1d",
    "ipda_windows":       [20, 40, 60],
    "reversal_threshold_pct": 0.8,
    "reversal_fwd_window":    10,
    "n_splits":           5,
    "random_state":       42,
}

PAIR = CONFIG["pair"]
PAIR_LABEL = PAIR.replace("=X", "")

print(f"{'='*65}")
print(f"  IPDA REVERSAL PREDICTION SYSTEM — {PAIR_LABEL}")
print(f"{'='*65}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA ACQUISITION
# ─────────────────────────────────────────────────────────────────────────────
def fetch_data(pair: str, start: str, end: str, interval: str) -> pd.DataFrame:
    print(f"[1] Fetching {pair} from {start} to {end}...")
    df = yf.download(pair, start=start, end=end, interval=interval,
                     auto_adjust=True, progress=False)
    # Fix tuple columns from yfinance
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    print(f"    → {len(df)} daily bars loaded.\n")
    return df

df = fetch_data(PAIR, CONFIG["start_date"], CONFIG["end_date"], CONFIG["interval"])

# ─────────────────────────────────────────────────────────────────────────────
# 2. IPDA FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("[2] Engineering IPDA features...")
df_feat = engineer_ipda_features(df, CONFIG["ipda_windows"])

# ─────────────────────────────────────────────────────────────────────────────
# 3. REVERSAL LABELING
# ─────────────────────────────────────────────────────────────────────────────
def label_reversals(df: pd.DataFrame, threshold_pct: float, fwd_window: int) -> pd.DataFrame:
    print("[3] Labeling reversal periods...")
    df = df.copy()
    close = df["close"].values
    n = len(close)
    labels = np.zeros(n, dtype=int)

    for i in range(5, n - fwd_window):
        trend = close[i] - close[i - 5]
        fwd_prices = close[i + 1 : i + 1 + fwd_window]
        if trend > 0:
            min_fwd = fwd_prices.min()
            drawdown = (close[i] - min_fwd) / close[i] * 100
            if drawdown >= threshold_pct:
                labels[i] = 1
        elif trend < 0:
            max_fwd = fwd_prices.max()
            rally = (max_fwd - close[i]) / close[i] * 100
            if rally >= threshold_pct:
                labels[i] = 1

    df["reversal"] = labels
    rev_count = labels.sum()
    total = n - 5 - fwd_window
    pct = rev_count / total * 100
    print(f"    → {rev_count} reversal periods labeled out of {total} bars ({pct:.1f}%)\n")
    return df

df_labeled = label_reversals(df_feat, CONFIG["reversal_threshold_pct"], CONFIG["reversal_fwd_window"])

# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE SELECTION & TRAIN/TEST PREP
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [c for c in df_labeled.columns if c not in [
    "open", "high", "low", "close", "volume", "reversal",
    "ipda_20d_high", "ipda_20d_low", "ipda_40d_high", "ipda_40d_low",
    "ipda_60d_high", "ipda_60d_low", "atr_14"
]]

print("[4] Preparing model dataset...")
model_df = df_labeled[FEATURE_COLS + ["reversal"]].dropna()
print(f"    → Dataset: {len(model_df)} rows × {len(FEATURE_COLS)} features")
print(f"    → Class balance: {model_df['reversal'].value_counts().to_dict()}\n")

X = model_df[FEATURE_COLS].values
y = model_df["reversal"].values
split_idx = int(len(X) * 0.80)
X_train_full, X_test = X[:split_idx], X[split_idx:]
y_train_full, y_test = y[:split_idx], y[split_idx:]
dates_test = model_df.index[split_idx:]

cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train_full)
scale_pos_weight = cw[1] / cw[0]

# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
print("[5] Training XGBoost with TimeSeriesSplit cross-validation...")
tscv = TimeSeriesSplit(n_splits=CONFIG["n_splits"])
xgb_params = {
    "objective": "binary:logistic", "eval_metric": "auc", "n_estimators": 300,
    "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 3, "scale_pos_weight": scale_pos_weight,
    "random_state": CONFIG["random_state"], "verbosity": 0,
}

cv_aucs = []
for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_full)):
    Xtr, Xval = X_train_full[tr_idx], X_train_full[val_idx]
    ytr, yval = y_train_full[tr_idx], y_train_full[val_idx]
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    prob = model.predict_proba(Xval)[:, 1]
    auc = roc_auc_score(yval, prob)
    cv_aucs.append(auc)
    print(f"    Fold {fold+1}: AUC = {auc:.4f}")
print(f"\n    → Mean CV AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}\n")

print("[6] Training final model on full training set...")
final_model = xgb.XGBClassifier(**xgb_params)
final_model.fit(X_train_full, y_train_full, verbose=False)

# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("[7] Evaluating on holdout test set...\n")
y_prob = final_model.predict_proba(X_test)[:, 1]
THRESHOLD = 0.35
y_pred = (y_prob >= THRESHOLD).astype(int)
test_auc = roc_auc_score(y_test, y_prob)
print(f"    Holdout AUC:  {test_auc:.4f}")
print(f"    Threshold:    {THRESHOLD}\n")
print(classification_report(y_test, y_pred, target_names=["No Reversal", "Reversal"]))

# ─────────────────────────────────────────────────────────────────────────────
# 7. LIVE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] Live Prediction on latest data...")
latest = model_df[FEATURE_COLS].iloc[[-1]]
live_prob = final_model.predict_proba(latest.values)[0][1]
live_pred = int(live_prob >= THRESHOLD)

print(f"\n{'─'*50}")
print(f"  Pair:              {PAIR_LABEL}")
print(f"  Latest Date:       {model_df.index[-1].date()}")
print(f"  Reversal Probability: {live_prob*100:.1f}%")
print(f"  Signal:            {'⚠️  HIGH PROBABILITY REVERSAL' if live_pred else '✅  No Reversal Expected'}")
print(f"{'─'*50}\n")

recent = model_df.tail(10).copy()
recent_prob = final_model.predict_proba(recent[FEATURE_COLS].values)[:, 1]
print("  Recent 10 bars — Reversal Probabilities:")
for date, prob in zip(recent.index, recent_prob):
    bar = "█" * int(prob * 20)
    signal = " ← SIGNAL" if prob >= THRESHOLD else ""
    print(f"    {date.date()}  {prob*100:5.1f}%  |{bar:<20}|{signal}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] Generating plots...")
fig, axes = plt.subplots(3, 2, figsize=(18, 16))
fig.suptitle(f"IPDA Reversal Prediction System — {PAIR_LABEL}", fontsize=16, fontweight="bold", y=1.01)

# Plot 1
ax1 = axes[0, 0]
test_close = df_labeled["close"].loc[dates_test]
ax1.plot(dates_test, test_close, color="#2196F3", linewidth=1.2, label="Close")
for dt in dates_test[y_pred == 1]:
    ax1.axvspan(dt, dt + timedelta(days=1), alpha=0.35, color="#FF5722", linewidth=0)
actual_rev_dates = dates_test[y_test == 1]
ax1.scatter(actual_rev_dates, test_close.loc[actual_rev_dates], color="#4CAF50", s=40, zorder=5, label="Actual Reversal", marker="^")
ax1.set_title("Price with Predicted Reversal Windows (Orange=Predicted, Green=Actual)")
ax1.set_ylabel("Price")
ax1.legend(loc="upper left", fontsize=8)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30)

# Plot 2
ax2 = axes[0, 1]
ax2.fill_between(dates_test, y_prob, alpha=0.6, color="#9C27B0", label="Rev. Probability")
ax2.axhline(THRESHOLD, color="#FF5722", linestyle="--", linewidth=1.5, label=f"Threshold ({THRESHOLD})")
ax2.set_title("Predicted Reversal Probability")
ax2.set_ylabel("Probability")
ax2.set_ylim(0, 1)
ax2.legend(fontsize=8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)

# Plot 3
ax3 = axes[1, 0]
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax3.plot(fpr, tpr, color="#2196F3", linewidth=2, label=f"AUC = {test_auc:.3f}")
ax3.plot([0, 1], [0, 1], "k--", linewidth=1)
ax3.fill_between(fpr, tpr, alpha=0.15, color="#2196F3")
ax3.set_title("ROC Curve — Holdout Test Set")
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.legend(fontsize=10)

# Plot 4
ax4 = axes[1, 1]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax4, xticklabels=["No Rev.", "Reversal"], yticklabels=["No Rev.", "Reversal"])
ax4.set_title("Confusion Matrix")
ax4.set_xlabel("Predicted")
ax4.set_ylabel("Actual")

# Plot 5
ax5 = axes[2, 0]
importance = pd.Series(final_model.feature_importances_, index=FEATURE_COLS)
top20 = importance.nlargest(20).sort_values()
colors = ["#FF5722" if any(x in i for x in ["breach", "ipda", "mss"]) else "#2196F3" for i in top20.index]
top20.plot(kind="barh", ax=ax5, color=colors)
ax5.set_title("Top 20 Feature Importances\n(Orange = IPDA-specific)")
ax5.set_xlabel("XGBoost Importance Score")

# Plot 6
ax6 = axes[2, 1]
recent_plot = df_labeled.tail(120)
ax6.plot(recent_plot.index, recent_plot["close"], color="#212121", linewidth=1.5, label="Close", zorder=5)
colors_w = {"20": "#4CAF50", "40": "#FF9800", "60": "#F44336"}
for w in [20, 40, 60]:
    ax6.plot(recent_plot.index, recent_plot[f"ipda_{w}d_high"], linestyle="--", linewidth=1, color=colors_w[str(w)], alpha=0.8, label=f"{w}d High")
    ax6.plot(recent_plot.index, recent_plot[f"ipda_{w}d_low"], linestyle=":", linewidth=1, color=colors_w[str(w)], alpha=0.8, label=f"{w}d Low")

rev_prob_recent = final_model.predict_proba(model_df[FEATURE_COLS].tail(120).values)[:, 1]
rev_dates_recent = model_df.index[-120:]
for dt, p in zip(rev_dates_recent, rev_prob_recent):
    if p >= THRESHOLD:
        ax6.axvspan(dt, dt + timedelta(days=1), alpha=0.25, color="#9C27B0", linewidth=0)
ax6.set_title("Recent 120 Days: IPDA Ranges + Reversal Signals (Purple)")
ax6.set_ylabel("Price")
ax6.legend(fontsize=7, ncol=2, loc="upper left")
ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=30)

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/ipda_reversal_analysis.png", dpi=150, bbox_inches="tight")
print("    → Plots saved to: outputs/ipda_reversal_analysis.png\n")
print("=" * 65)
print("  SYSTEM COMPLETE")
print("  Files: ipda_reversal_predictor.py + outputs/ipda_reversal_analysis.png")
print("=" * 65)