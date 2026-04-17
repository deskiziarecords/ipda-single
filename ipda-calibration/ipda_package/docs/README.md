# IPDA Threshold Calibration System

Rust production binaries for calibrating and running the IPDA regime fracture
detectors: **λ₃** (Spectral Phase Inversion) and **H₁** (Persistent Homology /
TopoStress).

---

## Binaries

| Binary | Purpose |
|--------|---------|
| `ipda_calibrate` | Batch calibration from labelled CSV data |
| `ipda_detect` | Real-time streaming detector (stdin → stdout) |

---

## ipda_calibrate

### Input CSV format

One row per bar. All bars belonging to the same window share a `window_id`.

```
window_id,is_fracture,price,ofi,liquidity_grad,vol_idx
frac_00,1,6.9077,-1.823,-0.54,3.0
frac_00,1,6.9082,-1.791,-0.61,3.0
...
stab_000,0,6.9079,1.543,0.82,2.0
```

| Column | Type | Description |
|--------|------|-------------|
| `window_id` | string | Unique ID per window (all rows with same ID = one window) |
| `is_fracture` | 0/1 | 1 = regime fracture window, 0 = stable |
| `price` | float | Log price (or raw price — will be detrended) |
| `ofi` | float | Order flow imbalance |
| `liquidity_grad` | float | Liquidity gradient (optional, 0 if unavailable) |
| `vol_idx` | float | Volatility regime index (optional, 0 if unavailable) |

### Usage

```bash
# Standard calibration (100% win-rate constraint)
ipda_calibrate \
  --data windows.csv \
  --output thresholds.json \
  --seg-len 64 \
  --min-tpr 1.0

# Verbose mode (shows DEBUG-level detail)
ipda_calibrate --data windows.csv --output thresholds.json --verbose

# Relax recall if 100% is not achievable on training set
ipda_calibrate --data windows.csv --output thresholds.json --min-tpr 0.9375
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | required | Input CSV path |
| `--output` | `thresholds.json` | Output JSON path |
| `--seg-len` | 64 | FFT segment length in bars (use 32 for short windows) |
| `--min-tpr` | 1.0 | Minimum recall on training set (1.0 = zero missed fractures) |
| `--merge-tol` | 0.5 | Atlas merge tolerance in σ units (isneighbor gate) |
| `--verbose` | off | Enable DEBUG logging |

### Output JSON

```json
{
  "lambda3_psi_thresh": -0.50,
  "lambda3_angle_deg": 120.0,
  "gamma_drift": -0.15,
  "gamma_topo": 4.57,
  "tau_topo": 56.65,
  "lambda3_train_tpr": 1.0,
  "lambda3_train_fpr": 0.0,
  "h1_train_tpr": 0.9375,
  "h1_train_fpr": 0.86,
  "n_fracture_windows": 16,
  "n_stable_windows": 100
}
```

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success — thresholds saved, verification passed |
| 1 | Failure — missed fracture windows or invalid input |

---

## ipda_detect

Reads a live bar stream from **stdin** (same CSV format, `is_fracture` ignored),
outputs one detection line per completed window to **stdout**.

### Usage

```bash
# Human-readable output
cat live_feed.csv | ipda_detect --thresholds thresholds.json --seg-len 64

# JSON output (for downstream parsing / alerting)
cat live_feed.csv | ipda_detect --thresholds thresholds.json --json

# Pipe from a live data source
market_feed_generator | ipda_detect --thresholds thresholds.json --json \
  | jq 'select(.regime_fracture == true)'
```

### Human-readable output columns

```
window_id         psi    dpsi_dt  topo_stress     lam3       h1       FRACTURE
W_001          -0.920    -0.182       62.310    true     true  *** FRACTURE ***
W_002           0.981     0.021       18.440    false    false
```

### JSON output format (one object per line)

```json
{"window_id":"W_001","psi":-0.9200,"dpsi_dt":-0.1820,"topo_stress":62.31,"lambda3":true,"h1":true,"regime_fracture":true}
```

---

## Threshold semantics

| Sensor | Fires when | Meaning |
|--------|-----------|---------|
| λ₃ | `Ψ < lambda3_psi_thresh` AND `dΨ/dt < gamma_drift` | Price/OFI cross-spectrum phase has inverted (institutional decoupling) |
| H₁ | `TopoStress > tau_topo` AND atlas charts can't merge | Persistent topological loops indicate regime boundary |
| Combined | λ₃ OR H₁ | Regime fracture — halt or exit position |

**dΨ/dt warmup:** The drift gate (`gamma_drift`) is only applied once 2+ windows
have been processed. On the first window, Ψ level alone decides.

---

## Rebuilding from source

Requires Rust ≥ 1.75 (tested on 1.75.0).

```bash
cargo build --release
# Binaries: target/release/ipda_calibrate, target/release/ipda_detect
```

Dependencies (all pure Rust, no C FFI required):
- `rustfft` — FFT for Welch PSD / cross-PSD
- `num-complex` — complex arithmetic
- `statrs` — statistical distributions
- `serde` / `serde_json` — JSON serialisation
- `clap` — CLI argument parsing
- `csv` — CSV loading
- `anyhow` / `thiserror` — error handling
- `log` / `env_logger` — structured logging

---

## Notes on H₁ (Persistent Homology)

The built-in Rips persistence uses a lightweight union-find implementation.
For production use with real market microstructure data:

1. The 4D embedding `[z_price, z_ofi, z_liq_grad, z_vol_idx]` is computed
   automatically from your input columns.
2. If TopoStress is not separable on your training data (WARN logged), H₁
   operates in **confirmatory mode** — λ₃ provides the primary recall guarantee.
3. For higher-dimensional or more precise persistence, replace
   `compute_h1_persistence()` in `src/topo.rs` with a GUDHI or Ripser FFI call.
