/// Core data types for IPDA calibration system

use serde::{Deserialize, Serialize};

/// A single market window (1-minute bar sample)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketWindow {
    /// Log price series (length = window_len)
    pub price: Vec<f64>,
    /// Order flow imbalance (length = window_len)
    pub ofi: Vec<f64>,
    /// Liquidity gradient (optional; zeros if unavailable)
    pub liquidity_grad: Vec<f64>,
    /// Volatility regime index per bar (1=low, 2=normal, 3=high)
    pub vol_idx: Vec<f64>,
    /// Label: true = regime fracture, false = stable
    pub is_fracture: bool,
}

impl MarketWindow {
    pub fn len(&self) -> usize {
        self.price.len()
    }

    pub fn is_empty(&self) -> bool {
        self.price.is_empty()
    }

    /// Validate all series have equal length
    pub fn validate(&self) -> anyhow::Result<()> {
        let n = self.price.len();
        anyhow::ensure!(!self.ofi.is_empty(), "ofi must not be empty");
        anyhow::ensure!(self.ofi.len() == n, "ofi length mismatch");
        anyhow::ensure!(
            self.liquidity_grad.len() == n || self.liquidity_grad.is_empty(),
            "liquidity_grad length mismatch"
        );
        anyhow::ensure!(
            self.vol_idx.len() == n || self.vol_idx.is_empty(),
            "vol_idx length mismatch"
        );
        Ok(())
    }
}

/// Calibrated thresholds output — JSON-serialisable for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IPDAThresholds {
    // --- λ₃ spectral phase inversion ---
    /// Ψ threshold: fire λ₃ when Ψ(t) < lambda3_psi_thresh
    pub lambda3_psi_thresh: f64,
    /// Phase angle in degrees (informational)
    pub lambda3_angle_deg: f64,
    /// Rate-of-change threshold: dΨ/dt must be < gamma_drift to fire
    pub gamma_drift: f64,

    // --- H₁ persistent homology ---
    /// Noise floor: loops with persistence < gamma_topo are ignored
    pub gamma_topo: f64,
    /// Fracture threshold: TopoStress must exceed tau_topo to fire
    pub tau_topo: f64,

    // --- Diagnostics ---
    pub lambda3_train_tpr: f64,
    pub lambda3_train_fpr: f64,
    pub h1_train_tpr: f64,
    pub h1_train_fpr: f64,
    pub n_fracture_windows: usize,
    pub n_stable_windows: usize,
}

/// Per-window detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub psi: f64,
    pub dpsi_dt: f64,
    pub topo_stress: f64,
    pub lambda3_fired: bool,
    pub h1_fired: bool,
    /// Combined OR trigger
    pub regime_fracture: bool,
}

/// Point cloud row for TDA embedding (4-dimensional)
#[derive(Debug, Clone)]
pub struct PointCloudRow {
    pub price: f64,
    pub ofi: f64,
    pub liquidity_grad: f64,
    pub vol_idx: f64,
}
