/// Spectral phase alignment (λ₃) — direct Rust port of MATLAB procedure
///
/// Computes Ψ(t) = cos(phase_angle at dominant frequency of cross-PSD between
/// price and order-flow imbalance).  Ψ → +1 = aligned, Ψ → -1 = inverted.

use anyhow::{anyhow, Result};
use num_complex::Complex64;
use rustfft::FftPlanner;

use crate::utils::{detrend, movmean, diff, optimal_threshold_min_fpr, percentile};
use crate::types::MarketWindow;

// ─────────────────────────────────────────────────────────────────────────────
// Hamming window
// ─────────────────────────────────────────────────────────────────────────────

fn hamming(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Welch PSD — single-sided, magnitude squared
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Welch power spectral density.
/// Returns (psd, frequency_bins).
/// Uses non-overlapping segments of length `seg_len` with a Hamming window.
fn pwelch(x: &[f64], seg_len: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if x.len() < seg_len {
        return Err(anyhow!("Signal shorter than FFT segment length {}", seg_len));
    }

    let win = hamming(seg_len);
    let win_power: f64 = win.iter().map(|w| w * w).sum::<f64>();

    let n_segs = x.len() / seg_len;
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(seg_len);

    let n_freq = seg_len / 2 + 1;
    let mut psd_acc = vec![0.0_f64; n_freq];

    for s in 0..n_segs {
        let seg = &x[s * seg_len..(s + 1) * seg_len];
        let mut buf: Vec<Complex64> = seg
            .iter()
            .zip(win.iter())
            .map(|(v, w)| Complex64::new(v * w, 0.0))
            .collect();
        fft.process(&mut buf);
        for (k, val) in buf[..n_freq].iter().enumerate() {
            psd_acc[k] += val.norm_sqr() / (win_power * n_segs as f64);
        }
    }

    // Single-sided: double non-DC, non-Nyquist bins
    for k in 1..(n_freq - 1) {
        psd_acc[k] *= 2.0;
    }

    let freqs: Vec<f64> = (0..n_freq).map(|k| k as f64 / seg_len as f64).collect();

    Ok((psd_acc, freqs))
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-PSD — Welch cross-spectral density
// ─────────────────────────────────────────────────────────────────────────────

/// Returns complex cross-PSD (price, ofi) for each frequency bin.
fn cpsd(x: &[f64], y: &[f64], seg_len: usize) -> Result<Vec<Complex64>> {
    if x.len() != y.len() {
        return Err(anyhow!("cpsd: x and y must have equal length"));
    }
    if x.len() < seg_len {
        return Err(anyhow!("cpsd: signal shorter than seg_len"));
    }

    let win = hamming(seg_len);
    let win_power: f64 = win.iter().map(|w| w * w).sum::<f64>();
    let n_segs = x.len() / seg_len;
    let n_freq = seg_len / 2 + 1;

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(seg_len);

    let mut cpsd_acc = vec![Complex64::new(0.0, 0.0); n_freq];

    for s in 0..n_segs {
        let seg_x = &x[s * seg_len..(s + 1) * seg_len];
        let seg_y = &y[s * seg_len..(s + 1) * seg_len];

        let mut buf_x: Vec<Complex64> = seg_x
            .iter()
            .zip(win.iter())
            .map(|(v, w)| Complex64::new(v * w, 0.0))
            .collect();
        let mut buf_y: Vec<Complex64> = seg_y
            .iter()
            .zip(win.iter())
            .map(|(v, w)| Complex64::new(v * w, 0.0))
            .collect();

        fft.process(&mut buf_x);
        fft.process(&mut buf_y);

        for k in 0..n_freq {
            // Pxy = X * conj(Y)
            cpsd_acc[k] += buf_x[k] * buf_y[k].conj() / (win_power * n_segs as f64);
        }
    }

    Ok(cpsd_acc)
}

// ─────────────────────────────────────────────────────────────────────────────
// Core: compute_spectral_alignment
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Ψ(t) for a single window.
///
/// Returns `(psi, dominant_freq)` where:
///   - `psi` = cos(phase angle at dominant frequency of cross-PSD)
///   - `dominant_freq` = normalised frequency [0, 0.5] of peak average PSD
pub fn compute_spectral_alignment(price: &[f64], ofi: &[f64], seg_len: usize) -> Result<(f64, f64)> {
    let price_dt = detrend(price);
    let ofi_dt = detrend(ofi);

    let (pxx, freqs) = pwelch(&price_dt, seg_len)?;
    let (pxx_ofi, _) = pwelch(&ofi_dt, seg_len)?;
    let cxy = cpsd(&price_dt, &ofi_dt, seg_len)?;

    // Dominant frequency = peak of average PSD
    let avg_psd: Vec<f64> = pxx
        .iter()
        .zip(pxx_ofi.iter())
        .map(|(a, b)| (a + b) / 2.0)
        .collect();

    let idx_max = avg_psd
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let f_dom = freqs[idx_max];
    let phase_angle = cxy[idx_max].arg(); // atan2(im, re)
    let psi = phase_angle.cos();

    Ok((psi, f_dom))
}

// ─────────────────────────────────────────────────────────────────────────────
// λ₃ calibration
// ─────────────────────────────────────────────────────────────────────────────

/// Output of λ₃ calibration
#[derive(Debug, Clone)]
pub struct Lambda3Calibration {
    pub psi_threshold: f64,
    pub angle_deg: f64,
    pub gamma_drift: f64,
    pub train_tpr: f64,
    pub train_fpr: f64,
}

/// Calibrate λ₃ thresholds from labelled windows.
///
/// `min_tpr` should be 1.0 for the 100% win-rate constraint.
pub fn calibrate_lambda3(
    fracture_windows: &[MarketWindow],
    stable_windows: &[MarketWindow],
    seg_len: usize,
    min_tpr: f64,
) -> Result<Lambda3Calibration> {
    log::info!(
        "λ₃ calibration: {} fracture, {} stable windows, seg_len={}",
        fracture_windows.len(),
        stable_windows.len(),
        seg_len
    );

    // Step 1: compute Ψ for every window
    let mut psi_f: Vec<f64> = Vec::with_capacity(fracture_windows.len());
    for w in fracture_windows {
        let (psi, _) = compute_spectral_alignment(&w.price, &w.ofi, seg_len)?;
        psi_f.push(psi);
    }

    let mut psi_s: Vec<f64> = Vec::with_capacity(stable_windows.len());
    for w in stable_windows {
        let (psi, _) = compute_spectral_alignment(&w.price, &w.ofi, seg_len)?;
        psi_s.push(psi);
    }

    log::debug!("Ψ fracture: mean={:.3}, min={:.3}", mean(&psi_f), psi_f.iter().cloned().fold(f64::INFINITY, f64::min));
    log::debug!("Ψ stable:   mean={:.3}, max={:.3}", mean(&psi_s), psi_s.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    // Step 2: ROC optimisation — fire when Ψ < threshold
    let mut all_scores: Vec<f64> = psi_f.clone();
    all_scores.extend_from_slice(&psi_s);

    let mut all_labels: Vec<bool> = vec![true; psi_f.len()];
    all_labels.extend(vec![false; psi_s.len()]);

    let (psi_thresh, tpr, fpr) = optimal_threshold_min_fpr(&all_scores, &all_labels, min_tpr)
        .map_err(|e| anyhow!("λ₃ threshold search failed: {}", e))?;

    log::info!("λ₃ Ψ threshold = {:.4} ({:.1}°), TPR={:.3}, FPR={:.3}",
               psi_thresh, psi_thresh.acos().to_degrees(), tpr, fpr);

    // Step 3: drift calibration (5th percentile of dΨ/dt in fracture set)
    let psi_smooth = movmean(&psi_f, 5);
    let dpsi = diff(&psi_smooth);
    // If only one fracture window, gamma_drift falls back to a conservative default
    let gamma_drift = if dpsi.is_empty() {
        -0.15
    } else {
        percentile(&dpsi, 5.0)?
    };

    log::info!("λ₃ γ_drift = {:.4}/window", gamma_drift);

    Ok(Lambda3Calibration {
        psi_threshold: psi_thresh,
        angle_deg: psi_thresh.acos().to_degrees(),
        gamma_drift,
        train_tpr: tpr,
        train_fpr: fpr,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// λ₃ run-time detection
// ─────────────────────────────────────────────────────────────────────────────

/// Run-time λ₃ detector.
///
/// Maintains a small rolling buffer of Ψ values to compute dΨ/dt.
pub struct Lambda3Detector {
    pub psi_threshold: f64,
    pub gamma_drift: f64,
    psi_history: Vec<f64>,
    smooth_window: usize,
    pub seg_len: usize,
}

impl Lambda3Detector {
    pub fn new(cal: &Lambda3Calibration, seg_len: usize) -> Self {
        Self {
            psi_threshold: cal.psi_threshold,
            gamma_drift: cal.gamma_drift,
            psi_history: Vec::new(),
            smooth_window: 5,
            seg_len,
        }
    }

    /// Feed a new window; returns (fired, psi, dpsi_dt).
    pub fn update(&mut self, window: &MarketWindow) -> Result<(bool, f64, f64)> {
        let (psi, _) = compute_spectral_alignment(&window.price, &window.ofi, self.seg_len)?;
        self.psi_history.push(psi);

        // Keep only last smooth_window values
        if self.psi_history.len() > self.smooth_window {
            self.psi_history.remove(0);
        }

        let dpsi_dt = if self.psi_history.len() >= 2 {
            let smooth = movmean(&self.psi_history, self.smooth_window);
            let d = diff(&smooth);
            *d.last().unwrap_or(&0.0)
        } else {
            0.0
        };

        let drift_ok = self.psi_history.len() < 2 || dpsi_dt < self.gamma_drift;
        let fired = psi < self.psi_threshold && drift_ok;
        Ok((fired, psi, dpsi_dt))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn mean(x: &[f64]) -> f64 {
    if x.is_empty() { return 0.0; }
    x.iter().sum::<f64>() / x.len() as f64
}
