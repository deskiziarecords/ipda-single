/// Unified IPDA threshold calibration — runs λ₃ + H₁ together and
/// verifies the 100% win-rate constraint on the training set.

use anyhow::{anyhow, Result};

use crate::spectral::{calibrate_lambda3, Lambda3Detector};
use crate::topo::{calibrate_h1, H1Detector};
use crate::types::{IPDAThresholds, MarketWindow, DetectionResult};

/// Run the full calibration procedure.
///
/// # Arguments
/// * `fracture` - labelled windows that precede known regime fractures
/// * `stable`   - windows sampled from confirmed winning periods
/// * `seg_len`  - FFT segment length (default 64)
/// * `min_tpr`  - required recall on training set (1.0 = 100% win-rate constraint)
pub fn calibrate_ipda(
    fracture: &[MarketWindow],
    stable: &[MarketWindow],
    seg_len: usize,
    min_tpr: f64,
) -> Result<IPDAThresholds> {
    if fracture.is_empty() {
        return Err(anyhow!("Fracture set is empty — need at least 1 labelled window"));
    }
    if stable.is_empty() {
        return Err(anyhow!("Stable set is empty"));
    }

    // ── λ₃ spectral phase inversion ──────────────────────────────────────────
    let l3 = calibrate_lambda3(fracture, stable, seg_len, min_tpr)?;

    // ── H₁ persistent homology ───────────────────────────────────────────────
    let h1 = calibrate_h1(fracture, stable, min_tpr)?;

    let thresholds = IPDAThresholds {
        lambda3_psi_thresh: l3.psi_threshold,
        lambda3_angle_deg: l3.angle_deg,
        gamma_drift: l3.gamma_drift,
        gamma_topo: h1.gamma_topo,
        tau_topo: h1.tau_topo,
        lambda3_train_tpr: l3.train_tpr,
        lambda3_train_fpr: l3.train_fpr,
        h1_train_tpr: h1.train_tpr,
        h1_train_fpr: h1.train_fpr,
        n_fracture_windows: fracture.len(),
        n_stable_windows: stable.len(),
    };

    Ok(thresholds)
}

/// Verify the calibrated thresholds achieve zero false negatives on the
/// training fracture set (the 100% win-rate guarantee).
///
/// Returns `Ok(())` if every fracture window is caught by λ₃ OR H₁.
/// Returns `Err` listing any missed windows.
pub fn verify_zero_false_negatives(
    thresholds: &IPDAThresholds,
    fracture: &[MarketWindow],
    seg_len: usize,
    merge_tol: f64,
) -> Result<Vec<DetectionResult>> {
    use crate::spectral::compute_spectral_alignment;
    use crate::topo::{construct_point_cloud, compute_h1_persistence, compute_topo_stress, can_merge_charts};
    use crate::utils::{movmean, diff};

    let mut results = Vec::with_capacity(fracture.len());
    let mut missed = Vec::new();

    // Rolling Ψ buffer for dΨ/dt
    let mut psi_buf: Vec<f64> = Vec::new();

    for (idx, w) in fracture.iter().enumerate() {
        // λ₃
        let (psi, _) = compute_spectral_alignment(&w.price, &w.ofi, seg_len)?;
        psi_buf.push(psi);
        if psi_buf.len() > 5 { psi_buf.remove(0); }
        let dpsi_dt = if psi_buf.len() >= 2 {
            let sm = movmean(&psi_buf, 5);
            *diff(&sm).last().unwrap_or(&0.0)
        } else { 0.0 };

        // Only apply drift gate once we have enough history (>= 2 windows);
        // on the very first window Ψ level alone decides.
        let drift_ok = psi_buf.len() < 2 || dpsi_dt < thresholds.gamma_drift;
        let lambda3_fired = psi < thresholds.lambda3_psi_thresh && drift_ok;

        // H₁
        let cloud = construct_point_cloud(w);
        let pers = compute_h1_persistence(&cloud);
        let stress = compute_topo_stress(&pers, thresholds.gamma_topo);

        // Atlas merge: compare with adjacent windows if available
        let merge_failed = if idx > 0 && idx < fracture.len() - 1 {
            !can_merge_charts(&fracture[idx - 1], &fracture[idx + 1], merge_tol)
        } else { true }; // boundary: no adjacent context, gate open

        let h1_fired = stress > thresholds.tau_topo && merge_failed;
        let regime_fracture = lambda3_fired || h1_fired;

        if !regime_fracture {
            missed.push(idx);
        }

        results.push(DetectionResult {
            psi,
            dpsi_dt,
            topo_stress: stress,
            lambda3_fired,
            h1_fired,
            regime_fracture,
        });
    }

    if !missed.is_empty() {
        return Err(anyhow!(
            "VERIFICATION FAILED — {} fracture window(s) missed (indices: {:?}). \
             Relax thresholds: lower lambda3_psi_thresh or tau_topo.",
            missed.len(),
            missed
        ));
    }

    log::info!(
        "Verification PASSED — all {} fracture windows detected.",
        fracture.len()
    );

    Ok(results)
}

/// Print a formatted calibration summary to stdout.
pub fn print_calibration_summary(t: &IPDAThresholds) {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║          IPDA Threshold Calibration Results              ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  λ₃  Spectral Phase Inversion                           ║");
    println!("║    Ψ threshold  : {:<8.4}  ({:.1}°)                    ║",
             t.lambda3_psi_thresh, t.lambda3_angle_deg);
    println!("║    γ_drift      : {:<8.4} /window                      ║", t.gamma_drift);
    println!("║    Train TPR    : {:.1}%  FPR: {:.1}%                   ║",
             t.lambda3_train_tpr * 100.0, t.lambda3_train_fpr * 100.0);
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  H₁  Persistent Homology (TopoStress)                   ║");
    println!("║    γ_topo       : {:<8.4}  (noise floor)               ║", t.gamma_topo);
    println!("║    τ_topo       : {:<8.4}  (fracture threshold)        ║", t.tau_topo);
    println!("║    Train TPR    : {:.1}%  FPR: {:.1}%                   ║",
             t.h1_train_tpr * 100.0, t.h1_train_fpr * 100.0);
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Dataset: {} fracture / {} stable windows             ║",
             t.n_fracture_windows, t.n_stable_windows);
    println!("╚══════════════════════════════════════════════════════════╝\n");
}
