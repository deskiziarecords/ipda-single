/// ipda_detect — runtime regime fracture detector
///
/// Reads a stream of market windows from stdin (one CSV row per bar,
/// same schema as calibration), loads pre-calibrated thresholds from JSON,
/// and outputs a detection signal for each completed window.
///
/// Usage:
///   ipda_detect --thresholds thresholds.json [--seg-len 64] [--merge-tol 0.5]
///   cat live_feed.csv | ipda_detect --thresholds thresholds.json

use anyhow::Result;
use clap::Parser;
use std::collections::BTreeMap;
use std::io::{self, BufRead};
use std::path::PathBuf;

use ipda_calibration::loader::load_thresholds_json;
use ipda_calibration::spectral::compute_spectral_alignment;
use ipda_calibration::topo::{construct_point_cloud, compute_h1_persistence, compute_topo_stress};
use ipda_calibration::types::MarketWindow;
use ipda_calibration::utils::{movmean, diff};

#[derive(Parser, Debug)]
#[command(
    name = "ipda_detect",
    about = "IPDA Runtime Detector — reads market windows from stdin, outputs fracture signals",
    version = "0.1.0"
)]
struct Args {
    /// Path to calibrated thresholds JSON
    #[arg(short, long)]
    thresholds: PathBuf,

    /// FFT segment length (must match calibration)
    #[arg(long, default_value_t = 64)]
    seg_len: usize,

    /// Atlas merge tolerance
    #[arg(long, default_value_t = 0.5)]
    merge_tol: f64,

    /// Output JSON lines (default: human-readable)
    #[arg(long)]
    json: bool,
}

fn main() -> Result<()> {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "warn");
    }
    env_logger::init();

    let args = Args::parse();
    let t = load_thresholds_json(&args.thresholds)?;

    // Rolling state for dPsi/dt
    let mut psi_buf: Vec<f64> = Vec::new();
    let mut prev_window: Option<MarketWindow> = None;
    let mut prev_prev_window: Option<MarketWindow> = None;

    // Buffer accumulating CSV rows for the current window
    let mut row_buf: BTreeMap<String, (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> = BTreeMap::new();
    let mut current_window_id = String::new();

    // Print header
    if !args.json {
        println!("{:<12} {:>8} {:>10} {:>12} {:>8} {:>8} {:>14}",
                 "window_id", "psi", "dpsi_dt", "topo_stress", "lam3", "h1", "FRACTURE");
    }

    let stdin = io::stdin();
    let mut first_line = true;

    for line in stdin.lock().lines() {
        let line = line?;
        if first_line && line.starts_with("window_id") {
            first_line = false;
            continue; // skip header
        }
        first_line = false;

        let parts: Vec<&str> = line.trim().split(',').collect();
        if parts.len() < 4 {
            continue;
        }

        let wid = parts[0].to_string();
        let price: f64 = parts[2].parse().unwrap_or(0.0);
        let ofi: f64 = parts[3].parse().unwrap_or(0.0);
        let liq: f64 = if parts.len() > 4 { parts[4].parse().unwrap_or(0.0) } else { 0.0 };
        let vol: f64 = if parts.len() > 5 { parts[5].parse().unwrap_or(0.0) } else { 0.0 };

        if wid != current_window_id && !current_window_id.is_empty() {
            // Process completed window
            if let Some(entry) = row_buf.remove(&current_window_id) {
                let window = MarketWindow {
                    price: entry.0,
                    ofi: entry.1,
                    liquidity_grad: entry.2,
                    vol_idx: entry.3,
                    is_fracture: false,
                };

                // λ₃
                let (psi, _) = compute_spectral_alignment(&window.price, &window.ofi, args.seg_len)
                    .unwrap_or((0.0, 0.0));
                psi_buf.push(psi);
                if psi_buf.len() > 5 { psi_buf.remove(0); }
                let dpsi_dt = if psi_buf.len() >= 2 {
                    let sm = movmean(&psi_buf, 5);
                    *diff(&sm).last().unwrap_or(&0.0)
                } else { 0.0 };

                let lambda3 = psi < t.lambda3_psi_thresh && dpsi_dt < t.gamma_drift;

                // H₁
                let cloud = construct_point_cloud(&window);
                let pers = compute_h1_persistence(&cloud);
                let stress = compute_topo_stress(&pers, t.gamma_topo);

                // Atlas merge
                let merge_ok = match (&prev_prev_window, &prev_window) {
                    (Some(ll), Some(l)) => {
                        let max_ll = ll.price.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let min_l = l.price.iter().cloned().fold(f64::INFINITY, f64::min);
                        let all: Vec<f64> = ll.price.iter().chain(l.price.iter()).cloned().collect();
                        let mean = all.iter().sum::<f64>() / all.len() as f64;
                        let std = (all.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / all.len() as f64).sqrt();
                        if std < f64::EPSILON { true } else { (min_l - max_ll).abs() / std < args.merge_tol }
                    }
                    _ => true,
                };

                let h1 = stress > t.tau_topo && !merge_ok;
                let fracture = lambda3 || h1;

                if args.json {
                    println!(r#"{{"window_id":"{wid}","psi":{psi:.4},"dpsi_dt":{dpsi:.4},"topo_stress":{stress:.4},"lambda3":{lambda3},"h1":{h1},"regime_fracture":{fracture}}}"#,
                             wid = current_window_id,
                             psi = psi, dpsi = dpsi_dt, stress = stress,
                             lambda3 = lambda3, h1 = h1, fracture = fracture);
                } else {
                    let flag = if fracture { "*** FRACTURE ***" } else { "" };
                    println!("{:<12} {:>8.4} {:>10.4} {:>12.4} {:>8} {:>8} {:>14}",
                             current_window_id, psi, dpsi_dt, stress,
                             lambda3, h1, flag);
                }

                prev_prev_window = prev_window.take();
                prev_window = Some(window);
            }
        }

        current_window_id = wid.clone();
        let e = row_buf.entry(wid).or_insert_with(|| (vec![], vec![], vec![], vec![]));
        e.0.push(price);
        e.1.push(ofi);
        e.2.push(liq);
        e.3.push(vol);
    }

    Ok(())
}
