/// ipda_calibrate — calibration binary
///
/// Usage:
///   ipda_calibrate --data windows.csv --output thresholds.json [--seg-len 64] [--min-tpr 1.0]
///
/// CSV format: window_id,is_fracture,price,ofi,liquidity_grad,vol_idx

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use ipda_calibration::calibration::{calibrate_ipda, verify_zero_false_negatives, print_calibration_summary};
use ipda_calibration::loader::{load_windows_from_csv, save_thresholds_json};

#[derive(Parser, Debug)]
#[command(
    name = "ipda_calibrate",
    about = "IPDA Threshold Calibration — lambda3 Spectral Phase Inversion + H1 Persistent Homology",
    version = "0.1.0"
)]
struct Args {
    /// Path to input CSV file
    #[arg(short, long)]
    data: PathBuf,

    /// Output path for calibrated thresholds (JSON)
    #[arg(short, long, default_value = "thresholds.json")]
    output: PathBuf,

    /// FFT segment length (default: 64 bars)
    #[arg(long, default_value_t = 64)]
    seg_len: usize,

    /// Minimum TPR on training set (1.0 = 100% win-rate constraint)
    #[arg(long, default_value_t = 1.0)]
    min_tpr: f64,

    /// Atlas merge tolerance (std-dev units)
    #[arg(long, default_value_t = 0.5)]
    merge_tol: f64,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.verbose {
        std::env::set_var("RUST_LOG", "debug");
    } else if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    log::info!("=== IPDA Calibration ===");
    log::info!("Data: {} | seg_len: {} | min_tpr: {:.3}",
               args.data.display(), args.seg_len, args.min_tpr);

    let (fracture, stable) = load_windows_from_csv(&args.data)?;
    log::info!("Fracture: {} | Stable: {}", fracture.len(), stable.len());

    let thresholds = calibrate_ipda(&fracture, &stable, args.seg_len, args.min_tpr)?;
    print_calibration_summary(&thresholds);

    log::info!("Verifying zero false negatives on training set...");
    verify_zero_false_negatives(&thresholds, &fracture, args.seg_len, args.merge_tol)
        .map_err(|e| { log::error!("{}", e); e })?;

    save_thresholds_json(&thresholds, &args.output)?;
    println!("Calibration complete. Thresholds -> '{}'", args.output.display());

    Ok(())
}
