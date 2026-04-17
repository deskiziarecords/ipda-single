/// Data loading utilities — CSV format for market windows
///
/// Expected CSV schema (one row per bar):
///   window_id,is_fracture,price,ofi,liquidity_grad,vol_idx
///
/// All windows with the same `window_id` are grouped together.
/// `is_fracture` should be the same for all rows in a window (1 or 0).

use anyhow::{anyhow, Result};
use std::collections::BTreeMap;
use std::path::Path;

use crate::types::MarketWindow;

#[derive(Debug, serde::Deserialize)]
struct CsvRow {
    window_id: String,
    is_fracture: u8,
    price: f64,
    ofi: f64,
    #[serde(default)]
    liquidity_grad: f64,
    #[serde(default)]
    vol_idx: f64,
}

/// Load windows from a CSV file.
///
/// Returns `(fracture_windows, stable_windows)`.
pub fn load_windows_from_csv(path: &Path) -> Result<(Vec<MarketWindow>, Vec<MarketWindow>)> {
    let mut reader = csv::Reader::from_path(path)
        .map_err(|e| anyhow!("Cannot open CSV '{}': {}", path.display(), e))?;

    // Group rows by window_id
    let mut groups: BTreeMap<String, (bool, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> =
        BTreeMap::new();

    for result in reader.deserialize::<CsvRow>() {
        let row = result.map_err(|e| anyhow!("CSV parse error: {}", e))?;
        let entry = groups.entry(row.window_id.clone()).or_insert_with(|| {
            (row.is_fracture != 0, Vec::new(), Vec::new(), Vec::new(), Vec::new())
        });
        entry.1.push(row.price);
        entry.2.push(row.ofi);
        entry.3.push(row.liquidity_grad);
        entry.4.push(row.vol_idx);
    }

    let mut fracture = Vec::new();
    let mut stable = Vec::new();

    for (id, (is_frac, price, ofi, liq, vol)) in groups {
        let w = MarketWindow { price, ofi, liquidity_grad: liq, vol_idx: vol, is_fracture: is_frac };
        w.validate().map_err(|e| anyhow!("Window '{}' invalid: {}", id, e))?;
        if is_frac { fracture.push(w); } else { stable.push(w); }
    }

    log::info!("Loaded {} fracture + {} stable windows from '{}'",
               fracture.len(), stable.len(), path.display());

    Ok((fracture, stable))
}

/// Save calibrated thresholds as JSON.
pub fn save_thresholds_json(thresholds: &crate::types::IPDAThresholds, path: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(thresholds)?;
    std::fs::write(path, json)
        .map_err(|e| anyhow!("Cannot write thresholds to '{}': {}", path.display(), e))?;
    log::info!("Thresholds saved to '{}'", path.display());
    Ok(())
}

/// Load calibrated thresholds from JSON.
pub fn load_thresholds_json(path: &Path) -> Result<crate::types::IPDAThresholds> {
    let json = std::fs::read_to_string(path)
        .map_err(|e| anyhow!("Cannot read thresholds from '{}': {}", path.display(), e))?;
    let t = serde_json::from_str(&json)?;
    Ok(t)
}
