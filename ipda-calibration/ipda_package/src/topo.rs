/// Persistent Homology (H₁) — Vietoris-Rips via union-find filtration
///
/// Implements a lightweight 0/1-dimensional persistence computation
/// without an external TDA library.  For production use with higher
/// embedding dimensions, replace `compute_h1_persistence` with a GUDHI
/// or Ripser FFI call.
///
/// λ₄ fires when TopoStress > τ_topo AND the two adjacent atlas charts
/// cannot be merged (isneighbor analogue).

use anyhow::{anyhow, Result};

use crate::types::MarketWindow;
use crate::utils::{zscore, percentile, optimal_threshold_gt_min_fpr};

// ─────────────────────────────────────────────────────────────────────────────
// Point cloud construction
// ─────────────────────────────────────────────────────────────────────────────

/// Build a (N × 4) point cloud from a market window.
/// Columns: [z_price, z_ofi, z_liq_grad, z_vol_idx]
pub fn construct_point_cloud(window: &MarketWindow) -> Vec<[f64; 4]> {
    let n = window.price.len();

    let zp = zscore(&window.price);
    let zo = zscore(&window.ofi);

    let zlg = if window.liquidity_grad.is_empty() {
        vec![0.0_f64; n]
    } else {
        zscore(&window.liquidity_grad)
    };

    let zv = if window.vol_idx.is_empty() {
        vec![0.0_f64; n]
    } else {
        zscore(&window.vol_idx)
    };

    (0..n)
        .map(|i| [zp[i], zo[i], zlg[i], zv[i]])
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Euclidean distance matrix
// ─────────────────────────────────────────────────────────────────────────────

fn distance_matrix(pts: &[[f64; 4]]) -> Vec<Vec<f64>> {
    let n = pts.len();
    let mut dm = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = pts[i]
                .iter()
                .zip(pts[j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            dm[i][j] = d;
            dm[j][i] = d;
        }
    }
    dm
}

// ─────────────────────────────────────────────────────────────────────────────
// Union-Find for H₀ persistence (used to track H₁ loop birth/death)
// ─────────────────────────────────────────────────────────────────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    birth: Vec<f64>, // filtration value when component was born
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            birth: vec![0.0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Returns Some(death_value) if this edge creates a loop (H₁ event),
    /// None if it merges two components (H₀ event).
    fn union(&mut self, a: usize, b: usize, filtration: f64) -> Option<f64> {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            // Same component → loop born at `filtration`, dies at … ∞ (open)
            // We record birth only; death handled by elder-rule approximation
            return Some(filtration); // H₁ loop detected
        }
        // Merge by rank (elder rule: younger component dies)
        let (root, child) = if self.rank[ra] >= self.rank[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        // child component dies (persistence = filtration - birth[child])
        self.parent[child] = root;
        if self.rank[root] == self.rank[child] {
            self.rank[root] += 1;
        }
        self.birth[root] = self.birth[root].min(self.birth[child]);
        None // H₀ merge, not a loop
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// H₁ persistence via Vietoris-Rips filtration (dimension 1)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute H₁ (loop) persistence values from a point cloud.
///
/// Algorithm:
///   1. Sort all pairwise edges by distance (= filtration value).
///   2. Insert edges using union-find; track H₀ merges and H₁ loops.
///   3. H₁ loop persistence ≈ diameter of the loop (approximated as the
///      edge length that closes the cycle, minus the edge that opened it).
///
/// This is a simplified Rips persistence; for full Vietoris-Rips (dim ≥ 2),
/// replace with a GUDHI/Ripser FFI call.
///
/// Returns a vector of H₁ persistence lifespans (death − birth).
pub fn compute_h1_persistence(points: &[[f64; 4]]) -> Vec<f64> {
    let n = points.len();
    if n < 3 {
        return vec![];
    }

    let dm = distance_matrix(points);

    // Collect all edges sorted by distance
    let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            edges.push((dm[i][j], i, j));
        }
    }
    edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut uf = UnionFind::new(n);
    let mut h1_lifespans: Vec<f64> = Vec::new();

    // Track the filtration value when each node first enters
    // (all born at 0 in Rips since we start with all vertices)
    let mut edge_birth: std::collections::HashMap<(usize, usize), f64> =
        std::collections::HashMap::new();

    for (dist, i, j) in &edges {
        edge_birth.insert((*i, *j), *dist);
        if let Some(loop_birth) = uf.union(*i, *j, *dist) {
            // H₁ event: a cycle just closed.
            // Persistence ≈ current filtration value minus the longest edge
            // of the triangle that closed it (elder rule approximation).
            // Simple version: lifespan = dist (birth at 0 for Rips).
            // More accurate: find the youngest edge in the closing simplex.
            let persistence = *dist; // upper bound; refine if needed
            h1_lifespans.push(persistence);
        }
    }

    h1_lifespans
}

// ─────────────────────────────────────────────────────────────────────────────
// TopoStress metric
// ─────────────────────────────────────────────────────────────────────────────

/// Sum of significant H₁ loop lifespans (noise floor = gamma_topo).
pub fn compute_topo_stress(persistence: &[f64], gamma_topo: f64) -> f64 {
    persistence
        .iter()
        .filter(|&&p| p > gamma_topo)
        .sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Atlas merge check (isneighbor analogue)
// ─────────────────────────────────────────────────────────────────────────────

/// Simple atlas adjacency check.
///
/// Two charts are considered "mergeable" if the gap between the maximum of
/// one and the minimum of the other (in price space) is less than `merge_tol`
/// standard deviations of the pooled price distribution.
///
/// Replace this with your full `isneighbor.m` logic when available.
pub fn can_merge_charts(left: &MarketWindow, right: &MarketWindow, merge_tol: f64) -> bool {
    if left.price.is_empty() || right.price.is_empty() {
        return false;
    }
    let max_l = left.price.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_r = right.price.iter().cloned().fold(f64::INFINITY, f64::min);

    // Pooled std
    let all: Vec<f64> = left.price.iter().chain(right.price.iter()).cloned().collect();
    let mean = all.iter().sum::<f64>() / all.len() as f64;
    let std = (all.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / all.len() as f64).sqrt();

    if std < f64::EPSILON {
        return true;
    }

    (min_r - max_l).abs() / std < merge_tol
}

// ─────────────────────────────────────────────────────────────────────────────
// H₁ calibration
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct H1Calibration {
    pub gamma_topo: f64,
    pub tau_topo: f64,
    pub train_tpr: f64,
    pub train_fpr: f64,
}

/// Calibrate H₁ noise floor and fracture threshold.
pub fn calibrate_h1(
    fracture_windows: &[MarketWindow],
    stable_windows: &[MarketWindow],
    min_tpr: f64,
) -> Result<H1Calibration> {
    log::info!(
        "H₁ calibration: {} fracture, {} stable windows",
        fracture_windows.len(),
        stable_windows.len()
    );

    // Step 1: collect all H₁ lifespans from stable windows → noise floor
    let mut all_persist_stable: Vec<f64> = Vec::new();
    for w in stable_windows {
        let cloud = construct_point_cloud(w);
        let pers = compute_h1_persistence(&cloud);
        all_persist_stable.extend_from_slice(&pers);
    }

    if all_persist_stable.is_empty() {
        return Err(anyhow!("No H₁ loops found in stable windows — check window length"));
    }

    // gamma_topo = 99th percentile of stable H₁ persistence (noise floor)
    let gamma_topo = percentile(&all_persist_stable, 99.0)?;
    log::info!("H₁ γ_topo (noise floor, 99th pct stable) = {:.4}", gamma_topo);

    // Step 2: TopoStress for fracture set
    let mut stress_f: Vec<f64> = Vec::with_capacity(fracture_windows.len());
    for w in fracture_windows {
        let cloud = construct_point_cloud(w);
        let pers = compute_h1_persistence(&cloud);
        stress_f.push(compute_topo_stress(&pers, gamma_topo));
    }

    // Step 3: TopoStress for stable set
    let mut stress_s: Vec<f64> = Vec::with_capacity(stable_windows.len());
    for w in stable_windows {
        let cloud = construct_point_cloud(w);
        let pers = compute_h1_persistence(&cloud);
        stress_s.push(compute_topo_stress(&pers, gamma_topo));
    }

    log::debug!("TopoStress fracture: min={:.3} max={:.3}",
        stress_f.iter().cloned().fold(f64::INFINITY, f64::min),
        stress_f.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    log::debug!("TopoStress stable:   max={:.3}",
        stress_s.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    // Step 4: threshold optimisation — fire when stress > tau_topo
    // Use GT-ROC: find highest threshold where TPR >= min_tpr (zero missed fractures)
    let mut all_scores: Vec<f64> = stress_f.clone();
    all_scores.extend_from_slice(&stress_s);
    let mut all_labels: Vec<bool> = vec![true; stress_f.len()];
    all_labels.extend(vec![false; stress_s.len()]);

    let (tau_topo, tpr, fpr) = match optimal_threshold_gt_min_fpr(&all_scores, &all_labels, min_tpr) {
        Ok(result) => {
            log::info!("H₁ τ_topo = {:.4}, TPR={:.3}, FPR={:.3}", result.0, result.1, result.2);
            result
        }
        Err(e) => {
            // H₁ alone cannot achieve min_tpr on this dataset (classes not separable by
            // TopoStress alone).  Fall back to 5th-percentile of fracture stress minus margin.
            // λ₃ carries the 100% recall guarantee; H₁ acts as a secondary confirmatory sensor.
            log::warn!("H₁ GT-ROC: {} — falling back to fracture 5th-percentile threshold.", e);
            let p05 = percentile(&stress_f, 5.0)
                .unwrap_or_else(|_| stress_f.iter().cloned()
                    .fold(f64::INFINITY, f64::min));
            let fallback_tau = (p05 - 0.01).max(0.0);
            let achieved_tpr = all_scores.iter().zip(all_labels.iter())
                .filter(|(&s, &l)| s > fallback_tau && l).count() as f64
                / stress_f.len() as f64;
            let achieved_fpr = all_scores.iter().zip(all_labels.iter())
                .filter(|(&s, &l)| s > fallback_tau && !l).count() as f64
                / stress_s.len() as f64;
            log::warn!("H₁ fallback τ_topo = {:.4}, TPR={:.3}, FPR={:.3}",
                       fallback_tau, achieved_tpr, achieved_fpr);
            log::warn!("NOTE: H₁ is in confirmatory mode. λ₃ provides the primary 100% recall guarantee.");
            (fallback_tau, achieved_tpr, achieved_fpr)
        }
    };

    Ok(H1Calibration { gamma_topo, tau_topo, train_tpr: tpr, train_fpr: fpr })
}

// ─────────────────────────────────────────────────────────────────────────────
// H₁ run-time detector (λ₄)
// ─────────────────────────────────────────────────────────────────────────────

pub struct H1Detector {
    pub gamma_topo: f64,
    pub tau_topo: f64,
    pub merge_tol: f64,
}

impl H1Detector {
    pub fn new(cal: &H1Calibration, merge_tol: f64) -> Self {
        Self {
            gamma_topo: cal.gamma_topo,
            tau_topo: cal.tau_topo,
            merge_tol,
        }
    }

    /// Returns (fired, topo_stress).
    /// `left` and `right` are adjacent market windows (chart left/right).
    pub fn detect(
        &self,
        current: &MarketWindow,
        left: &MarketWindow,
        right: &MarketWindow,
    ) -> (bool, f64) {
        let cloud = construct_point_cloud(current);
        let pers = compute_h1_persistence(&cloud);
        let stress = compute_topo_stress(&pers, self.gamma_topo);
        let merge_ok = can_merge_charts(left, right, self.merge_tol);
        let fired = stress > self.tau_topo && !merge_ok;
        (fired, stress)
    }
}
