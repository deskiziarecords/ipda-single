/// Statistical utility functions — zscore, percentile, ROC optimisation

use anyhow::{anyhow, Result};

/// Z-score normalise a slice in place (returns new Vec)
pub fn zscore(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    if std < f64::EPSILON {
        return vec![0.0; x.len()];
    }
    x.iter().map(|v| (v - mean) / std).collect()
}

/// Linear detrend: subtract best-fit line
pub fn detrend(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 2 {
        return x.to_vec();
    }
    let nf = n as f64;
    let idx: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let sum_i: f64 = idx.iter().sum();
    let sum_i2: f64 = idx.iter().map(|i| i * i).sum();
    let sum_x: f64 = x.iter().sum();
    let sum_ix: f64 = idx.iter().zip(x.iter()).map(|(i, v)| i * v).sum();

    let denom = nf * sum_i2 - sum_i * sum_i;
    if denom.abs() < f64::EPSILON {
        return x.to_vec();
    }
    let slope = (nf * sum_ix - sum_i * sum_x) / denom;
    let intercept = (sum_x - slope * sum_i) / nf;

    x.iter()
        .enumerate()
        .map(|(i, v)| v - (slope * i as f64 + intercept))
        .collect()
}

/// Compute percentile (linear interpolation) — requires sorted copy
pub fn percentile(x: &[f64], p: f64) -> Result<f64> {
    if x.is_empty() {
        return Err(anyhow!("percentile of empty slice"));
    }
    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        return Ok(sorted[lo]);
    }
    let frac = idx - lo as f64;
    Ok(sorted[lo] * (1.0 - frac) + sorted[hi] * frac)
}

/// Centred moving average (same length as input, pads with edge values)
pub fn movmean(x: &[f64], window: usize) -> Vec<f64> {
    let n = x.len();
    let half = window / 2;
    (0..n)
        .map(|i| {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(n);
            let slice = &x[lo..hi];
            slice.iter().sum::<f64>() / slice.len() as f64
        })
        .collect()
}

/// Discrete first difference
pub fn diff(x: &[f64]) -> Vec<f64> {
    x.windows(2).map(|w| w[1] - w[0]).collect()
}

/// ROC curve — returns (thresholds, tpr, fpr)
///
/// `scores`:  continuous score for each sample (higher → more likely positive)
/// `labels`:  true = positive class
///
/// Threshold semantics: **positive when score < thresh** (matching Ψ < θ convention).
pub fn roc_curve_lt(scores: &[f64], labels: &[bool], n_steps: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let min_s = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let step = (max_s - min_s) / n_steps as f64;

    let n_pos = labels.iter().filter(|&&l| l).count() as f64;
    let n_neg = labels.iter().filter(|&&l| !l).count() as f64;

    let mut thresholds = Vec::with_capacity(n_steps + 1);
    let mut tpr_vec = Vec::with_capacity(n_steps + 1);
    let mut fpr_vec = Vec::with_capacity(n_steps + 1);

    for k in 0..=n_steps {
        let thresh = min_s + k as f64 * step;
        let tp: f64 = scores
            .iter()
            .zip(labels.iter())
            .filter(|(&s, &l)| s < thresh && l)
            .count() as f64;
        let fp: f64 = scores
            .iter()
            .zip(labels.iter())
            .filter(|(&s, &l)| s < thresh && !l)
            .count() as f64;

        thresholds.push(thresh);
        tpr_vec.push(if n_pos > 0.0 { tp / n_pos } else { 0.0 });
        fpr_vec.push(if n_neg > 0.0 { fp / n_neg } else { 0.0 });
    }

    (thresholds, tpr_vec, fpr_vec)
}

/// Find the threshold achieving TPR >= min_tpr with minimum FPR.
/// Returns (threshold, achieved_tpr, achieved_fpr).
pub fn optimal_threshold_min_fpr(
    scores: &[f64],
    labels: &[bool],
    min_tpr: f64,
) -> Result<(f64, f64, f64)> {
    let (thresholds, tpr, fpr) = roc_curve_lt(scores, labels, 200);

    let candidates: Vec<usize> = tpr
        .iter()
        .enumerate()
        .filter(|(_, &t)| t >= min_tpr)
        .map(|(i, _)| i)
        .collect();

    if candidates.is_empty() {
        return Err(anyhow!(
            "No threshold achieves TPR >= {:.3}. Max TPR = {:.3}",
            min_tpr,
            tpr.iter().cloned().fold(0.0_f64, f64::max)
        ));
    }

    let best = candidates
        .into_iter()
        .min_by(|&a, &b| fpr[a].partial_cmp(&fpr[b]).unwrap())
        .unwrap();

    Ok((thresholds[best], tpr[best], fpr[best]))
}

/// ROC curve — positive when score **>** threshold (for TopoStress: fire when stress > tau).
pub fn roc_curve_gt(scores: &[f64], labels: &[bool], n_steps: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let min_s = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let step = if (max_s - min_s).abs() < f64::EPSILON { 1.0 } else { (max_s - min_s) / n_steps as f64 };

    let n_pos = labels.iter().filter(|&&l| l).count() as f64;
    let n_neg = labels.iter().filter(|&&l| !l).count() as f64;

    let mut thresholds = Vec::with_capacity(n_steps + 1);
    let mut tpr_vec    = Vec::with_capacity(n_steps + 1);
    let mut fpr_vec    = Vec::with_capacity(n_steps + 1);

    for k in 0..=n_steps {
        let thresh = min_s + k as f64 * step;
        let tp: f64 = scores.iter().zip(labels.iter()).filter(|(&s,&l)| s > thresh &&  l).count() as f64;
        let fp: f64 = scores.iter().zip(labels.iter()).filter(|(&s,&l)| s > thresh && !l).count() as f64;
        thresholds.push(thresh);
        tpr_vec.push(if n_pos > 0.0 { tp / n_pos } else { 0.0 });
        fpr_vec.push(if n_neg > 0.0 { fp / n_neg } else { 0.0 });
    }
    (thresholds, tpr_vec, fpr_vec)
}

/// Find the highest threshold where (score > thresh) still achieves TPR >= min_tpr,
/// with minimum FPR. Returns (threshold, achieved_tpr, achieved_fpr).
pub fn optimal_threshold_gt_min_fpr(scores: &[f64], labels: &[bool], min_tpr: f64) -> Result<(f64, f64, f64)> {
    let (thresholds, tpr, fpr) = roc_curve_gt(scores, labels, 400);

    let candidates: Vec<usize> = tpr.iter().enumerate()
        .filter(|(_, &t)| t >= min_tpr)
        .map(|(i, _)| i)
        .collect();

    if candidates.is_empty() {
        return Err(anyhow!(
            "No threshold achieves TPR >= {:.3} (score > thresh). Max TPR = {:.3}. \
             Ensure fracture windows have higher TopoStress than stable windows.",
            min_tpr,
            tpr.iter().cloned().fold(0.0_f64, f64::max)
        ));
    }

    // Lowest FPR; break ties by highest threshold (most specific detector)
    let best = candidates.into_iter()
        .min_by(|&a, &b| fpr[a].partial_cmp(&fpr[b]).unwrap()
            .then(thresholds[b].partial_cmp(&thresholds[a]).unwrap()))
        .unwrap();

    Ok((thresholds[best], tpr[best], fpr[best]))
}
