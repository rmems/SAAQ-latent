//! # saaq-latent
//!
//! Standalone library that owns the SAAQ latent-telemetry calibration pipeline:
//! SAAQ update rules, latent snapshot structs, single-rule and dual-rule
//! calibrators, and the CSV emission schema consumed by downstream analysis
//! (e.g. `Surrogate_Viz.jl`).
//!
//! The logic is a copy-forward extraction from
//! `corinth-canal/src/latent.rs`. The arithmetic, constants and CSV column
//! layout are preserved; the only API changes are narrower crate-local input
//! types (`TelemetryFrame`, `LatentActivitySummary`, `RoutingObservation`) so
//! this crate does not depend on corinth-canal runtime structs.
//!
//! ## What this crate owns
//! - SAAQ update rules ([`SaaqUpdateRule`])
//! - Latent snapshot ([`SnnLatentSnapshot`])
//! - Single-rule calibration ([`SnnLatentCalibrator`])
//! - Dual-rule calibration ([`SnnDualLatentCalibrator`])
//! - CSV emission ([`SnnLatentCsvExporter`])
//!
//! ## What this crate does NOT own
//! - Telemetry collection (caller builds [`TelemetryFrame`]).
//! - Symbolic regression (downstream, via the emitted CSV).
//! - Dashboards / plots.
//! - GGUF parsing, MoE / router forward, GPU execution, model orchestration.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Minimal narrow telemetry data-carrier consumed by the calibrators.
///
/// Replaces corinth-canal's broader `TelemetrySnapshot`: only the fields
/// actually read by the latent pipeline are exposed.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct TelemetryFrame {
    pub timestamp_ms: u64,
    pub heartbeat_signal: f32,
    pub heartbeat_enabled: bool,
    pub gpu_temp_c: f32,
    pub gpu_power_w: f32,
    pub cpu_tctl_c: f32,
    pub cpu_package_power_w: f32,
}

/// Borrowed summary of one tick of SNN hidden-layer activity.
///
/// The caller is expected to have already reduced their spike-train
/// representation down to:
/// - `total_hidden_spikes`: the total number of hidden spikes across *all*
///   time-buckets in the tick,
/// - `hidden_neuron_count`: the number of hidden neurons (denominator of
///   the population firing rate),
/// - `potentials`: the per-neuron membrane potentials at the tick.
///
/// This keeps the crate agnostic to any particular spike-train layout.
#[derive(Debug, Clone, Copy)]
pub struct LatentActivitySummary<'a> {
    pub total_hidden_spikes: usize,
    pub hidden_neuron_count: usize,
    pub potentials: &'a [f32],
}

/// Borrowed routing observation used to compute normalized routing entropy.
///
/// Replaces corinth-canal's `ModelOutput::expert_weights: Option<Vec<f32>>`
/// with a plain slice; the empty-slice case is reported as
/// [`LatentError::MissingExpertWeights`].
#[derive(Debug, Clone, Copy)]
pub struct RoutingObservation<'a> {
    pub expert_weights: &'a [f32],
}

/// Errors surfaced by the calibrators and CSV exporter.
#[derive(Debug)]
pub enum LatentError {
    /// The routing observation had no expert weights. Mirrors the source's
    /// `missing expert_weights in ModelOutput` error path.
    MissingExpertWeights,
    /// I/O error while writing the latent CSV.
    Io(io::Error),
}

impl std::fmt::Display for LatentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LatentError::MissingExpertWeights => {
                write!(f, "missing expert_weights in RoutingObservation")
            }
            LatentError::Io(err) => write!(f, "latent CSV I/O error: {err}"),
        }
    }
}

impl std::error::Error for LatentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LatentError::Io(err) => Some(err),
            LatentError::MissingExpertWeights => None,
        }
    }
}

impl From<io::Error> for LatentError {
    fn from(err: io::Error) -> Self {
        LatentError::Io(err)
    }
}

/// Crate-local `Result` alias returning [`LatentError`].
pub type Result<T> = std::result::Result<T, LatentError>;

/// SAAQ update rule selector. `LegacyV1_0` is the original SAAQ 1.0 rule;
/// `SaaqV1_5SqrtRate` is the sqrt-rate-based SAAQ 1.5 rule.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum SaaqUpdateRule {
    #[default]
    LegacyV1_0,
    SaaqV1_5SqrtRate,
}

/// One emitted latent-telemetry row.
///
/// Column names and order match corinth-canal's `SnnLatentSnapshot` exactly
/// so existing `Surrogate_Viz.jl` / SR.jl pipelines continue to load the CSV
/// unchanged.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SnnLatentSnapshot {
    pub timestamp_ms: u64,
    pub avg_pop_firing_rate_hz: f32,
    pub membrane_dv_dt: f32,
    pub routing_entropy: f32,
    /// Legacy SAAQ column carrying whichever rule is designated primary.
    /// Preserved so existing SR.jl scripts that reference this name keep
    /// working after dual-rule emission was added.
    pub saaq_delta_q_prev: f32,
    /// Legacy SAAQ column carrying whichever rule is designated primary.
    pub saaq_delta_q_target: f32,
    pub heartbeat_signal: f32,
    pub heartbeat_enabled: bool,
    pub gpu_temp_c: f32,
    pub gpu_power_w: f32,
    pub cpu_tctl_c: f32,
    pub cpu_package_power_w: f32,
    /// SAAQ 1.0 (`LegacyV1_0`) previous target at this tick. Emitted by
    /// [`SnnDualLatentCalibrator`] so SR.jl can fit candidate laws against
    /// either rule without rerunning the whole sweep.
    pub saaq_delta_q_legacy_prev: f32,
    pub saaq_delta_q_legacy_target: f32,
    /// SAAQ 1.5 (`SaaqV1_5SqrtRate`) previous target at this tick.
    pub saaq_delta_q_v15_prev: f32,
    pub saaq_delta_q_v15_target: f32,
}

/// Single-rule latent calibrator.
///
/// Evolves the SAAQ delta-Q state under exactly one [`SaaqUpdateRule`] and
/// produces a [`SnnLatentSnapshot`] per observed tick. The dual-rule
/// `*_legacy_*` / `*_v15_*` snapshot columns are left at their defaults and
/// are only filled in by [`SnnDualLatentCalibrator`].
#[derive(Debug, Clone, Default)]
pub struct SnnLatentCalibrator {
    prev_timestamp_ms: Option<u64>,
    prev_mean_membrane: Option<f32>,
    prev_delta_q: f32,
    update_rule: SaaqUpdateRule,
}

impl SnnLatentCalibrator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_update_rule(update_rule: SaaqUpdateRule) -> Self {
        Self {
            update_rule,
            ..Self::default()
        }
    }

    pub fn set_update_rule(&mut self, update_rule: SaaqUpdateRule) {
        self.update_rule = update_rule;
    }

    pub fn update_rule(&self) -> SaaqUpdateRule {
        self.update_rule
    }

    pub fn observe(
        &mut self,
        frame: &TelemetryFrame,
        activity: &LatentActivitySummary<'_>,
        routing: &RoutingObservation<'_>,
    ) -> Result<SnnLatentSnapshot> {
        let dt_ms = self.window_dt_ms(frame.timestamp_ms);
        let dt_seconds = (dt_ms as f32 / 1000.0).max(1e-3);
        let hidden_neurons = activity.hidden_neuron_count.max(1) as f32;
        let total_hidden_spikes = activity.total_hidden_spikes as f32;
        let avg_pop_firing_rate_hz = total_hidden_spikes / hidden_neurons / dt_seconds;

        let mean_membrane = mean(activity.potentials);
        let previous_mean_membrane = self.prev_mean_membrane.unwrap_or(mean_membrane);
        let membrane_dv_dt = (mean_membrane - previous_mean_membrane) / dt_seconds;

        if routing.expert_weights.is_empty() {
            return Err(LatentError::MissingExpertWeights);
        }
        let routing_entropy = normalized_entropy(routing.expert_weights);

        let saaq_delta_q_prev = self.prev_delta_q;
        let saaq_delta_q_target = match self.update_rule {
            SaaqUpdateRule::LegacyV1_0 => {
                let activity_pressure = (avg_pop_firing_rate_hz / 24.0).clamp(0.0, 1.0);
                let membrane_pressure = (membrane_dv_dt / 12.0).clamp(-1.0, 1.0);
                0.52 * saaq_delta_q_prev
                    + 0.28 * activity_pressure
                    + 0.12 * membrane_pressure
                    + 0.20 * routing_entropy
                    - 0.18
            }
            SaaqUpdateRule::SaaqV1_5SqrtRate => {
                0.0573 * avg_pop_firing_rate_hz.max(0.0).sqrt() + 0.496 * saaq_delta_q_prev
            }
        };

        self.prev_timestamp_ms = Some(frame.timestamp_ms);
        self.prev_mean_membrane = Some(mean_membrane);
        self.prev_delta_q = saaq_delta_q_target;

        // Solo calibrators only populate the legacy `saaq_delta_q_*` columns.
        // The dual-rule fields are left at their defaults (0.0) and filled in
        // by `SnnDualLatentCalibrator::observe` when both rules are run
        // together.
        Ok(SnnLatentSnapshot {
            timestamp_ms: frame.timestamp_ms,
            avg_pop_firing_rate_hz,
            membrane_dv_dt,
            routing_entropy,
            saaq_delta_q_prev,
            saaq_delta_q_target,
            heartbeat_signal: frame.heartbeat_signal,
            heartbeat_enabled: frame.heartbeat_enabled,
            gpu_temp_c: frame.gpu_temp_c,
            gpu_power_w: frame.gpu_power_w,
            cpu_tctl_c: frame.cpu_tctl_c,
            cpu_package_power_w: frame.cpu_package_power_w,
            ..Default::default()
        })
    }

    fn window_dt_ms(&self, timestamp_ms: u64) -> u64 {
        match self.prev_timestamp_ms {
            Some(prev_timestamp_ms) if timestamp_ms > prev_timestamp_ms => {
                timestamp_ms - prev_timestamp_ms
            }
            _ => 1,
        }
    }
}

/// CSV exporter for [`SnnLatentSnapshot`] rows.
///
/// The header and row format are byte-identical to the corinth-canal
/// reference implementation so downstream Julia / Python analysis scripts
/// keep working.
#[derive(Debug)]
pub struct SnnLatentCsvExporter {
    writer: BufWriter<File>,
}

impl SnnLatentCsvExporter {
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut writer = BufWriter::new(File::create(path)?);
        writeln!(
            writer,
            "timestamp_ms,avg_pop_firing_rate_hz,membrane_dv_dt,routing_entropy,saaq_delta_q_prev,saaq_delta_q_target,heartbeat_signal,heartbeat_enabled,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w,saaq_delta_q_legacy_prev,saaq_delta_q_legacy_target,saaq_delta_q_v15_prev,saaq_delta_q_v15_target"
        )?;
        Ok(Self { writer })
    }

    pub fn write_row(&mut self, snapshot: &SnnLatentSnapshot) -> Result<()> {
        writeln!(
            self.writer,
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            snapshot.timestamp_ms,
            snapshot.avg_pop_firing_rate_hz,
            snapshot.membrane_dv_dt,
            snapshot.routing_entropy,
            snapshot.saaq_delta_q_prev,
            snapshot.saaq_delta_q_target,
            snapshot.heartbeat_signal,
            snapshot.heartbeat_enabled as u8,
            snapshot.gpu_temp_c,
            snapshot.gpu_power_w,
            snapshot.cpu_tctl_c,
            snapshot.cpu_package_power_w,
            snapshot.saaq_delta_q_legacy_prev,
            snapshot.saaq_delta_q_legacy_target,
            snapshot.saaq_delta_q_v15_prev,
            snapshot.saaq_delta_q_v15_target,
        )?;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

/// Dual-rule latent calibrator that runs `LegacyV1_0` (SAAQ 1.0) and
/// `SaaqV1_5SqrtRate` (SAAQ 1.5) in parallel over the same
/// `(frame, activity, routing)` stream.
///
/// Emits a single [`SnnLatentSnapshot`] per observation with *both* SAAQ
/// trajectories populated. The `primary_rule` selects which of the two rules
/// fills the legacy `saaq_delta_q_{prev,target}` columns so existing
/// SymbolicRegression.jl scripts (which read those names) remain valid.
///
/// Why dual-emit matters: the feature columns fed to SR.jl
/// (`avg_pop_firing_rate_hz`, `membrane_dv_dt`, `routing_entropy`) are
/// computed from activity/routing only and do **not** depend on which SAAQ
/// rule is being evolved. Running two solo calibrators against identical
/// inputs therefore produces paired `(X, y_legacy)` and `(X, y_v15)` data
/// with no run-to-run noise, at the cost of a few extra arithmetic ops.
#[derive(Debug, Clone)]
pub struct SnnDualLatentCalibrator {
    legacy: SnnLatentCalibrator,
    v15: SnnLatentCalibrator,
    primary_rule: SaaqUpdateRule,
}

impl SnnDualLatentCalibrator {
    pub fn new(primary_rule: SaaqUpdateRule) -> Self {
        Self {
            legacy: SnnLatentCalibrator::with_update_rule(SaaqUpdateRule::LegacyV1_0),
            v15: SnnLatentCalibrator::with_update_rule(SaaqUpdateRule::SaaqV1_5SqrtRate),
            primary_rule,
        }
    }

    pub fn primary_rule(&self) -> SaaqUpdateRule {
        self.primary_rule
    }

    /// Observe one tick against both rules and return a merged snapshot.
    ///
    /// The returned snapshot has:
    /// - `saaq_delta_q_{legacy,v15}_{prev,target}` populated from each rule,
    /// - `saaq_delta_q_{prev,target}` populated from the [`primary_rule`],
    /// - all other fields taken from the legacy calibrator (identical across
    ///   rules by construction, since they are computed from activity alone).
    pub fn observe(
        &mut self,
        frame: &TelemetryFrame,
        activity: &LatentActivitySummary<'_>,
        routing: &RoutingObservation<'_>,
    ) -> Result<SnnLatentSnapshot> {
        let legacy_snapshot = self.legacy.observe(frame, activity, routing)?;
        let v15_snapshot = self.v15.observe(frame, activity, routing)?;

        let (primary_prev, primary_target) = match self.primary_rule {
            SaaqUpdateRule::LegacyV1_0 => (
                legacy_snapshot.saaq_delta_q_prev,
                legacy_snapshot.saaq_delta_q_target,
            ),
            SaaqUpdateRule::SaaqV1_5SqrtRate => (
                v15_snapshot.saaq_delta_q_prev,
                v15_snapshot.saaq_delta_q_target,
            ),
        };

        Ok(SnnLatentSnapshot {
            timestamp_ms: legacy_snapshot.timestamp_ms,
            avg_pop_firing_rate_hz: legacy_snapshot.avg_pop_firing_rate_hz,
            membrane_dv_dt: legacy_snapshot.membrane_dv_dt,
            routing_entropy: legacy_snapshot.routing_entropy,
            saaq_delta_q_prev: primary_prev,
            saaq_delta_q_target: primary_target,
            heartbeat_signal: legacy_snapshot.heartbeat_signal,
            heartbeat_enabled: legacy_snapshot.heartbeat_enabled,
            gpu_temp_c: legacy_snapshot.gpu_temp_c,
            gpu_power_w: legacy_snapshot.gpu_power_w,
            cpu_tctl_c: legacy_snapshot.cpu_tctl_c,
            cpu_package_power_w: legacy_snapshot.cpu_package_power_w,
            saaq_delta_q_legacy_prev: legacy_snapshot.saaq_delta_q_prev,
            saaq_delta_q_legacy_target: legacy_snapshot.saaq_delta_q_target,
            saaq_delta_q_v15_prev: v15_snapshot.saaq_delta_q_prev,
            saaq_delta_q_v15_target: v15_snapshot.saaq_delta_q_target,
        })
    }
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn normalized_entropy(weights: &[f32]) -> f32 {
    if weights.len() <= 1 {
        return 0.0;
    }

    let entropy = weights
        .iter()
        .copied()
        .filter(|weight| weight.is_finite() && *weight > 0.0)
        .map(|weight| -weight * weight.ln())
        .sum::<f32>();
    let max_entropy = (weights.len() as f32).ln();

    if max_entropy > 0.0 {
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hidden-neuron count used by every test. Picked once so the test
    /// bodies stay tight; the ported assertions all check *relative*
    /// equalities (not absolute firing-rate magnitudes), so the exact
    /// value is irrelevant.
    const TEST_HIDDEN_NEURONS: usize = 16;
    /// The source's `FunnelActivity` carried a 4-bucket spike train
    /// (`vec![vec![0; hidden_spike_count]; 4]`). We reproduce the same
    /// total-count arithmetic here.
    const TEST_SPIKE_BUCKETS: usize = 4;

    fn sample_potentials(mean_potential: f32) -> Vec<f32> {
        vec![mean_potential; TEST_HIDDEN_NEURONS]
    }

    fn total_hidden_spikes(hidden_spike_count: usize) -> usize {
        TEST_SPIKE_BUCKETS * hidden_spike_count
    }

    #[test]
    fn calibrator_emits_expected_latent_fields() {
        let mut calibrator = SnnLatentCalibrator::new();
        let frame_a = TelemetryFrame {
            timestamp_ms: 1_000,
            gpu_temp_c: 60.0,
            gpu_power_w: 250.0,
            cpu_tctl_c: 70.0,
            cpu_package_power_w: 120.0,
            heartbeat_signal: 0.0,
            heartbeat_enabled: false,
        };
        let frame_b = TelemetryFrame {
            timestamp_ms: 1_100,
            ..frame_a.clone()
        };
        let weights = vec![0.7, 0.2, 0.1];
        let routing = RoutingObservation {
            expert_weights: &weights,
        };

        let potentials_a = sample_potentials(0.20);
        let activity_a = LatentActivitySummary {
            total_hidden_spikes: total_hidden_spikes(0),
            hidden_neuron_count: TEST_HIDDEN_NEURONS,
            potentials: &potentials_a,
        };
        let first = calibrator.observe(&frame_a, &activity_a, &routing).unwrap();

        let potentials_b = sample_potentials(0.35);
        let activity_b = LatentActivitySummary {
            total_hidden_spikes: total_hidden_spikes(8),
            hidden_neuron_count: TEST_HIDDEN_NEURONS,
            potentials: &potentials_b,
        };
        let second = calibrator.observe(&frame_b, &activity_b, &routing).unwrap();

        assert_eq!(first.timestamp_ms, 1_000);
        assert_eq!(first.saaq_delta_q_prev, 0.0);
        assert!(first.routing_entropy > 0.0);
        assert!(second.avg_pop_firing_rate_hz > 0.0);
        assert!(second.membrane_dv_dt > 0.0);
        assert!((second.saaq_delta_q_prev - first.saaq_delta_q_target).abs() < 1e-6);
    }

    #[test]
    fn exporter_writes_expected_header_and_row_count() {
        let path = std::env::temp_dir().join(format!(
            "saaq_latent_{}.csv",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let mut exporter = SnnLatentCsvExporter::create(&path).unwrap();
        exporter
            .write_row(&SnnLatentSnapshot {
                timestamp_ms: 42,
                avg_pop_firing_rate_hz: 1.0,
                membrane_dv_dt: -0.5,
                routing_entropy: 0.7,
                saaq_delta_q_prev: 0.1,
                saaq_delta_q_target: 0.2,
                heartbeat_signal: 0.0,
                heartbeat_enabled: false,
                gpu_temp_c: 60.0,
                gpu_power_w: 250.0,
                cpu_tctl_c: 70.0,
                cpu_package_power_w: 120.0,
                ..Default::default()
            })
            .unwrap();
        exporter.flush().unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let mut lines = contents.lines();
        assert_eq!(
            lines.next().unwrap(),
            "timestamp_ms,avg_pop_firing_rate_hz,membrane_dv_dt,routing_entropy,saaq_delta_q_prev,saaq_delta_q_target,heartbeat_signal,heartbeat_enabled,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w,saaq_delta_q_legacy_prev,saaq_delta_q_legacy_target,saaq_delta_q_v15_prev,saaq_delta_q_v15_target"
        );
        assert!(lines.next().is_some());
        assert!(lines.next().is_none());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn saaq_v1_5_uses_sqrt_rate_equation() {
        let mut calibrator =
            SnnLatentCalibrator::with_update_rule(SaaqUpdateRule::SaaqV1_5SqrtRate);
        let frame_a = TelemetryFrame {
            timestamp_ms: 1_000,
            gpu_temp_c: 60.0,
            gpu_power_w: 250.0,
            cpu_tctl_c: 70.0,
            cpu_package_power_w: 120.0,
            heartbeat_signal: 0.0,
            heartbeat_enabled: false,
        };
        let frame_b = TelemetryFrame {
            timestamp_ms: 1_100,
            ..frame_a.clone()
        };
        let weights = vec![0.7, 0.2, 0.1];
        let routing = RoutingObservation {
            expert_weights: &weights,
        };

        let potentials_a = sample_potentials(0.20);
        let activity_a = LatentActivitySummary {
            total_hidden_spikes: total_hidden_spikes(0),
            hidden_neuron_count: TEST_HIDDEN_NEURONS,
            potentials: &potentials_a,
        };
        let first = calibrator.observe(&frame_a, &activity_a, &routing).unwrap();

        let potentials_b = sample_potentials(0.35);
        let activity_b = LatentActivitySummary {
            total_hidden_spikes: total_hidden_spikes(8),
            hidden_neuron_count: TEST_HIDDEN_NEURONS,
            potentials: &potentials_b,
        };
        let second = calibrator.observe(&frame_b, &activity_b, &routing).unwrap();

        assert_eq!(first.saaq_delta_q_target, 0.0);
        let expected =
            0.0573 * second.avg_pop_firing_rate_hz.sqrt() + 0.496 * second.saaq_delta_q_prev;
        assert!((second.saaq_delta_q_target - expected).abs() < 1e-6);
    }

    #[test]
    fn calibrator_update_rule_can_be_changed() {
        let mut calibrator = SnnLatentCalibrator::new();
        assert_eq!(calibrator.update_rule(), SaaqUpdateRule::LegacyV1_0);
        calibrator.set_update_rule(SaaqUpdateRule::SaaqV1_5SqrtRate);
        assert_eq!(calibrator.update_rule(), SaaqUpdateRule::SaaqV1_5SqrtRate);
    }

    #[test]
    fn dual_calibrator_matches_solo_calibrators_bit_for_bit() {
        let mut legacy_solo = SnnLatentCalibrator::with_update_rule(SaaqUpdateRule::LegacyV1_0);
        let mut v15_solo = SnnLatentCalibrator::with_update_rule(SaaqUpdateRule::SaaqV1_5SqrtRate);
        let mut dual = SnnDualLatentCalibrator::new(SaaqUpdateRule::SaaqV1_5SqrtRate);

        let weights = vec![0.6, 0.3, 0.1];
        let routing = RoutingObservation {
            expert_weights: &weights,
        };
        let base = TelemetryFrame {
            timestamp_ms: 1_000,
            gpu_temp_c: 62.0,
            gpu_power_w: 245.0,
            cpu_tctl_c: 71.0,
            cpu_package_power_w: 118.0,
            heartbeat_signal: 0.2,
            heartbeat_enabled: true,
        };

        for step in 0..6 {
            let frame = TelemetryFrame {
                timestamp_ms: 1_000 + step as u64 * 50,
                ..base.clone()
            };
            let potentials = sample_potentials(0.15 + 0.05 * step as f32);
            let activity = LatentActivitySummary {
                total_hidden_spikes: total_hidden_spikes(step * 2),
                hidden_neuron_count: TEST_HIDDEN_NEURONS,
                potentials: &potentials,
            };

            let legacy = legacy_solo.observe(&frame, &activity, &routing).unwrap();
            let v15 = v15_solo.observe(&frame, &activity, &routing).unwrap();
            let merged = dual.observe(&frame, &activity, &routing).unwrap();

            assert_eq!(merged.saaq_delta_q_legacy_prev, legacy.saaq_delta_q_prev);
            assert_eq!(merged.saaq_delta_q_legacy_target, legacy.saaq_delta_q_target);
            assert_eq!(merged.saaq_delta_q_v15_prev, v15.saaq_delta_q_prev);
            assert_eq!(merged.saaq_delta_q_v15_target, v15.saaq_delta_q_target);
            // Primary rule is v1.5, so legacy compatibility columns should
            // track the v15 trajectory.
            assert_eq!(merged.saaq_delta_q_prev, v15.saaq_delta_q_prev);
            assert_eq!(merged.saaq_delta_q_target, v15.saaq_delta_q_target);
            // Shared feature columns are rule-independent.
            assert_eq!(merged.avg_pop_firing_rate_hz, legacy.avg_pop_firing_rate_hz);
            assert_eq!(merged.avg_pop_firing_rate_hz, v15.avg_pop_firing_rate_hz);
            assert_eq!(merged.routing_entropy, legacy.routing_entropy);
        }
    }

    #[test]
    fn dual_calibrator_primary_rule_legacy_fills_legacy_columns() {
        let mut dual = SnnDualLatentCalibrator::new(SaaqUpdateRule::LegacyV1_0);
        let weights = vec![0.5, 0.3, 0.2];
        let routing = RoutingObservation {
            expert_weights: &weights,
        };
        let frame = TelemetryFrame {
            timestamp_ms: 2_000,
            gpu_temp_c: 65.0,
            gpu_power_w: 240.0,
            cpu_tctl_c: 70.0,
            cpu_package_power_w: 120.0,
            heartbeat_signal: 0.0,
            heartbeat_enabled: false,
        };
        let potentials = sample_potentials(0.25);
        let activity = LatentActivitySummary {
            total_hidden_spikes: total_hidden_spikes(4),
            hidden_neuron_count: TEST_HIDDEN_NEURONS,
            potentials: &potentials,
        };
        let merged = dual.observe(&frame, &activity, &routing).unwrap();
        // Legacy primary -> saaq_delta_q_{prev,target} should equal the
        // *_legacy_* fields, not the *_v15_* fields.
        assert_eq!(merged.saaq_delta_q_prev, merged.saaq_delta_q_legacy_prev);
        assert_eq!(merged.saaq_delta_q_target, merged.saaq_delta_q_legacy_target);
    }

    #[test]
    fn exporter_header_includes_dual_saaq_columns() {
        let path = std::env::temp_dir().join(format!(
            "saaq_latent_dual_{}.csv",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut exporter = SnnLatentCsvExporter::create(&path).unwrap();
        exporter.flush().unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        let header = contents.lines().next().unwrap();
        for column in [
            "saaq_delta_q_legacy_prev",
            "saaq_delta_q_legacy_target",
            "saaq_delta_q_v15_prev",
            "saaq_delta_q_v15_target",
        ] {
            assert!(header.contains(column), "missing column: {column}");
        }
        let _ = std::fs::remove_file(path);
    }
}
