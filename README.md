# saaq-latent

A focused, standalone Rust library for the **SAAQ latent-telemetry calibration
pipeline**: SAAQ update rules, latent snapshot structs, single-rule and
dual-rule calibrators, and the CSV schema consumed by downstream analysis
(e.g. `Surrogate_Viz.jl` / `SymbolicRegression.jl`).

It is a **copy-forward extraction** from
[`corinth-canal/src/latent.rs`](../corinth-canal/src/latent.rs). The
arithmetic, constants and CSV column layout are preserved exactly. The only
API changes are narrower crate-local input types so this crate does not
depend on corinth-canal's runtime structs.

## What this crate owns

- SAAQ update rules (`SaaqUpdateRule`)
- Latent snapshot (`SnnLatentSnapshot`, 16 columns)
- Single-rule latent calibration (`SnnLatentCalibrator`)
- Dual-rule latent calibration (`SnnDualLatentCalibrator`)
- Latent CSV emission (`SnnLatentCsvExporter`)
- Narrow input data-carriers (`TelemetryFrame`, `LatentActivitySummary`,
  `RoutingObservation`)
- Determinism + schema-stability tests

## What this crate does NOT own

- Telemetry collection / daemons. The caller builds a `TelemetryFrame` from
  whatever source it already has (`lm-sensors`, `nvml`, `sysinfo`, a log
  replay, fixtures, etc.).
- Symbolic regression. The CSV is consumed downstream by
  `SymbolicRegression.jl` / `Surrogate_Viz.jl`.
- Dashboards, plotting, or any visualization code.
- GGUF parsing.
- MoE router / expert forward implementations.
- GPU execution, kernel scheduling, CUDA/ROCm/Metal integration.
- Model orchestration, prompt routing, hybrid loops.
- corinth-canal-specific example bootstrap / config loading.

## Relationship to corinth-canal

- **Reference source**: `corinth-canal/src/latent.rs` (read-only).
- **No modifications** to `corinth-canal` are made by anything in this
  crate. This is a one-way extraction.
- The behavior of `SnnLatentCalibrator` and `SnnDualLatentCalibrator`
  (including every numeric constant and clamp) matches the reference
  implementation. The only deltas are:
  - Inputs are narrow crate-local types (`TelemetryFrame`,
    `LatentActivitySummary`, `RoutingObservation`) instead of corinth-canal's
    `TelemetrySnapshot`, `FunnelActivity`, `ModelOutput`.
  - The hidden-neuron count is taken from
    `LatentActivitySummary::hidden_neuron_count` rather than the
    corinth-canal `FUNNEL_HIDDEN_NEURONS` constant.
  - Errors are crate-local (`LatentError`) rather than corinth-canal's
    `HybridError`.
- The crate does not depend on corinth-canal as a library crate; it can be
  built, tested, and published independently.

## Relationship to `Surrogate_Viz.jl`

The CSV schema emitted by `SnnLatentCsvExporter` is intentionally
**byte-identical** to the one written by corinth-canal:

```
timestamp_ms,avg_pop_firing_rate_hz,membrane_dv_dt,routing_entropy,
saaq_delta_q_prev,saaq_delta_q_target,heartbeat_signal,heartbeat_enabled,
gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w,
saaq_delta_q_legacy_prev,saaq_delta_q_legacy_target,
saaq_delta_q_v15_prev,saaq_delta_q_v15_target
```

This is a stability contract: existing `SR.jl` / `Surrogate_Viz.jl` scripts
that load corinth-canal latent CSVs must keep working when pointed at a CSV
written by this crate.

The `saaq_delta_q_prev` / `saaq_delta_q_target` columns carry whichever rule
is designated primary (via `SnnDualLatentCalibrator::new(primary_rule)`), so
legacy scripts that only read those names stay valid while newer scripts can
target the explicit `*_legacy_*` / `*_v15_*` columns.

## Quick start

```rust
use saaq_latent::{
    LatentActivitySummary, RoutingObservation, SaaqUpdateRule,
    SnnLatentCalibrator, SnnLatentCsvExporter, TelemetryFrame,
};

let frame = TelemetryFrame {
    timestamp_ms: 1_000,
    heartbeat_signal: 0.0,
    heartbeat_enabled: false,
    gpu_temp_c: 60.0,
    gpu_power_w: 250.0,
    cpu_tctl_c: 70.0,
    cpu_package_power_w: 120.0,
};
let potentials = vec![0.25_f32; 16];
let weights = vec![0.7_f32, 0.2, 0.1];

let activity = LatentActivitySummary {
    total_hidden_spikes: 32,
    hidden_neuron_count: 16,
    potentials: &potentials,
};
let routing = RoutingObservation { expert_weights: &weights };

let mut calibrator =
    SnnLatentCalibrator::with_update_rule(SaaqUpdateRule::SaaqV1_5SqrtRate);
let snapshot = calibrator.observe(&frame, &activity, &routing).unwrap();

let mut exporter = SnnLatentCsvExporter::create("latent.csv").unwrap();
exporter.write_row(&snapshot).unwrap();
exporter.flush().unwrap();
```

See [`examples/single_row.rs`](examples/single_row.rs) for a runnable
version.

## Testing

```
cargo test
```

The unit tests in `src/lib.rs` cover:

- expected latent fields on a solo calibrator,
- SAAQ 1.5 sqrt-rate equation (`0.0573 * sqrt(rate) + 0.496 * prev`),
- update-rule switching on a solo calibrator,
- dual calibrator matches paired solo calibrators bit-for-bit,
- primary-rule routing of the legacy `saaq_delta_q_{prev,target}` columns,
- CSV header stability (exact string + dual-rule column presence) and
  row-count correctness.

## License

GPL-3.0-only.
