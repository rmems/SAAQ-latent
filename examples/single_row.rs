//! Minimal end-to-end example: build one tick of inputs, run the SAAQ 1.5
//! calibrator, and write a single CSV row.
//!
//! Run with:
//! ```ignore
//! cargo run --example single_row
//! ```

use saaq_latent::{
    LatentActivitySummary, RoutingObservation, SaaqUpdateRule, SnnLatentCalibrator,
    SnnLatentCsvExporter, TelemetryFrame,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let frame = TelemetryFrame {
        timestamp_ms: 1_000,
        heartbeat_signal: 0.0,
        heartbeat_enabled: false,
        gpu_temp_c: 62.5,
        gpu_power_w: 248.0,
        cpu_tctl_c: 71.0,
        cpu_package_power_w: 118.0,
    };

    let potentials: Vec<f32> = vec![0.25; 16];
    let expert_weights: Vec<f32> = vec![0.7, 0.2, 0.1];

    let activity = LatentActivitySummary {
        total_hidden_spikes: 32,
        hidden_neuron_count: potentials.len(),
        potentials: &potentials,
    };
    let routing = RoutingObservation {
        expert_weights: &expert_weights,
    };

    let mut calibrator = SnnLatentCalibrator::with_update_rule(SaaqUpdateRule::SaaqV1_5SqrtRate);
    let snapshot = calibrator.observe(&frame, &activity, &routing)?;

    let out_path = std::env::temp_dir().join("saaq_latent_example.csv");
    let mut exporter = SnnLatentCsvExporter::create(&out_path)?;
    exporter.write_row(&snapshot)?;
    exporter.flush()?;

    println!("wrote one SAAQ latent row to {}", out_path.display());
    println!(
        "  avg_pop_firing_rate_hz={:.6}  saaq_delta_q_target={:.6}  routing_entropy={:.6}",
        snapshot.avg_pop_firing_rate_hz, snapshot.saaq_delta_q_target, snapshot.routing_entropy,
    );
    Ok(())
}
