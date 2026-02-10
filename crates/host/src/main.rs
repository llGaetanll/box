use std::error::Error;
use std::fs;
use std::path::Path;

use clap::Parser;

mod bench_app;
mod cli;
mod gpu;
mod live_app;
mod window_surface;

use cli::Cli;
use cli::Commands;

fn run_chart() -> Result<(), Box<dyn Error>> {
    let bench_results_dir = Path::new("bench/results");
    if !bench_results_dir.exists() {
        return Err("bench/results/ directory not found. Run some benchmarks first.".into());
    }

    let data = voxels_bench::load_all_benchmarks(bench_results_dir)?;
    if data.is_empty() {
        return Err("No benchmark data found in bench/results/".into());
    }

    log::debug!(
        "Loaded {} benchmark(s): {}",
        data.len(),
        data.iter()
            .map(|b| b.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    let svg = voxels_bench::generate_svg(&data);

    fs::create_dir_all("bench/charts")?;
    let output_path = "bench/charts/chart.svg";
    fs::write(output_path, &svg)?;

    log::info!("Saved {}", output_path);
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Live) => live_app::run_live(),
        Some(Commands::Bench { name: Some(name) }) => bench_app::run_bench(name),
        Some(Commands::Bench { name: None }) => bench_app::run_all_benchmarks(),
        Some(Commands::Chart) => run_chart(),
        None => live_app::run_live(),
    }
}
