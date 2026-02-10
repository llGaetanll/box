use clap::Parser;
use clap::Subcommand;

#[derive(Parser)]
#[command(name = "voxels")]
#[command(about = "A ray-traced voxel engine using rust-gpu")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Open a window and render the scene live
    Live,
    /// Run benchmark with animated camera path
    Bench {
        /// Benchmark definition name (loads from bench/configs/<name>.toml).
        /// If not specified, runs all benchmarks in the bench/configs/ directory.
        name: Option<String>,
    },
    /// Generate SVG charts from benchmark results
    Chart,
}
