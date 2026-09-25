use clap::Parser;
use clap::Subcommand;

#[derive(Parser)]
#[command(name = "box")]
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

        /// Render offscreen instead of opening a window, at exactly the size each
        /// config asks for.
        #[arg(long)]
        headless: bool,
    },
    /// Generate SVG charts from benchmark results
    Chart,
}
