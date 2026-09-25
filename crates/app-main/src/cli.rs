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

        /// Headless only: save every 100th frame and the last one of each
        /// benchmark as a PPM image under bench/frames/<sha>/, to check by eye
        /// that a faster renderer still draws the same picture.
        #[arg(long)]
        save_frames: bool,
    },
    /// Generate SVG charts from benchmark results
    Chart,
    /// Print frame time percentiles for every recorded benchmark run
    Stats,
}
