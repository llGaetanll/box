mod camera_path;
mod chart;
mod percentiles;
mod spline;
mod types;

use std::collections::HashMap;
use std::fs;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
pub use camera_path::CameraFrame;
pub use camera_path::CameraPath;
pub use chart::generate_svg;
pub use percentiles::Percentiles;
pub use spline::CatmullRomSpline;
pub use types::BenchmarkMetadata;
pub use types::FrameRecord;
pub use types::GpuInfo;

/// A single frame's data: frame number and render time.
#[derive(Clone)]
pub struct FrameData {
    pub frame: u32,
    pub time_us: u64,
}

/// A parsed benchmark run with metadata and frame data.
pub struct BenchmarkRun {
    pub metadata: BenchmarkMetadata,
    pub frames: Vec<FrameData>,
}

/// Data for a single SHA's benchmark run, including timestamp for ordering.
#[derive(Clone)]
pub struct ShaRun {
    pub sha: String,
    pub timestamp: String,
    pub frames: Vec<FrameData>,
    pub metadata: BenchmarkMetadata,
}

/// All runs for a single benchmark, grouped by SHA.
pub struct BenchmarkData {
    pub name: String,
    pub runs: Vec<ShaRun>,
}

/// Load a benchmark run from a JSONL file.
pub fn load_benchmark_run(path: &Path) -> Result<BenchmarkRun> {
    let file =
        fs::File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // First line is metadata
    let metadata_line = lines.next().context("Empty benchmark file")??;
    let metadata: BenchmarkMetadata =
        serde_json::from_str(&metadata_line).context("Failed to parse benchmark metadata")?;

    // Remaining lines are frame records
    let mut frames = Vec::new();
    for line in lines {
        let line = line?;
        let record: FrameRecord = serde_json::from_str(&line)?;
        frames.push(FrameData {
            frame: record.frame,
            time_us: record.time_us,
        });
    }

    Ok(BenchmarkRun { metadata, frames })
}

/// Parsed benchmark filename components.
struct BenchmarkFilename {
    timestamp: String,
    name: String,
}

/// Parse a benchmark filename like "2026-01-15-23-45-09-two_spheres.jsonl"
fn parse_benchmark_filename(filename: &str) -> Option<BenchmarkFilename> {
    let stem = filename.strip_suffix(".jsonl")?;
    // Format: YYYY-MM-DD-HH-MM-SS-name
    // The timestamp is 19 chars (2026-01-15-23-45-09) plus a dash
    if stem.len() < 21 {
        return None;
    }
    let timestamp = &stem[..19];
    let name = &stem[20..];
    Some(BenchmarkFilename {
        timestamp: timestamp.to_string(),
        name: name.to_string(),
    })
}

/// Load all benchmark data from bench/results directory.
///
/// Returns a list of BenchmarkData, each containing runs sorted by timestamp (oldest first).
/// Only includes the most recent run per SHA per benchmark.
pub fn load_all_benchmarks(bench_results_dir: &Path) -> Result<Vec<BenchmarkData>> {
    /// One loaded run, keyed by benchmark name and then SHA while collecting.
    struct LoadedRun {
        timestamp: String,
        frames: Vec<FrameData>,
        metadata: BenchmarkMetadata,
    }

    let mut all_runs: HashMap<String, HashMap<String, LoadedRun>> = HashMap::new();

    for sha_entry in fs::read_dir(bench_results_dir)? {
        let sha_entry = sha_entry?;
        let sha_path = sha_entry.path();
        if !sha_path.is_dir() {
            continue;
        }
        let sha = sha_entry.file_name().to_string_lossy().to_string();

        for file_entry in fs::read_dir(&sha_path)? {
            let file_entry = file_entry?;
            let file_path = file_entry.path();
            let filename = file_entry.file_name().to_string_lossy().to_string();

            if !filename.ends_with(".jsonl") {
                continue;
            }

            let parsed = match parse_benchmark_filename(&filename) {
                Some(p) => p,
                None => continue,
            };

            let run = match load_benchmark_run(&file_path) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Warning: failed to load {}: {}", file_path.display(), e);
                    continue;
                }
            };

            let by_sha = all_runs.entry(parsed.name.clone()).or_default();
            let entry = by_sha.entry(sha.clone());

            let loaded = LoadedRun {
                timestamp: parsed.timestamp,
                frames: run.frames,
                metadata: run.metadata,
            };

            match entry {
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(loaded);
                }
                std::collections::hash_map::Entry::Occupied(mut o) => {
                    if loaded.timestamp > o.get().timestamp {
                        o.insert(loaded);
                    }
                }
            }
        }
    }

    let mut result: Vec<BenchmarkData> = all_runs
        .into_iter()
        .map(|(name, by_sha)| {
            let mut runs: Vec<ShaRun> = by_sha
                .into_iter()
                .map(|(sha, run)| ShaRun {
                    sha,
                    timestamp: run.timestamp,
                    frames: run.frames,
                    metadata: run.metadata,
                })
                .collect();
            runs.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
            BenchmarkData { name, runs }
        })
        .collect();

    result.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(result)
}
