use std::collections::HashMap;
use std::error::Error;
use std::io::IsTerminal;
use std::path::Path;
use std::process::Command;

use util_bench::Percentiles;

const BENCH_RESULTS_DIR: &str = "bench/results";

/// Dim styling for rows whose commit git does not know about.
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

/// One benchmark run, flattened for display.
struct Row {
    sha: String,
    benchmark: String,
    size: String,
    mode: String,
    frames: String,
    percentiles: Percentiles,
    /// Position in git history, counting from the newest commit. `None` when the
    /// commit is not in this repository's history.
    age: Option<usize>,
}

impl Row {
    /// The columns in display order.
    fn cells(&self) -> [&str; 5] {
        [
            &self.sha,
            &self.benchmark,
            &self.size,
            &self.mode,
            &self.frames,
        ]
    }

    fn times(&self) -> [u64; 7] {
        let p = &self.percentiles;
        [p.min, p.p1, p.p25, p.p50, p.p75, p.p99, p.max]
    }
}

const TEXT_HEADERS: [&str; 5] = ["sha", "benchmark", "size", "mode", "frames"];
const TIME_HEADERS: [&str; 7] = ["min", "p1", "p25", "p50", "p75", "p99", "max"];

/// Map every commit in this repository to its distance from the newest commit, so
/// runs can be ordered by history rather than by when they happened to be run.
/// Commits reachable from no ref, such as ones rebased away, are simply absent.
fn commit_ages() -> HashMap<String, usize> {
    let output = Command::new("git")
        .args(["rev-list", "--topo-order", "--all"])
        .output();

    let stdout = match output {
        Ok(out) if out.status.success() => out.stdout,
        _ => return HashMap::new(),
    };

    String::from_utf8_lossy(&stdout)
        .lines()
        .enumerate()
        .map(|(age, sha)| (short_sha(sha.trim()), age))
        .collect()
}

/// Results are stored under abbreviated SHAs, so compare on the same prefix.
fn short_sha(sha: &str) -> String {
    sha.chars().take(7).collect()
}

pub fn run_stats() -> Result<(), Box<dyn Error>> {
    let dir = Path::new(BENCH_RESULTS_DIR);
    if !dir.exists() {
        return Err(format!("{BENCH_RESULTS_DIR}/ not found. Run some benchmarks first.").into());
    }

    let data = util_bench::load_all_benchmarks(dir)?;
    let ages = commit_ages();

    let mut rows: Vec<Row> = Vec::new();
    for benchmark in &data {
        for run in &benchmark.runs {
            let Some(percentiles) = Percentiles::from_frames(&run.frames) else {
                continue;
            };
            let [width, height] = run.metadata.resolution;

            rows.push(Row {
                age: ages.get(&run.sha).copied(),
                sha: run.sha.clone(),
                benchmark: benchmark.name.clone(),
                size: format!("{width}x{height}"),
                // A window's size is up to the window manager, so a run in one
                // is only comparable to another run in the same kind of window
                mode: if run.metadata.headless { "headless" } else { "window" }.to_string(),
                frames: run.frames.len().to_string(),
                percentiles,
            });
        }
    }

    if rows.is_empty() {
        return Err(format!("No benchmark data found in {BENCH_RESULTS_DIR}/").into());
    }

    // Oldest commit first so the newest run is the last thing printed. Commits git
    // cannot place have no position in that order, so they go above all of it.
    rows.sort_by(|a, b| match (a.age, b.age) {
        (Some(x), Some(y)) => y.cmp(&x).then_with(|| a.benchmark.cmp(&b.benchmark)),
        (None, None) => a
            .sha
            .cmp(&b.sha)
            .then_with(|| a.benchmark.cmp(&b.benchmark)),
        (None, Some(_)) => std::cmp::Ordering::Less,
        (Some(_), None) => std::cmp::Ordering::Greater,
    });

    print_table(&rows);

    Ok(())
}

fn print_table(rows: &[Row]) {
    let times: Vec<[u64; 7]> = rows.iter().map(Row::times).collect();

    let text_widths: Vec<usize> = TEXT_HEADERS
        .iter()
        .enumerate()
        .map(|(i, header)| {
            rows.iter()
                .map(|row| row.cells()[i].len())
                .chain([header.len()])
                .max()
                .unwrap_or(0)
        })
        .collect();

    let time_widths: Vec<usize> = TIME_HEADERS
        .iter()
        .enumerate()
        .map(|(i, header)| {
            times
                .iter()
                .map(|row| count_digits(row[i]))
                .chain([header.len()])
                .max()
                .unwrap_or(0)
        })
        .collect();

    let mut header = String::new();
    for (i, label) in TEXT_HEADERS.iter().enumerate() {
        // Text columns read better left aligned, numbers right aligned
        let aligned = if i <= 3 {
            format!("{label:<width$}", width = text_widths[i])
        } else {
            format!("{label:>width$}", width = text_widths[i])
        };
        header.push_str(&aligned);
        header.push_str("  ");
    }
    for (i, label) in TIME_HEADERS.iter().enumerate() {
        header.push_str(&format!("{label:>width$}", width = time_widths[i]));
        if i + 1 < TIME_HEADERS.len() {
            header.push_str("  ");
        }
    }
    println!("{header}");

    let dim = std::io::stdout().is_terminal();

    for (row, times) in rows.iter().zip(&times) {
        let mut line = String::new();

        for (i, cell) in row.cells().iter().enumerate() {
            let aligned = if i <= 3 {
                format!("{cell:<width$}", width = text_widths[i])
            } else {
                format!("{cell:>width$}", width = text_widths[i])
            };
            line.push_str(&aligned);
            line.push_str("  ");
        }
        for (i, time) in times.iter().enumerate() {
            line.push_str(&format!("{time:>width$}", width = time_widths[i]));
            if i + 1 < times.len() {
                line.push_str("  ");
            }
        }

        if row.age.is_none() && dim {
            println!("{DIM}{line}{RESET}");
        } else {
            println!("{line}");
        }
    }
}

fn count_digits(value: u64) -> usize {
    value.to_string().len()
}
