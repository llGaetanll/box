use crate::FrameData;

/// Frame time percentiles for one benchmark run, in microseconds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Percentiles {
    pub min: u64,
    pub p1: u64,
    pub p25: u64,
    pub p50: u64,
    pub p75: u64,
    pub p99: u64,
    pub max: u64,
}

impl Percentiles {
    /// Summarize a run. Returns `None` for a run with no frames.
    pub fn from_frames(frames: &[FrameData]) -> Option<Self> {
        if frames.is_empty() {
            return None;
        }

        let mut times: Vec<u64> = frames.iter().map(|f| f.time_us).collect();
        times.sort_unstable();

        Some(Self {
            min: times[0],
            p1: nearest_rank(&times, 1.0),
            p25: nearest_rank(&times, 25.0),
            p50: nearest_rank(&times, 50.0),
            p75: nearest_rank(&times, 75.0),
            p99: nearest_rank(&times, 99.0),
            max: times[times.len() - 1],
        })
    }
}

/// Nearest-rank percentile: the smallest value at or above which `percent` of the
/// samples fall. Returns an observed sample rather than interpolating between two,
/// so every column of the table is a frame time that actually happened.
fn nearest_rank(sorted: &[u64], percent: f64) -> u64 {
    let rank = (percent / 100.0 * sorted.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(sorted.len() - 1);

    sorted[index]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames(times: &[u64]) -> Vec<FrameData> {
        times
            .iter()
            .enumerate()
            .map(|(i, &time_us)| FrameData {
                frame: i as u32,
                time_us,
            })
            .collect()
    }

    #[test]
    fn empty_run_has_no_percentiles() {
        assert_eq!(Percentiles::from_frames(&[]), None);
    }

    #[test]
    fn single_frame_is_every_percentile() {
        let p = Percentiles::from_frames(&frames(&[42])).unwrap();

        assert_eq!(p.min, 42);
        assert_eq!(p.p50, 42);
        assert_eq!(p.max, 42);
    }

    #[test]
    fn percentiles_of_one_to_hundred() {
        let times: Vec<u64> = (1..=100).collect();
        let p = Percentiles::from_frames(&frames(&times)).unwrap();

        assert_eq!(p.min, 1);
        assert_eq!(p.p1, 1);
        assert_eq!(p.p25, 25);
        assert_eq!(p.p50, 50);
        assert_eq!(p.p75, 75);
        assert_eq!(p.p99, 99);
        assert_eq!(p.max, 100);
    }

    #[test]
    fn input_order_does_not_matter() {
        let ascending = Percentiles::from_frames(&frames(&[1, 2, 3, 4, 5])).unwrap();
        let shuffled = Percentiles::from_frames(&frames(&[4, 1, 5, 3, 2])).unwrap();

        assert_eq!(ascending, shuffled);
    }

    /// A startup spike belongs to max, not to the middle of the distribution.
    #[test]
    fn outlier_does_not_move_the_median() {
        let p = Percentiles::from_frames(&frames(&[10, 10, 10, 10, 9000])).unwrap();

        assert_eq!(p.p50, 10);
        assert_eq!(p.max, 9000);
    }
}
