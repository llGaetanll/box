use serde::Deserialize;
use serde::Serialize;

use crate::CameraPath;

/// GPU information captured from the wgpu adapter.
#[derive(Serialize, Deserialize, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub driver: String,
    pub backend: String,
}

impl GpuInfo {
    pub fn new(name: String, driver: String, backend: String) -> Self {
        Self {
            name,
            driver,
            backend,
        }
    }
}

/// Benchmark metadata written as the first line of the JSONL output.
#[derive(Serialize, Deserialize, Clone)]
pub struct BenchmarkMetadata {
    pub version: u32,
    pub timestamp: String,
    pub git_sha: String,
    pub scene: String,
    pub resolution: [u32; 2],
    /// Rendered offscreen rather than into a window. Absent from runs recorded
    /// before headless mode existed, all of which had a window.
    #[serde(default)]
    pub headless: bool,
    pub gpu: GpuInfo,
    pub camera_path: CameraPath,
}

/// Per-frame timing and camera data.
#[derive(Serialize, Deserialize)]
pub struct FrameRecord {
    pub frame: u32,
    pub t: f32,
    pub time_us: u64,
    pub cam_pos: [f32; 3],
    pub cam_dir: [f32; 3],
    pub cam_vup: [f32; 3],
}
