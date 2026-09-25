#![cfg_attr(target_arch = "spirv", no_std)]

use bytemuck::Pod;
use bytemuck::Zeroable;

/// The history buffer holds nothing this frame can use: first frame, or the
/// window was just resized.
pub const HISTORY_NONE: u32 = 0;

/// The camera has not moved since the history was written, so each pixel's
/// history is its own and can be summed without limit.
pub const HISTORY_STILL: u32 = 1;

/// The camera moved. Each pixel looks up where its surface was in the
/// previous frame and blends into that, up to a cap so reprojection error
/// fades rather than accumulates.
pub const HISTORY_MOVED: u32 = 2;

/// Words per pixel in the history and accumulation buffers: color, sample
/// count, first hit position, and the packed surface description.
pub const PIXEL_WORDS: u32 = 8;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ShaderConstants {
    pub width: u32,
    pub height: u32,
    pub time: f32,
    pub cursor_x: f32,
    pub cursor_y: f32,
    pub cam_pos: [f32; 3],
    pub cam_dir: [f32; 3],
    pub cam_vup: [f32; 3],
    pub tree_depth: u32,
    pub tree_root: u32,
    /// Counts up every frame and never resets. It varies the sub-pixel jitter
    /// and the random sequence from frame to frame, which is what makes
    /// summing frames converge rather than repeat.
    pub frame_count: u32,
    /// The camera the history buffer was rendered from.
    pub prev_cam_pos: [f32; 3],
    pub prev_cam_dir: [f32; 3],
    pub prev_cam_vup: [f32; 3],
    /// One of the `HISTORY_*` values.
    pub history: u32,
    /// Paths traced per pixel this frame. At least 1.
    pub samples: u32,
}
