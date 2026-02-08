#![cfg_attr(target_arch = "spirv", no_std)]

use bytemuck::Pod;
use bytemuck::Zeroable;

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
}
