#![no_std]

use prim::Ray;
use prim::Vec3;
use shared::ShaderConstants;
use spirv_std::glam::Vec4;
use spirv_std::glam::vec2;
use spirv_std::glam::vec4;
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use spirv_std::spirv;
use world::VoxelHit;

const OCTREE_DEPTH: u32 = 6;

#[spirv(vertex)]
pub fn main_vs(#[spirv(vertex_index)] vert_id: i32, #[spirv(position)] out_pos: &mut Vec4) {
    let uv = vec2(((vert_id << 1) & 2) as f32, (vert_id & 2) as f32);
    let pos = uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    *out_pos = vec4(pos.x, pos.y, 0.0, 1.0);
}

fn generate_ray(constants: &ShaderConstants, frag_coord: Vec4) -> Ray {
    let cam_pos = Vec3::new(
        constants.cam_pos[0],
        constants.cam_pos[1],
        constants.cam_pos[2],
    );
    let cam_dir = Vec3::new(
        constants.cam_dir[0],
        constants.cam_dir[1],
        constants.cam_dir[2],
    )
    .normalize();
    let cam_vup = Vec3::new(
        constants.cam_vup[0],
        constants.cam_vup[1],
        constants.cam_vup[2],
    );

    let aspect = constants.width as f32 / constants.height as f32;
    let fov = 70.0 * core::f32::consts::PI / 180.0;
    let half_h = (fov / 2.0).tan();
    let half_w = half_h * aspect;

    let w = (-cam_dir).normalize();
    let u = w.cross(cam_vup).normalize();
    let v = u.cross(w);
    let u = -u;

    let px = (2.0 * frag_coord.x / constants.width as f32 - 1.0) * half_w;
    let py = (1.0 - 2.0 * frag_coord.y / constants.height as f32) * half_h;

    let dir = (px * u + py * v - w).normalize();
    Ray::new(cam_pos, dir, 0.0)
}

fn material_color(value: u32) -> Vec3 {
    match value {
        1 => Vec3::new(0.3, 0.7, 0.2),    // grass green
        2 => Vec3::new(0.55, 0.35, 0.15), // wood brown
        3 => Vec3::new(0.6, 0.6, 0.6),    // stone gray
        4 => Vec3::new(0.8, 0.2, 0.2),    // red
        _ => Vec3::new(1.0, 0.0, 1.0),    // magenta = unknown
    }
}

#[spirv(fragment)]
pub fn main_fs(
    #[spirv(frag_coord)] frag_coord: Vec4,
    #[spirv(push_constant)] constants: &ShaderConstants,
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] octree_data: &[u32],
    output: &mut Vec4,
) {
    let ray = generate_ray(constants, frag_coord);

    let mut hit = VoxelHit::default();
    if world::trace_octree(octree_data, OCTREE_DEPTH, &ray, 0.001, 1000.0, &mut hit) {
        // Simple directional lighting
        let sun_dir = Vec3::new(0.4, 0.8, 0.3).normalize();
        let ndotl = hit.normal.dot(sun_dir).max(0.0);

        let base_color = material_color(hit.value);
        let ambient = 0.25;
        let color = base_color * (ambient + (1.0 - ambient) * ndotl);

        *output = vec4(color.x, color.y, color.z, 1.0);
    } else {
        // Sky gradient
        let v = frag_coord.y / constants.height as f32;
        let t = 1.0 - v;
        let sky = Vec3::new(1.0 - 0.5 * t, 1.0 - 0.3 * t, 1.0);
        *output = vec4(sky.x, sky.y, sky.z, 1.0);
    }
}
