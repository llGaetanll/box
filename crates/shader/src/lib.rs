#![cfg_attr(target_arch = "spirv", no_std)]

use prim::Ray;
use prim::Vec3;
use prim::rand::RandState;
use prim::traits::Vec3Ext;
use shared::ShaderConstants;
use spirv_std::glam::Vec4;
use spirv_std::glam::vec2;
use spirv_std::glam::vec4;
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use spirv_std::spirv;
use world::VoxelHit;

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

fn gen_state(frag_coord: Vec4) -> RandState {
    let x = frag_coord.x as u32;
    let y = frag_coord.y as u32;
    let state = x.wrapping_mul(747796405).wrapping_add(y);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}

fn sky_color(dir: Vec3) -> Vec3 {
    let t = 0.5 * (dir.y + 1.0);
    Vec3::new(1.0 - 0.5 * t, 1.0 - 0.3 * t, 1.0)
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

const MAX_BOUNCES: u32 = 8;

/// Trace a ray through the scene, returning the final color.
///
/// This is the core path tracing loop, factored out of main_fs so it can be
/// tested on the CPU.
pub fn trace_color(
    node_data: &[u32],
    voxel_data: &[u32],
    tree_depth: u32,
    tree_root: u32,
    ray: &Ray,
    state: &mut prim::RandState,
) -> Vec3 {
    let mut ray = *ray;
    let mut throughput = Vec3::new(1.0, 1.0, 1.0);
    let mut accumulated = Vec3::new(0.0, 0.0, 0.0);

    let mut bounce = 0u32;
    loop {
        if bounce >= MAX_BOUNCES {
            break;
        }

        let mut hit = VoxelHit::default();
        if !world::trace_tree64(
            node_data,
            voxel_data,
            tree_depth,
            tree_root,
            &ray,
            0.001,
            1000.0,
            &mut hit,
        ) {
            accumulated += throughput * sky_color(ray.dir());
            break;
        }

        let attenuation = material_color(hit.value);
        throughput *= attenuation;

        // Lambertian scatter: new direction = normal + random unit vector
        let mut scatter_dir = hit.normal + Vec3::rand_unit(state);
        if scatter_dir.near_zero() {
            scatter_dir = hit.normal;
        }

        let hit_point = ray.orig() + hit.t * ray.dir();
        ray = Ray::new(hit_point, scatter_dir.normalize(), 0.0);
        bounce += 1;
    }

    accumulated
}

#[spirv(fragment)]
pub fn main_fs(
    #[spirv(frag_coord)] frag_coord: Vec4,
    #[spirv(push_constant)] constants: &ShaderConstants,
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] node_data: &[u32],
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] voxel_data: &[u32],
    output: &mut Vec4,
) {
    let mut state = gen_state(frag_coord);
    let ray = generate_ray(constants, frag_coord);
    let color = trace_color(
        node_data,
        voxel_data,
        constants.tree_depth,
        constants.tree_root,
        &ray,
        &mut state,
    );
    *output = vec4(color.x, color.y, color.z, 1.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tree64 with a single voxel of the given value at position (x,y,z)
    /// within a 4x4x4 grid. Returns (nodes_u32, data_u32, num_levels, root_index).
    fn single_voxel_tree(x: u32, y: u32, z: u32, value: u8) -> (Vec<u32>, Vec<u32>, u32, u32) {
        let mut flat = [0u8; 64];
        flat[(x + y * 4 + z * 16) as usize] = value;
        let tree = tree64::Tree64::new((&flat[..], [4, 4, 4]));
        let root = tree.root_state();
        let nodes_u32: Vec<u32> = bytemuck::cast_slice(&tree.nodes).to_vec();
        let data_u32: Vec<u32> = tree
            .data
            .chunks(4)
            .map(|chunk| {
                let mut word = 0u32;
                for (i, &b) in chunk.iter().enumerate() {
                    word |= (b as u32) << (i * 8);
                }
                word
            })
            .collect();
        (nodes_u32, data_u32, root.num_levels as u32, root.index)
    }

    #[test]
    fn ray_hits_single_voxel_and_sees_sky() {
        // Place a stone voxel (value=3) at (1,1,1) in a 4^3 grid.
        // Shoot a ray from outside, straight at it along +X.
        // The ray should hit the voxel, scatter, and (with this seed)
        // eventually escape to the sky, producing a nonzero color.
        let (nodes, data, depth, root) = single_voxel_tree(1, 1, 1, 3);
        let ray = Ray::new(Vec3::new(-5.0, 1.5, 1.5), Vec3::new(1.0, 0.0, 0.0), 0.0);
        let mut state: prim::RandState = 42;

        let color = trace_color(&nodes, &data, depth, root, &ray, &mut state);

        // Should have some color contribution (hit stone -> sky)
        assert!(color.x > 0.0 || color.y > 0.0 || color.z > 0.0,
            "expected nonzero color after hitting voxel, got {color:?}");
    }

    #[test]
    fn ray_misses_everything_returns_sky() {
        // Same scene but ray goes in -X direction, away from the voxel.
        let (nodes, data, depth, root) = single_voxel_tree(1, 1, 1, 3);
        let ray = Ray::new(Vec3::new(-5.0, 1.5, 1.5), Vec3::new(-1.0, 0.0, 0.0), 0.0);
        let mut state: prim::RandState = 42;

        let color = trace_color(&nodes, &data, depth, root, &ray, &mut state);
        let expected_sky = sky_color(ray.dir());

        assert!((color.x - expected_sky.x).abs() < 1e-6);
        assert!((color.y - expected_sky.y).abs() < 1e-6);
        assert!((color.z - expected_sky.z).abs() < 1e-6);
    }

    #[test]
    fn deterministic_with_same_seed() {
        let (nodes, data, depth, root) = single_voxel_tree(2, 0, 2, 1);
        let ray = Ray::new(Vec3::new(-5.0, 0.5, 2.5), Vec3::new(1.0, 0.0, 0.0), 0.0);

        let mut state1: prim::RandState = 123;
        let color1 = trace_color(&nodes, &data, depth, root, &ray, &mut state1);

        let mut state2: prim::RandState = 123;
        let color2 = trace_color(&nodes, &data, depth, root, &ray, &mut state2);

        assert_eq!(color1, color2, "same seed should produce identical results");
    }
}
