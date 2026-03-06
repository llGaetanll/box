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

    // R2 low-discrepancy subpixel jitter for anti-aliasing during accumulation
    let frame = constants.frame_count as f32;
    let jx_raw = 0.7548776662 * frame;
    let jy_raw = 0.5698402910 * frame;
    let jx = jx_raw - jx_raw.floor();
    let jy = jy_raw - jy_raw.floor();
    let sx = frag_coord.x + jx - 0.5;
    let sy = frag_coord.y + jy - 0.5;

    let px = (2.0 * sx / constants.width as f32 - 1.0) * half_w;
    let py = (1.0 - 2.0 * sy / constants.height as f32) * half_h;

    let dir = (px * u + py * v - w).normalize();
    Ray::new(cam_pos, dir, 0.0)
}

fn gen_state(frag_coord: Vec4, frame: u32) -> RandState {
    let x = frag_coord.x as u32;
    let y = frag_coord.y as u32;
    let state = x.wrapping_mul(747796405).wrapping_add(y).wrapping_add(frame.wrapping_mul(2654435761));
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}

pub fn sky_color(dir: Vec3) -> Vec3 {
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

const MAX_BOUNCES: u32 = 64;

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

        // Russian roulette: after bounce 3, randomly terminate dim paths
        if bounce >= 3 {
            let luminance = 0.2126 * throughput.x + 0.7152 * throughput.y + 0.0722 * throughput.z;
            let survive = luminance.clamp(0.05, 0.95);
            if prim::rand::rand_f(state) >= survive {
                break;
            }
            throughput /= survive;
        }

        // Cosine-weighted hemisphere sampling (Malley's method)
        let scatter_dir = Vec3::rand_cosine_hemisphere(state, hit.normal);

        let hit_point = ray.orig() + hit.t * ray.dir();
        ray = Ray::new(hit_point, scatter_dir, 0.0);
        bounce += 1;
    }

    accumulated
}

fn tonemap_aces(c: Vec3) -> Vec3 {
    let a = c * (c * 2.51 + Vec3::splat(0.03));
    let b = c * (c * 2.43 + Vec3::splat(0.59)) + Vec3::splat(0.14);
    Vec3::new(
        (a.x / b.x).clamp(0.0, 1.0),
        (a.y / b.y).clamp(0.0, 1.0),
        (a.z / b.z).clamp(0.0, 1.0),
    )
}

#[spirv(fragment)]
pub fn main_fs(
    #[spirv(frag_coord)] frag_coord: Vec4,
    #[spirv(push_constant)] constants: &ShaderConstants,
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] node_data: &[u32],
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] voxel_data: &[u32],
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] accum: &mut [u32],
    output: &mut Vec4,
) {
    let mut state = gen_state(frag_coord, constants.frame_count);
    let ray = generate_ray(constants, frag_coord);
    let color = trace_color(
        node_data,
        voxel_data,
        constants.tree_depth,
        constants.tree_root,
        &ray,
        &mut state,
    );

    // Temporal accumulation: blend with history in linear space
    let px = frag_coord.x as u32;
    let py = frag_coord.y as u32;
    let idx = (py * constants.width + px) as usize * 4;

    let color = if constants.frame_count == 0 {
        color
    } else {
        let prev = Vec3::new(
            f32::from_bits(accum[idx]),
            f32::from_bits(accum[idx + 1]),
            f32::from_bits(accum[idx + 2]),
        );
        let weight = 1.0 / (constants.frame_count as f32 + 1.0);
        prev + (color - prev) * weight
    };

    accum[idx] = color.x.to_bits();
    accum[idx + 1] = color.y.to_bits();
    accum[idx + 2] = color.z.to_bits();

    // Tonemap and gamma correct for display
    let display = tonemap_aces(color);
    let display = Vec3::new(
        display.x.powf(1.0 / 2.2),
        display.y.powf(1.0 / 2.2),
        display.z.powf(1.0 / 2.2),
    );
    *output = vec4(display.x, display.y, display.z, 1.0);
}
