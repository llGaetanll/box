#![cfg_attr(target_arch = "spirv", no_std)]

use gpu_prim::Ray;
use gpu_prim::Vec3;
use gpu_prim::rand::RandState;
use gpu_prim::traits::Vec3Ext;
use gpu_wire::HISTORY_MOVED;
use gpu_wire::HISTORY_NONE;
use gpu_wire::PIXEL_WORDS;
use gpu_wire::ShaderConstants;
use spirv_std::glam::Vec4;
use spirv_std::glam::vec2;
use spirv_std::glam::vec4;
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use spirv_std::spirv;
use voxel_octree::VoxelHit;

#[spirv(vertex)]
pub fn main_vs(#[spirv(vertex_index)] vert_id: i32, #[spirv(position)] out_pos: &mut Vec4) {
    let uv = vec2(((vert_id << 1) & 2) as f32, (vert_id & 2) as f32);
    let pos = uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    *out_pos = vec4(pos.x, pos.y, 0.0, 1.0);
}

/// Vertical field of view, in degrees.
const FOV_DEGREES: f32 = 70.0;

/// A pinhole camera: where it is, which way its image plane's axes point, and
/// how wide the plane is at unit distance. Built the same way for the current
/// and the previous frame's camera, so a point can be projected into the
/// previous frame to find where it was drawn.
#[derive(Clone, Copy)]
pub struct Camera {
    pub pos: Vec3,
    /// Image plane right, in world space.
    u: Vec3,
    /// Image plane up, in world space.
    v: Vec3,
    /// Backwards: the camera looks along `-w`.
    w: Vec3,
    half_w: f32,
    half_h: f32,
    width: f32,
    height: f32,
}

impl Camera {
    pub fn new(pos: [f32; 3], dir: [f32; 3], vup: [f32; 3], width: u32, height: u32) -> Self {
        let pos = Vec3::new(pos[0], pos[1], pos[2]);
        let dir = Vec3::new(dir[0], dir[1], dir[2]).normalize();
        let vup = Vec3::new(vup[0], vup[1], vup[2]);

        let aspect = width as f32 / height as f32;
        let fov = FOV_DEGREES * core::f32::consts::PI / 180.0;
        let half_h = (fov / 2.0).tan();
        let half_w = half_h * aspect;

        let w = -dir;
        let u = w.cross(vup).normalize();
        let v = u.cross(w);

        Self {
            pos,
            u: -u,
            v,
            w,
            half_w,
            half_h,
            width: width as f32,
            height: height as f32,
        }
    }

    pub fn current(constants: &ShaderConstants) -> Self {
        Self::new(
            constants.cam_pos,
            constants.cam_dir,
            constants.cam_vup,
            constants.width,
            constants.height,
        )
    }

    pub fn previous(constants: &ShaderConstants) -> Self {
        Self::new(
            constants.prev_cam_pos,
            constants.prev_cam_dir,
            constants.prev_cam_vup,
            constants.width,
            constants.height,
        )
    }

    /// The ray through continuous pixel coordinate `(sx, sy)`, where pixel
    /// `(i, j)` covers `[i, i + 1) x [j, j + 1)`.
    pub fn ray(&self, sx: f32, sy: f32) -> Ray {
        let px = (2.0 * sx / self.width - 1.0) * self.half_w;
        let py = (1.0 - 2.0 * sy / self.height) * self.half_h;
        let dir = (px * self.u + py * self.v - self.w).normalize();
        Ray::new(self.pos, dir, 0.0)
    }

    /// Where a direction from the camera lands on the image, as continuous
    /// pixel coordinates, which may lie outside the image. The flag is false
    /// when it points behind the camera.
    pub fn project_dir(&self, d: Vec3) -> (bool, f32, f32) {
        let depth = -d.dot(self.w);
        if depth <= 1e-6 {
            return (false, 0.0, 0.0);
        }
        let px = d.dot(self.u) / depth;
        let py = d.dot(self.v) / depth;
        let sx = (px / self.half_w + 1.0) * 0.5 * self.width;
        let sy = (1.0 - py / self.half_h) * 0.5 * self.height;
        (true, sx, sy)
    }

    /// Where a world point lands on the image. See `project_dir`.
    pub fn project(&self, p: Vec3) -> (bool, f32, f32) {
        self.project_dir(p - self.pos)
    }
}

/// Sub-pixel offset for a frame, in `[0, 1)^2`. The R2 low-discrepancy
/// sequence, so the offsets of successive frames spread evenly over the pixel.
pub fn jitter(frame: u32) -> (f32, f32) {
    let frame = frame as f32;
    let jx_raw = 0.7548776662 * frame;
    let jy_raw = 0.5698402910 * frame;
    (jx_raw - jx_raw.floor(), jy_raw - jy_raw.floor())
}

fn gen_state(frag_coord: Vec4, frame: u32) -> RandState {
    let x = frag_coord.x as u32;
    let y = frag_coord.y as u32;
    let state = x
        .wrapping_mul(747796405)
        .wrapping_add(y)
        .wrapping_add(frame.wrapping_mul(2654435761));
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

/// What a path's camera ray landed on, which is what decides whether another
/// frame's sample of the same pixel is looking at the same thing.
#[derive(Clone, Copy, Default)]
pub struct FirstHit {
    pub hit: bool,
    /// The hit point, or the ray direction when the ray reached the sky.
    pub pos: Vec3,
    pub normal: Vec3,
    pub value: u32,
}

/// Trace a ray through the scene, returning the final color and recording
/// what the camera ray hit in `first`.
pub fn trace_path(
    node_data: &[u32],
    voxel_data: &[u32],
    tree_depth: u32,
    tree_root: u32,
    ray: &Ray,
    state: &mut gpu_prim::RandState,
    first: &mut FirstHit,
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
        if !voxel_octree::trace_tree64(
            node_data, voxel_data, tree_depth, tree_root, &ray, 0.001, 1000.0, &mut hit,
        ) {
            if bounce == 0 {
                first.hit = false;
                first.pos = ray.dir();
            }
            accumulated += throughput * sky_color(ray.dir());
            break;
        }

        let hit_point = ray.orig() + hit.t * ray.dir();
        if bounce == 0 {
            first.hit = true;
            first.pos = hit_point;
            first.normal = hit.normal;
            first.value = hit.value;
        }

        let attenuation = material_color(hit.value);
        throughput *= attenuation;

        // Russian roulette: after bounce 3, randomly terminate dim paths
        if bounce >= 3 {
            let luminance = 0.2126 * throughput.x + 0.7152 * throughput.y + 0.0722 * throughput.z;
            let survive = luminance.clamp(0.05, 0.95);
            if gpu_prim::rand::rand_f(state) >= survive {
                break;
            }
            throughput /= survive;
        }

        // Cosine-weighted hemisphere sampling (Malley's method)
        let scatter_dir = Vec3::rand_cosine_hemisphere(state, hit.normal);

        ray = Ray::new(hit_point, scatter_dir, 0.0);
        bounce += 1;
    }

    accumulated
}

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
    state: &mut gpu_prim::RandState,
) -> Vec3 {
    let mut first = FirstHit::default();
    trace_path(
        node_data, voxel_data, tree_depth, tree_root, ray, state, &mut first,
    )
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

/// Most samples a pixel keeps while the camera moves. Every frame's
/// reprojection resamples the history between pixels, which blurs it a
/// little, so the history is kept short enough that the blur never builds up.
/// A still camera has no such limit.
const MAX_MOVING_SAMPLES: f32 = 32.0;

/// How far off the current surface's plane a history sample may lie and still
/// count as the same surface, in voxels. Voxel faces sit on integer planes,
/// so anything beyond rounding is a different face.
const PLANE_TOLERANCE: f32 = 0.01;

/// Pack what a pixel's camera ray landed on into one word: bit 31 for a hit,
/// the face in bits 0..3 and the voxel value in bits 8..16.
fn pack_surface(first: &FirstHit) -> u32 {
    if !first.hit {
        return 0;
    }
    let n = first.normal;
    let face = if n.x != 0.0 {
        if n.x > 0.0 { 0 } else { 1 }
    } else if n.y != 0.0 {
        if n.y > 0.0 { 2 } else { 3 }
    } else if n.z > 0.0 {
        4
    } else {
        5
    };
    (1 << 31) | ((first.value & 0xFF) << 8) | face
}

/// Read one history sample from the previous frame's buffer at pixel `(x, y)`
/// and say how much of it this pixel can use: its bilinear weight if it saw
/// the same surface, nothing otherwise.
fn history_tap(
    history: &[u32],
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    weight: f32,
    first: &FirstHit,
    surface: u32,
) -> (f32, Vec3, f32) {
    if weight <= 0.0 || x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return (0.0, Vec3::ZERO, 0.0);
    }
    let idx = (y as u32 * width + x as u32) as usize * PIXEL_WORDS as usize;
    let packed = history[idx + 7];
    if packed != surface {
        return (0.0, Vec3::ZERO, 0.0);
    }
    let pos = Vec3::new(
        f32::from_bits(history[idx + 4]),
        f32::from_bits(history[idx + 5]),
        f32::from_bits(history[idx + 6]),
    );
    if first.hit {
        // Same face, but is it the same plane? Sample positions along the
        // plane differ from pixel to pixel, so only the offset off it counts
        let off = (pos - first.pos).dot(first.normal);
        if off.abs() > PLANE_TOLERANCE {
            return (0.0, Vec3::ZERO, 0.0);
        }
    }
    let color = Vec3::new(
        f32::from_bits(history[idx]),
        f32::from_bits(history[idx + 1]),
        f32::from_bits(history[idx + 2]),
    );
    let count = f32::from_bits(history[idx + 3]);
    (weight, color * weight, count * weight)
}

/// The history this pixel can build on: its average color and how many
/// samples went into it. Zero samples when there is none.
fn fetch_history(
    constants: &ShaderConstants,
    history: &[u32],
    px: u32,
    py: u32,
    first: &FirstHit,
    surface: u32,
) -> (Vec3, f32) {
    if constants.history == HISTORY_NONE {
        return (Vec3::ZERO, 0.0);
    }

    let width = constants.width;
    let height = constants.height;

    if constants.history != HISTORY_MOVED {
        // Same camera, so this pixel's history is its own: every sample in it
        // went through this pixel, whatever surface each landed on, and all of
        // them belong in its average. Read as is, so a still image converges
        // to the exact pixel integral, edges included
        let idx = (py * width + px) as usize * PIXEL_WORDS as usize;
        let color = Vec3::new(
            f32::from_bits(history[idx]),
            f32::from_bits(history[idx + 1]),
            f32::from_bits(history[idx + 2]),
        );
        return (color, f32::from_bits(history[idx + 3]));
    }

    // Where the previous frame drew what this pixel sees now
    let prev = Camera::previous(constants);
    let (visible, sx, sy) = if first.hit {
        prev.project(first.pos)
    } else {
        prev.project_dir(first.pos)
    };
    if !visible {
        return (Vec3::ZERO, 0.0);
    }

    // Bilinear between the four pixels around it, each one only counting if
    // it saw the same surface
    let fx = sx - 0.5;
    let fy = sy - 0.5;
    let x0 = fx.floor();
    let y0 = fy.floor();
    let tx = fx - x0;
    let ty = fy - y0;
    let x0 = x0 as i32;
    let y0 = y0 as i32;

    let a = history_tap(
        history,
        width,
        height,
        x0,
        y0,
        (1.0 - tx) * (1.0 - ty),
        first,
        surface,
    );
    let b = history_tap(
        history,
        width,
        height,
        x0 + 1,
        y0,
        tx * (1.0 - ty),
        first,
        surface,
    );
    let c = history_tap(
        history,
        width,
        height,
        x0,
        y0 + 1,
        (1.0 - tx) * ty,
        first,
        surface,
    );
    let d = history_tap(
        history,
        width,
        height,
        x0 + 1,
        y0 + 1,
        tx * ty,
        first,
        surface,
    );

    let weight = a.0 + b.0 + c.0 + d.0;
    if weight <= 0.0 {
        return (Vec3::ZERO, 0.0);
    }
    let color = (a.1 + b.1 + c.1 + d.1) / weight;
    let count = (a.2 + b.2 + c.2 + d.2) / weight;
    (color, count.min(MAX_MOVING_SAMPLES))
}

#[spirv(fragment)]
pub fn main_fs(
    #[spirv(frag_coord)] frag_coord: Vec4,
    #[spirv(push_constant)] constants: &ShaderConstants,
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] node_data: &[u32],
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] voxel_data: &[u32],
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] history: &[u32],
    #[spirv(descriptor_set = 0, binding = 3, storage_buffer)] accum: &mut [u32],
    output: &mut Vec4,
) {
    let mut state = gen_state(frag_coord, constants.frame_count);
    let cam = Camera::current(constants);
    let (jx, jy) = jitter(constants.frame_count);

    // The first sample's camera ray describes the pixel's surface; later
    // samples through the same pixel almost always land on the same one
    let mut first = FirstHit::default();
    let mut sum = Vec3::ZERO;
    let samples = constants.samples.max(1);
    let mut s = 0u32;
    while s < samples {
        let ray = if s == 0 {
            cam.ray(frag_coord.x + jx - 0.5, frag_coord.y + jy - 0.5)
        } else {
            let ox = gpu_prim::rand::rand_f(&mut state);
            let oy = gpu_prim::rand::rand_f(&mut state);
            cam.ray(frag_coord.x + ox - 0.5, frag_coord.y + oy - 0.5)
        };
        let mut this = FirstHit::default();
        sum += trace_path(
            node_data,
            voxel_data,
            constants.tree_depth,
            constants.tree_root,
            &ray,
            &mut state,
            &mut this,
        );
        if s == 0 {
            first = this;
        }
        s += 1;
    }

    let px = frag_coord.x as u32;
    let py = frag_coord.y as u32;
    let surface = pack_surface(&first);

    // Blend into the history in linear space. The history's average is
    // weighted by its sample count, so each sample ever taken counts once
    let (prev_color, prev_count) = fetch_history(constants, history, px, py, &first, surface);
    let count = prev_count + samples as f32;
    let color = (prev_color * prev_count + sum) / count;

    let idx = (py * constants.width + px) as usize * PIXEL_WORDS as usize;
    accum[idx] = color.x.to_bits();
    accum[idx + 1] = color.y.to_bits();
    accum[idx + 2] = color.z.to_bits();
    accum[idx + 3] = count.to_bits();
    accum[idx + 4] = first.pos.x.to_bits();
    accum[idx + 5] = first.pos.y.to_bits();
    accum[idx + 6] = first.pos.z.to_bits();
    accum[idx + 7] = surface;

    // Tonemap and gamma correct for display
    let display = tonemap_aces(color);
    let display = Vec3::new(
        display.x.powf(1.0 / 2.2),
        display.y.powf(1.0 / 2.2),
        display.z.powf(1.0 / 2.2),
    );
    *output = vec4(display.x, display.y, display.z, 1.0);
}
