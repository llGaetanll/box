use gpu_prim::F;
use gpu_prim::Ray;
use gpu_prim::Vec3;

/// Bit 31 marks a leaf node.
pub const LEAF_BIT: u32 = 1 << 31;

/// Bits 22..0 hold the child pointer (index into the flat array).
pub const CHILD_PTR_MASK: u32 = 0x007F_FFFF;

/// Bits 30..23 hold the 8-bit child mask. Shift right by this to extract.
pub const CHILD_MASK_SHIFT: u32 = 23;

/// Maximum traversal depth supported.
const MAX_DEPTH: usize = 24;

/// Result of a ray-octree intersection.
#[derive(Clone, Copy)]
pub struct VoxelHit {
    /// Ray parameter at the hit point.
    pub t: F,
    /// Outward face normal of the voxel cube that was hit.
    pub normal: Vec3,
    /// Integer coordinates of the hit voxel.
    pub pos: [u32; 3],
    /// The stored voxel value.
    pub value: u32,
}

impl Default for VoxelHit {
    fn default() -> Self {
        Self {
            t: 0.0,
            normal: Vec3::ZERO,
            pos: [0; 3],
            value: 0,
        }
    }
}

/// Trace a ray through a flattened octree.
///
/// - `data`: the packed octree (see `Octree::flatten`)
/// - `depth`: the octree depth (world is `2^depth` on each axis)
/// - `ray`: the ray to trace
/// - `t_min` / `t_max`: the valid range of t for intersections
/// - `hit`: filled in on success
///
/// Returns `true` if a voxel was hit.
pub fn trace_octree(
    data: &[u32],
    depth: u32,
    ray: &Ray,
    t_min: F,
    t_max: F,
    hit: &mut VoxelHit,
) -> bool {
    if data.is_empty() {
        return false;
    }

    let size = (1u32 << depth) as F;

    // Intersect ray with the world AABB [0, size]^3
    let (t_enter, t_exit, _) = ray_aabb(ray, 0.0, 0.0, 0.0, size, size, size);

    let t_enter = t_enter.max(t_min);
    let t_exit = t_exit.min(t_max);

    if t_enter >= t_exit {
        return false;
    }

    // Iterative stack-based traversal
    struct StackEntry {
        node_idx: usize,
        // AABB min corner of this node
        x: F,
        y: F,
        z: F,
        // Half-size of this node
        half: F,
        // Which child octant to visit next (0..8, 8 = done)
        next_child: u32,
    }

    let mut stack = [const {
        StackEntry {
            node_idx: 0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            half: 0.0,
            next_child: 0,
        }
    }; MAX_DEPTH];
    // Push root
    stack[0] = StackEntry {
        node_idx: 0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
        half: size / 2.0,
        next_child: 0,
    };
    let mut sp: usize = 1;

    let mut best_t = t_max;
    let mut found = false;

    while sp > 0 {
        let top = sp - 1;
        let entry = &mut stack[top];

        if entry.next_child >= 8 {
            // Done with this node
            sp -= 1;
            continue;
        }

        let node_word = data[entry.node_idx];

        // If this is a leaf, we already handled it when pushing — shouldn't
        // reach here. But guard just in case.
        if node_word & LEAF_BIT != 0 {
            sp -= 1;
            continue;
        }

        let child_mask = (node_word >> CHILD_MASK_SHIFT) & 0xFF;
        let first_child = (node_word & CHILD_PTR_MASK) as usize;

        // Advance to next child octant
        let oct = entry.next_child;
        entry.next_child += 1;

        // Check if this octant has a child
        if child_mask & (1u32 << oct) == 0 {
            continue;
        }

        // Compute child AABB
        let half = entry.half;
        let cx = entry.x + if oct & 1 != 0 { half } else { 0.0 };
        let cy = entry.y + if oct & 2 != 0 { half } else { 0.0 };
        let cz = entry.z + if oct & 4 != 0 { half } else { 0.0 };

        // Intersect ray with child AABB
        let (ct_enter, ct_exit, face_axis) =
            ray_aabb(ray, cx, cy, cz, cx + half, cy + half, cz + half);

        let ct_enter = ct_enter.max(t_min);
        let ct_exit = ct_exit.min(best_t);

        if ct_enter >= ct_exit {
            continue;
        }

        // Find the child's index in the contiguous children block
        let child_slot = (child_mask & ((1u32 << oct) - 1)).count_ones() as usize;
        let child_idx = first_child + child_slot;

        if child_idx >= data.len() {
            continue;
        }

        let child_word = data[child_idx];

        if child_word & LEAF_BIT != 0 {
            // Hit a leaf voxel
            let value = child_word & !LEAF_BIT;
            if ct_enter < best_t {
                best_t = ct_enter;
                hit.t = ct_enter;
                hit.value = value;
                hit.normal = face_normal(ray, face_axis);

                // Compute integer voxel position from the AABB
                // If half <= 1.0, this is a single voxel
                hit.pos = [cx as u32, cy as u32, cz as u32];
                found = true;
            }
        } else {
            // Internal node — push onto stack for deeper traversal
            if sp < MAX_DEPTH {
                stack[sp] = StackEntry {
                    node_idx: child_idx,
                    x: cx,
                    y: cy,
                    z: cz,
                    half: half / 2.0,
                    next_child: 0,
                };
                sp += 1;
            }
        }
    }

    found
}

/// Ray-AABB intersection. Returns (t_enter, t_exit, entry_face_axis).
/// `entry_face_axis` encodes which slab the ray entered through:
/// 0 = X, 1 = Y, 2 = Z.
#[inline]
fn ray_aabb(ray: &Ray, x0: F, y0: F, z0: F, x1: F, y1: F, z1: F) -> (F, F, u32) {
    let orig = ray.orig();
    let dir = ray.dir();

    let inv_dx = 1.0 / dir.x;
    let inv_dy = 1.0 / dir.y;
    let inv_dz = 1.0 / dir.z;

    let (tx0, tx1) = if inv_dx >= 0.0 {
        ((x0 - orig.x) * inv_dx, (x1 - orig.x) * inv_dx)
    } else {
        ((x1 - orig.x) * inv_dx, (x0 - orig.x) * inv_dx)
    };

    let (ty0, ty1) = if inv_dy >= 0.0 {
        ((y0 - orig.y) * inv_dy, (y1 - orig.y) * inv_dy)
    } else {
        ((y1 - orig.y) * inv_dy, (y0 - orig.y) * inv_dy)
    };

    let (tz0, tz1) = if inv_dz >= 0.0 {
        ((z0 - orig.z) * inv_dz, (z1 - orig.z) * inv_dz)
    } else {
        ((z1 - orig.z) * inv_dz, (z0 - orig.z) * inv_dz)
    };

    let mut t_enter = tx0;
    let mut face: u32 = 0;

    if ty0 > t_enter {
        t_enter = ty0;
        face = 1;
    }
    if tz0 > t_enter {
        t_enter = tz0;
        face = 2;
    }

    let t_exit = tx1.min(ty1).min(tz1);

    (t_enter, t_exit, face)
}

/// Compute the outward face normal given the entry axis.
#[inline]
fn face_normal(ray: &Ray, axis: u32) -> Vec3 {
    let dir = ray.dir();
    match axis {
        0 => {
            if dir.x > 0.0 {
                Vec3::new(-1.0, 0.0, 0.0)
            } else {
                Vec3::new(1.0, 0.0, 0.0)
            }
        }
        1 => {
            if dir.y > 0.0 {
                Vec3::new(0.0, -1.0, 0.0)
            } else {
                Vec3::new(0.0, 1.0, 0.0)
            }
        }
        _ => {
            if dir.z > 0.0 {
                Vec3::new(0.0, 0.0, -1.0)
            } else {
                Vec3::new(0.0, 0.0, 1.0)
            }
        }
    }
}

/// Read a tree64 node from a u32 slice. Each node is 3 u32s:
/// word0: bit 0 = is_leaf, bits 1..31 = child/data pointer
/// word1: low 32 bits of pop_mask
/// word2: high 32 bits of pop_mask
///
/// Returns (is_leaf, ptr, mask_lo, mask_hi).
#[inline]
fn read_node(nodes: &[u32], index: u32) -> (bool, u32, u32, u32) {
    let base = index as usize * 3;
    let word0 = nodes[base];
    let is_leaf = (word0 & 1) != 0;
    let ptr = word0 >> 1;
    (is_leaf, ptr, nodes[base + 1], nodes[base + 2])
}

/// Read a u8 value from a u32 data buffer. The u8 data is packed in
/// little-endian order: 4 bytes per u32.
#[inline]
fn read_data_u8(data: &[u32], index: u32) -> u32 {
    let word_index = index as usize / 4;
    let byte_offset = (index as usize % 4) * 8;
    (data[word_index] >> byte_offset) & 0xFF
}

/// Count the number of set bits below `bit` in a 64-bit mask represented
/// as two u32 halves. This is the sparse child index.
#[inline]
fn popcount_below(mask_lo: u32, mask_hi: u32, bit: u32) -> u32 {
    if bit < 32 {
        // Only need bits from lo half, below position `bit`
        (mask_lo & ((1u32 << bit) - 1)).count_ones()
    } else if bit < 64 {
        // All bits of lo, plus bits below (bit - 32) in hi
        mask_lo.count_ones() + (mask_hi & ((1u32 << (bit - 32)) - 1)).count_ones()
    } else {
        mask_lo.count_ones() + mask_hi.count_ones()
    }
}

/// Check if bit `bit` is set in a 64-bit mask stored as two u32 halves.
#[inline]
fn mask_test(mask_lo: u32, mask_hi: u32, bit: u32) -> bool {
    if bit < 32 {
        mask_lo & (1u32 << bit) != 0
    } else {
        mask_hi & (1u32 << (bit - 32)) != 0
    }
}

/// Encode local (x, y, z) coordinates (each 0..3) into a bit position in the
/// 4×4×4 pop_mask. Inverse of `bit_to_xyz`.
#[inline]
fn xyz_to_bit(x: u32, y: u32, z: u32) -> u32 {
    x + y * 4 + z * 16
}

/// Most levels a tree64 can have and still be traversed. 4^8 voxels per axis.
const MAX_LEVELS: usize = 8;

/// Steps a single ray may take before it is declared a miss. A guard against
/// a degenerate ray looping, not a budget a real one gets close to.
const MAX_STEPS: u32 = 1024;

/// A direction component with a smaller magnitude than this is treated as
/// this, so its reciprocal stays finite and the boundary times stay ordered.
const MIN_DIR: F = 1e-8;

/// Trace a ray through a tree64 (64-ary voxel tree) using DDA traversal.
///
/// Each tree level is a 4×4×4 grid. The ray is stepped through each grid
/// in front-to-back order using Amanatides-Woo DDA, so the first occupied
/// voxel it reaches is the nearest one and traversal stops there.
///
/// The ray's position is tracked as the integer coordinate of the voxel it
/// is in. Because a level's cell size is a power of four, the cell the ray
/// occupies at any level is a two bit slice of that coordinate, and the node
/// containing it is the bits above that. Stepping across a cell boundary
/// changes the coordinate along one axis; XORing it with the previous value
/// says exactly which levels the ray has left, and the node for the level it
/// lands in is recovered from a per-level stack of node indices. Nothing else
/// is kept per level, so the traversal state fits in registers.
///
/// - `nodes`: the node array as `&[u32]` (bytemuck-cast from `&[Node]`, 3 u32s per node)
/// - `data`: the voxel data as `&[u32]` (packed u8 values, 4 per u32)
/// - `num_levels`: tree depth (e.g. 3 for 64^3 world)
/// - `root_index`: index of the root node in the node array
/// - `ray`: the ray to trace
/// - `t_min` / `t_max`: valid intersection range
/// - `hit`: filled in on success
///
/// Returns `true` if a voxel was hit.
pub fn trace_tree64(
    nodes: &[u32],
    data: &[u32],
    num_levels: u32,
    root_index: u32,
    ray: &Ray,
    t_min: F,
    t_max: F,
    hit: &mut VoxelHit,
) -> bool {
    if nodes.is_empty() || num_levels == 0 || num_levels as usize > MAX_LEVELS {
        return false;
    }

    // World is 4^num_levels voxels per axis, so 2 bits of coordinate per level
    let world_bits = 2 * num_levels;
    let world_max = (1i32 << world_bits) - 1;
    let world_size = (1u32 << world_bits) as F;

    let (t_enter, t_exit, enter_axis) =
        ray_aabb(ray, 0.0, 0.0, 0.0, world_size, world_size, world_size);
    let t_enter = t_enter.max(t_min);
    let t_exit = t_exit.min(t_max);
    if t_enter >= t_exit {
        return false;
    }

    let orig = ray.orig();
    let dir = ray.dir();

    // A zero component would make its boundary times NaN, which compares false
    // against everything and breaks the choice of axis to step along
    let dx = if dir.x.abs() < MIN_DIR {
        if dir.x < 0.0 { -MIN_DIR } else { MIN_DIR }
    } else {
        dir.x
    };
    let dy = if dir.y.abs() < MIN_DIR {
        if dir.y < 0.0 { -MIN_DIR } else { MIN_DIR }
    } else {
        dir.y
    };
    let dz = if dir.z.abs() < MIN_DIR {
        if dir.z < 0.0 { -MIN_DIR } else { MIN_DIR }
    } else {
        dir.z
    };
    let inv_dx = 1.0 / dx;
    let inv_dy = 1.0 / dy;
    let inv_dz = 1.0 / dz;
    let pos_x = dx >= 0.0;
    let pos_y = dy >= 0.0;
    let pos_z = dz >= 0.0;

    // Voxel the ray is in at t. On the world's face the coordinate may round
    // to the face itself, which the clamp folds back inside.
    let mut t = t_enter;
    let mut axis = enter_axis;
    let px = orig.x + t * dir.x;
    let py = orig.y + t * dir.y;
    let pz = orig.z + t * dir.z;
    let mut ix = (px as i32).clamp(0, world_max);
    let mut iy = (py as i32).clamp(0, world_max);
    let mut iz = (pz as i32).clamp(0, world_max);

    // stack[l] is the node at level l for every level above the current one
    let mut stack = [0u32; MAX_LEVELS];
    let mut level: u32 = 0;
    let mut node_index = root_index;
    let (mut is_leaf, mut ptr, mut mask_lo, mut mask_hi) = read_node(nodes, node_index);

    let mut steps = 0u32;
    while steps < MAX_STEPS {
        steps += 1;

        // Cell of the current node that the ray is in
        let shift = 2 * (num_levels - 1 - level);
        let cx = ((ix >> shift) & 3) as u32;
        let cy = ((iy >> shift) & 3) as u32;
        let cz = ((iz >> shift) & 3) as u32;
        let bit = xyz_to_bit(cx, cy, cz);

        let cell = 1i32 << shift;
        let cell_mask = !(cell - 1);
        let min_x = ix & cell_mask;
        let min_y = iy & cell_mask;
        let min_z = iz & cell_mask;

        if mask_test(mask_lo, mask_hi, bit) {
            let sparse_index = popcount_below(mask_lo, mask_hi, bit);

            if is_leaf {
                // The ray is at this voxel's entry face: t is the entry time and
                // the axis last stepped across is the face it came through
                hit.t = t;
                hit.value = read_data_u8(data, ptr + sparse_index);
                hit.normal = face_normal(ray, axis);
                hit.pos = [ix as u32, iy as u32, iz as u32];
                return true;
            }

            // Descend. Only the stepped axis has been kept current since this
            // level was entered, so refresh the low bits of the other two from
            // where the ray actually is, kept within the cell being entered
            let px = orig.x + t * dir.x;
            let py = orig.y + t * dir.y;
            let pz = orig.z + t * dir.z;
            ix = (px as i32).clamp(min_x, min_x + cell - 1);
            iy = (py as i32).clamp(min_y, min_y + cell - 1);
            iz = (pz as i32).clamp(min_z, min_z + cell - 1);

            stack[level as usize] = node_index;
            level += 1;
            node_index = ptr + sparse_index;
            let node = read_node(nodes, node_index);
            is_leaf = node.0;
            ptr = node.1;
            mask_lo = node.2;
            mask_hi = node.3;
            continue;
        }

        // Empty cell: step to the neighbouring cell the ray leaves through
        let bx = if pos_x { min_x + cell } else { min_x };
        let by = if pos_y { min_y + cell } else { min_y };
        let bz = if pos_z { min_z + cell } else { min_z };
        let tx = (bx as F - orig.x) * inv_dx;
        let ty = (by as F - orig.y) * inv_dy;
        let tz = (bz as F - orig.z) * inv_dz;

        let old;
        let new;
        if tx <= ty && tx <= tz {
            t = tx;
            axis = 0;
            old = ix;
            new = if pos_x { min_x + cell } else { min_x - 1 };
            ix = new;
        } else if ty <= tz {
            t = ty;
            axis = 1;
            old = iy;
            new = if pos_y { min_y + cell } else { min_y - 1 };
            iy = new;
        } else {
            t = tz;
            axis = 2;
            old = iz;
            new = if pos_z { min_z + cell } else { min_z - 1 };
            iz = new;
        }

        if t >= t_exit {
            return false;
        }

        // Which levels did that step leave? The node at level l covers the bits
        // from 2 * (num_levels - l) up, so the highest changed bit says how far
        // up the ray has gone. Past the root means out of the world.
        let changed = (old ^ new) as u32;
        let high_bit = 31 - changed.leading_zeros();
        let deepest = num_levels as i32 - 1 - (high_bit >> 1) as i32;
        if deepest < 0 {
            return false;
        }
        if (deepest as u32) < level {
            level = deepest as u32;
            node_index = stack[level as usize];
            let node = read_node(nodes, node_index);
            is_leaf = node.0;
            ptr = node.1;
            mask_lo = node.2;
            mask_hi = node.3;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use gpu_prim::Ray;

    use super::*;

    #[test]
    fn ray_aabb_hit() {
        let ray = Ray::new(Vec3::new(-1.0, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0), 0.0);
        let (t_enter, t_exit, axis) = ray_aabb(&ray, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
        assert!(t_enter < t_exit);
        assert!((t_enter - 1.0).abs() < 1e-6);
        assert!((t_exit - 2.0).abs() < 1e-6);
        assert_eq!(axis, 0); // entered through X face
    }

    #[test]
    fn ray_aabb_miss() {
        let ray = Ray::new(Vec3::new(-1.0, 5.0, 0.5), Vec3::new(1.0, 0.0, 0.0), 0.0);
        let (t_enter, t_exit, _) = ray_aabb(&ray, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
        assert!(t_enter >= t_exit);
    }
}

#[cfg(test)]
mod tree64_tests {
    use gpu_prim::Ray;
    use gpu_prim::Vec3;

    use super::*;

    const SIZE: u32 = 64;

    fn is_menger(mut x: u32, mut y: u32, mut z: u32, size: u32) -> bool {
        let mut s = size;
        while s > 1 {
            s /= 3;
            let center = u32::from((x / s) % 3 == 1)
                + u32::from((y / s) % 3 == 1)
                + u32::from((z / s) % 3 == 1);
            if center >= 2 {
                return false;
            }
            x %= s;
            y %= s;
            z %= s;
        }
        true
    }

    /// A Menger sponge plus a few stray voxels, as a dense grid and a tree64.
    fn scene() -> (Vec<u8>, Vec<u32>, Vec<u32>, u32, u32) {
        let mut flat = vec![0u8; (SIZE * SIZE * SIZE) as usize];
        let sponge = 27;
        let offset = (SIZE - sponge) / 2;
        for z in 0..sponge {
            for y in 0..sponge {
                for x in 0..sponge {
                    if is_menger(x, y, z, sponge) {
                        let i = (x + offset) + (y + offset) * SIZE + (z + offset) * SIZE * SIZE;
                        flat[i as usize] = 3;
                    }
                }
            }
        }
        for &(x, y, z, v) in &[
            (0, 0, 0, 1u8),
            (63, 63, 63, 2),
            (5, 60, 7, 4),
            (40, 2, 61, 1),
        ] {
            flat[(x + y * SIZE + z * SIZE * SIZE) as usize] = v;
        }

        let tree = voxel_tree64::Tree64::new((&flat[..], [SIZE; 3]));
        let root = tree.root_state();
        let nodes: Vec<u32> = bytemuck::cast_slice(&tree.nodes).to_vec();
        let data: Vec<u32> = tree
            .data
            .chunks(4)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |w, (i, &b)| w | ((b as u32) << (i * 8)))
            })
            .collect();
        (flat, nodes, data, root.num_levels as u32, root.index)
    }

    /// Plain Amanatides-Woo over the dense grid, one voxel at a time.
    fn reference(grid: &[u8], ray: &Ray, t_min: F, t_max: F) -> Option<(F, u32, Vec3, [u32; 3])> {
        let size = SIZE as F;
        let (t_enter, t_exit, mut axis) = ray_aabb(ray, 0.0, 0.0, 0.0, size, size, size);
        let t_enter = t_enter.max(t_min);
        let t_exit = t_exit.min(t_max);
        if t_enter >= t_exit {
            return None;
        }
        let o = ray.orig();
        let d = ray.dir();
        let p = o + t_enter * d;
        let max = SIZE as i32 - 1;
        let mut i = [
            (p.x.floor() as i32).clamp(0, max),
            (p.y.floor() as i32).clamp(0, max),
            (p.z.floor() as i32).clamp(0, max),
        ];
        let dd = [d.x, d.y, d.z];
        let oo = [o.x, o.y, o.z];
        let mut t = t_enter;
        loop {
            let v = grid[(i[0] + i[1] * SIZE as i32 + i[2] * SIZE as i32 * SIZE as i32) as usize];
            if v != 0 {
                return Some((
                    t,
                    v as u32,
                    face_normal(ray, axis),
                    [i[0] as u32, i[1] as u32, i[2] as u32],
                ));
            }
            let mut best = F::INFINITY;
            let mut best_axis = 0;
            for a in 0..3 {
                if dd[a] == 0.0 {
                    continue;
                }
                let b = if dd[a] > 0.0 { i[a] + 1 } else { i[a] };
                let ta = (b as F - oo[a]) / dd[a];
                if ta < best {
                    best = ta;
                    best_axis = a;
                }
            }
            t = best;
            axis = best_axis as u32;
            i[best_axis] += if dd[best_axis] > 0.0 { 1 } else { -1 };
            if t >= t_exit || i[best_axis] < 0 || i[best_axis] > max {
                return None;
            }
        }
    }

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> F {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as F) / (1u64 << 24) as F
        }
        fn range(&mut self, lo: F, hi: F) -> F {
            lo + self.f() * (hi - lo)
        }
    }

    #[test]
    fn matches_dense_grid_reference() {
        let (grid, nodes, data, levels, root) = scene();
        let mut rng = Lcg(7);
        let mut hits = 0;
        for k in 0..20_000 {
            // Half the rays start outside the world, half inside it
            let orig = if k % 2 == 0 {
                let u = Vec3::new(
                    rng.range(-1.0, 1.0),
                    rng.range(-1.0, 1.0),
                    rng.range(-1.0, 1.0),
                )
                .normalize_or_zero();
                Vec3::splat(32.0) + u * rng.range(60.0, 120.0)
            } else {
                Vec3::new(
                    rng.range(0.0, 64.0),
                    rng.range(0.0, 64.0),
                    rng.range(0.0, 64.0),
                )
            };
            let target = Vec3::new(
                rng.range(0.0, 64.0),
                rng.range(0.0, 64.0),
                rng.range(0.0, 64.0),
            );
            let mut dir = (target - orig).normalize_or_zero();
            if dir == Vec3::ZERO {
                continue;
            }
            // Some axis-aligned rays, which are the degenerate case for DDA
            if k % 7 == 0 {
                dir = match k % 3 {
                    0 => Vec3::new(dir.x.signum(), 0.0, 0.0),
                    1 => Vec3::new(0.0, dir.y.signum(), 0.0),
                    _ => Vec3::new(0.0, 0.0, dir.z.signum()),
                };
            }
            let ray = Ray::new(orig, dir, 0.0);

            let expected = reference(&grid, &ray, 0.001, 1000.0);
            let mut hit = VoxelHit::default();
            let found = trace_tree64(&nodes, &data, levels, root, &ray, 0.001, 1000.0, &mut hit);

            match expected {
                None => assert!(
                    !found,
                    "ray {k} {orig:?} {dir:?}: expected miss, hit {:?} t={}",
                    hit.pos, hit.t
                ),
                Some((t, value, normal, pos)) => {
                    hits += 1;
                    assert!(
                        found,
                        "ray {k} {orig:?} {dir:?}: expected hit at {pos:?} t={t}, got miss"
                    );
                    // A ray skimming a voxel edge may legitimately land in a
                    // neighbouring voxel of the same distance, so compare on t
                    // and value, and only demand the same voxel when t agrees
                    assert!((hit.t - t).abs() < 1e-3, "ray {k}: t {} vs {t}", hit.t);
                    assert_eq!(hit.value, value, "ray {k}: value");
                    if hit.pos != pos {
                        assert!(
                            (hit.t - t).abs() < 1e-4,
                            "ray {k}: pos {:?} vs {pos:?}",
                            hit.pos
                        );
                    } else {
                        assert_eq!(hit.normal, normal, "ray {k}: normal");
                    }
                }
            }
        }
        assert!(
            hits > 5000,
            "only {hits} rays hit; the test is not exercising much"
        );
    }
}
