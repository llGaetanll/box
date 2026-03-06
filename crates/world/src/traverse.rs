use prim::F;
use prim::Ray;
use prim::Vec3;

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

/// Trace a ray through a tree64 (64-ary voxel tree) using DDA traversal.
///
/// Each tree level is a 4×4×4 grid. The ray is stepped through each grid
/// in front-to-back order using Amanatides-Woo DDA. On first leaf hit the
/// traversal terminates immediately.
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
    if nodes.is_empty() {
        return false;
    }

    // World size: 4^num_levels per axis
    let mut world_size: F = 1.0;
    let mut i = 0u32;
    while i < num_levels {
        world_size *= 4.0;
        i += 1;
    }

    // Intersect ray with the world AABB [0, world_size]^3
    let (t_enter, t_exit, _) = ray_aabb(ray, 0.0, 0.0, 0.0, world_size, world_size, world_size);

    let t_enter = t_enter.max(t_min);
    let t_exit = t_exit.min(t_max);

    if t_enter >= t_exit {
        return false;
    }

    let (root_is_leaf, _, root_mask_lo, root_mask_hi) = read_node(nodes, root_index);
    if root_mask_lo == 0 && root_mask_hi == 0 {
        return false;
    }

    let orig = ray.orig();
    let dir = ray.dir();

    // Precompute inverse direction and step signs for DDA
    let inv_dir = Vec3::new(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
    let step_x: i32 = if dir.x >= 0.0 { 1 } else { -1 };
    let step_y: i32 = if dir.y >= 0.0 { 1 } else { -1 };
    let step_z: i32 = if dir.z >= 0.0 { 1 } else { -1 };

    // Stack for hierarchical DDA traversal
    struct StackEntry {
        node_index: u32,
        // Origin of this node's grid in world space
        ox: F,
        oy: F,
        oz: F,
        // Size of each child cell at this level
        cell_size: F,
        // Current DDA cell position (0..3 each)
        cx: i32,
        cy: i32,
        cz: i32,
        // DDA t values for next boundary crossing in each axis
        t_next_x: F,
        t_next_y: F,
        t_next_z: F,
        // DDA t increment per cell in each axis
        t_delta_x: F,
        t_delta_y: F,
        t_delta_z: F,
        // t value where ray exits this node
        t_exit: F,
    }

    let mut stack = [const {
        StackEntry {
            node_index: 0,
            ox: 0.0, oy: 0.0, oz: 0.0,
            cell_size: 0.0,
            cx: 0, cy: 0, cz: 0,
            t_next_x: 0.0, t_next_y: 0.0, t_next_z: 0.0,
            t_delta_x: 0.0, t_delta_y: 0.0, t_delta_z: 0.0,
            t_exit: 0.0,
        }
    }; MAX_DEPTH];

    // Initialize DDA state for a node: compute starting cell and t values
    // `t_cur` is the t at which the ray enters this node.
    #[inline]
    fn init_dda(
        orig: Vec3, inv_dir: Vec3, step_x: i32, step_y: i32, step_z: i32,
        node_ox: F, node_oy: F, node_oz: F,
        cell_size: F, t_cur: F, node_t_exit: F,
        entry: &mut StackEntry,
    ) {
        entry.cell_size = cell_size;
        entry.ox = node_ox;
        entry.oy = node_oy;
        entry.oz = node_oz;
        entry.t_exit = node_t_exit;

        // Position where ray enters this node (nudge slightly inside)
        let eps = cell_size * 1e-4;
        let p = orig + (t_cur + eps) * Vec3::new(1.0 / inv_dir.x, 1.0 / inv_dir.y, 1.0 / inv_dir.z);

        // Compute starting cell indices, clamped to [0, 3]
        let fx = ((p.x - node_ox) / cell_size).floor();
        let fy = ((p.y - node_oy) / cell_size).floor();
        let fz = ((p.z - node_oz) / cell_size).floor();
        entry.cx = (fx as i32).clamp(0, 3);
        entry.cy = (fy as i32).clamp(0, 3);
        entry.cz = (fz as i32).clamp(0, 3);

        // t values at next cell boundaries
        let bound_x = node_ox + (if step_x > 0 { entry.cx + 1 } else { entry.cx }) as F * cell_size;
        let bound_y = node_oy + (if step_y > 0 { entry.cy + 1 } else { entry.cy }) as F * cell_size;
        let bound_z = node_oz + (if step_z > 0 { entry.cz + 1 } else { entry.cz }) as F * cell_size;

        entry.t_next_x = (bound_x - orig.x) * inv_dir.x;
        entry.t_next_y = (bound_y - orig.y) * inv_dir.y;
        entry.t_next_z = (bound_z - orig.z) * inv_dir.z;

        // t increment per cell
        entry.t_delta_x = (cell_size * inv_dir.x).abs();
        entry.t_delta_y = (cell_size * inv_dir.y).abs();
        entry.t_delta_z = (cell_size * inv_dir.z).abs();
    }

    // Push root node
    let root_cell_size = if root_is_leaf { 1.0 } else { world_size / 4.0 };
    init_dda(
        orig, inv_dir, step_x, step_y, step_z,
        0.0, 0.0, 0.0,
        root_cell_size, t_enter, t_exit,
        &mut stack[0],
    );
    stack[0].node_index = root_index;
    let mut sp: usize = 1;

    while sp > 0 {
        let top = sp - 1;

        // Check if current cell is out of bounds (DDA stepped outside 4×4×4 grid)
        if stack[top].cx < 0 || stack[top].cx > 3
            || stack[top].cy < 0 || stack[top].cy > 3
            || stack[top].cz < 0 || stack[top].cz > 3
        {
            sp -= 1;
            continue;
        }

        let node_index = stack[top].node_index;
        let cx = stack[top].cx as u32;
        let cy = stack[top].cy as u32;
        let cz = stack[top].cz as u32;
        let cell_size = stack[top].cell_size;
        let ox = stack[top].ox;
        let oy = stack[top].oy;
        let oz = stack[top].oz;

        // Compute t_enter for the current cell from the DDA state
        // It's the max of t_next minus t_delta for each axis, but simpler
        // to just use the cell AABB entry. We use the minimum of t_next values
        // as t_exit for this cell.
        let t_next_x = stack[top].t_next_x;
        let t_next_y = stack[top].t_next_y;
        let t_next_z = stack[top].t_next_z;
        // Advance DDA to next cell (for the next iteration at this level)
        // Step along the axis with the smallest t_next
        if t_next_x <= t_next_y && t_next_x <= t_next_z {
            stack[top].cx += step_x;
            stack[top].t_next_x += stack[top].t_delta_x;
        } else if t_next_y <= t_next_z {
            stack[top].cy += step_y;
            stack[top].t_next_y += stack[top].t_delta_y;
        } else {
            stack[top].cz += step_z;
            stack[top].t_next_z += stack[top].t_delta_z;
        }

        let bit = xyz_to_bit(cx, cy, cz);
        let (is_leaf, ptr, mask_lo, mask_hi) = read_node(nodes, node_index);

        // Check if this cell is occupied
        if !mask_test(mask_lo, mask_hi, bit) {
            continue;
        }

        let sparse_index = popcount_below(mask_lo, mask_hi, bit);

        if is_leaf {
            // This node's children are data (voxels) — we found a hit
            let value = read_data_u8(data, ptr + sparse_index);
            let vox_x = ox + cx as F * cell_size;
            let vox_y = oy + cy as F * cell_size;
            let vox_z = oz + cz as F * cell_size;

            // Compute entry t and face for this specific voxel cell
            let (vt_enter, _, vface) = ray_aabb(
                ray,
                vox_x, vox_y, vox_z,
                vox_x + cell_size, vox_y + cell_size, vox_z + cell_size,
            );
            let vt_enter = vt_enter.max(t_min);

            hit.t = vt_enter;
            hit.value = value;
            hit.normal = face_normal(ray, vface);
            hit.pos = [vox_x as u32, vox_y as u32, vox_z as u32];
            return true;
        }

        // Internal child node — descend with a new DDA
        let child_node_index = ptr + sparse_index;
        let (_, _, c_mask_lo, c_mask_hi) = read_node(nodes, child_node_index);

        if (c_mask_lo == 0 && c_mask_hi == 0) || sp >= MAX_DEPTH {
            continue;
        }

        let child_ox = ox + cx as F * cell_size;
        let child_oy = oy + cy as F * cell_size;
        let child_oz = oz + cz as F * cell_size;

        // Compute t_enter for the child node
        let (child_t_enter, child_t_exit, _) = ray_aabb(
            ray,
            child_ox, child_oy, child_oz,
            child_ox + cell_size, child_oy + cell_size, child_oz + cell_size,
        );
        let child_t_enter = child_t_enter.max(t_min);
        let child_t_exit = child_t_exit.min(t_max);

        let (child_is_leaf, _, _, _) = read_node(nodes, child_node_index);
        let child_cell_size = if child_is_leaf { cell_size / 4.0 } else { cell_size / 4.0 };

        init_dda(
            orig, inv_dir, step_x, step_y, step_z,
            child_ox, child_oy, child_oz,
            child_cell_size, child_t_enter, child_t_exit,
            &mut stack[sp],
        );
        stack[sp].node_index = child_node_index;
        sp += 1;
    }

    false
}

#[cfg(test)]
mod tests {
    use prim::Ray;

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
