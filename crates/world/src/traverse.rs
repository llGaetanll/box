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

/// Decode a bit position (0..63) in the 4×4×4 pop_mask back to local (x, y, z)
/// coordinates. Encoding: index = x + y*4 + z*16.
#[inline]
fn bit_to_xyz(bit: u32) -> (u32, u32, u32) {
    (bit & 3, (bit >> 2) & 3, (bit >> 4) & 3)
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

/// Find the next set bit in a 64-bit mask (as two u32 halves) starting
/// from position `start` (inclusive). Returns 64 if no bit found.
#[inline]
fn next_set_bit(mask_lo: u32, mask_hi: u32, start: u32) -> u32 {
    if start < 32 {
        // Check remaining bits in lo half
        let masked_lo = mask_lo & !((1u32 << start) - 1);
        if masked_lo != 0 {
            return masked_lo.trailing_zeros();
        }
        // Check hi half
        if mask_hi != 0 {
            return 32 + mask_hi.trailing_zeros();
        }
    } else if start < 64 {
        let masked_hi = mask_hi & !((1u32 << (start - 32)) - 1);
        if masked_hi != 0 {
            return 32 + masked_hi.trailing_zeros();
        }
    }
    64
}

/// Trace a ray through a tree64 (64-ary voxel tree).
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
    let mut size: F = 1.0;
    let mut i = 0u32;
    while i < num_levels {
        size *= 4.0;
        i += 1;
    }

    // Intersect ray with the world AABB [0, size]^3
    let (t_enter, t_exit, _) = ray_aabb(ray, 0.0, 0.0, 0.0, size, size, size);

    let t_enter = t_enter.max(t_min);
    let t_exit = t_exit.min(t_max);

    if t_enter >= t_exit {
        return false;
    }

    struct StackEntry {
        node_index: u32,
        x: F,
        y: F,
        z: F,
        child_size: F,
        // Next child bit position to check (0..64, 64 = done)
        next_child: u32,
    }

    let mut stack = [const {
        StackEntry {
            node_index: 0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            child_size: 0.0,
            next_child: 0,
        }
    }; MAX_DEPTH];

    let (root_is_leaf, root_ptr, root_mask_lo, root_mask_hi) = read_node(nodes, root_index);

    if root_mask_lo == 0 && root_mask_hi == 0 {
        return false;
    }

    // If root is a leaf, we're at the bottom level already — handle inline
    if root_is_leaf {
        let child_size: F = 1.0;
        let mut best_t = t_max;
        let mut found = false;
        let mut bit = next_set_bit(root_mask_lo, root_mask_hi, 0);
        while bit < 64 {
            let (lx, ly, lz) = bit_to_xyz(bit);
            let cx = lx as F * child_size;
            let cy = ly as F * child_size;
            let cz = lz as F * child_size;
            let (ct_enter, ct_exit, face_axis) = ray_aabb(
                ray,
                cx,
                cy,
                cz,
                cx + child_size,
                cy + child_size,
                cz + child_size,
            );
            let ct_enter = ct_enter.max(t_min);
            let ct_exit = ct_exit.min(best_t);
            if ct_enter < ct_exit {
                let sparse_index = popcount_below(root_mask_lo, root_mask_hi, bit);
                let value = read_data_u8(data, root_ptr + sparse_index);
                best_t = ct_enter;
                hit.t = ct_enter;
                hit.value = value;
                hit.normal = face_normal(ray, face_axis);
                hit.pos = [cx as u32, cy as u32, cz as u32];
                found = true;
            }
            bit = next_set_bit(root_mask_lo, root_mask_hi, bit + 1);
        }
        return found;
    }

    let first_bit = next_set_bit(root_mask_lo, root_mask_hi, 0);
    stack[0] = StackEntry {
        node_index: root_index,
        x: 0.0,
        y: 0.0,
        z: 0.0,
        child_size: size / 4.0,
        next_child: first_bit,
    };
    let mut sp: usize = 1;

    let mut best_t = t_max;
    let mut found = false;

    while sp > 0 {
        let top = sp - 1;

        if stack[top].next_child >= 64 {
            sp -= 1;
            continue;
        }

        let bit = stack[top].next_child;
        let node_index = stack[top].node_index;
        let parent_x = stack[top].x;
        let parent_y = stack[top].y;
        let parent_z = stack[top].z;
        let child_size = stack[top].child_size;

        let (_, parent_ptr, p_mask_lo, p_mask_hi) = read_node(nodes, node_index);

        // Advance to next set bit for the next iteration
        stack[top].next_child = next_set_bit(p_mask_lo, p_mask_hi, bit + 1);

        // Compute child AABB
        let (lx, ly, lz) = bit_to_xyz(bit);
        let cx = parent_x + lx as F * child_size;
        let cy = parent_y + ly as F * child_size;
        let cz = parent_z + lz as F * child_size;

        // Intersect ray with child AABB
        let (ct_enter, ct_exit, _) = ray_aabb(
            ray,
            cx,
            cy,
            cz,
            cx + child_size,
            cy + child_size,
            cz + child_size,
        );

        let ct_enter = ct_enter.max(t_min);
        let ct_exit = ct_exit.min(best_t);

        if ct_enter >= ct_exit {
            continue;
        }

        // Find sparse child index
        let sparse_index = popcount_below(p_mask_lo, p_mask_hi, bit);
        let child_node_index = parent_ptr + sparse_index;

        let (child_is_leaf, child_ptr, c_mask_lo, c_mask_hi) = read_node(nodes, child_node_index);

        if child_is_leaf {
            // This child is a leaf node — iterate over its data entries
            let leaf_child_size = child_size / 4.0;
            let mut leaf_bit = next_set_bit(c_mask_lo, c_mask_hi, 0);
            while leaf_bit < 64 {
                let (vx, vy, vz) = bit_to_xyz(leaf_bit);
                let vox_x = cx + vx as F * leaf_child_size;
                let vox_y = cy + vy as F * leaf_child_size;
                let vox_z = cz + vz as F * leaf_child_size;
                let (vt_enter, vt_exit, vface_axis) = ray_aabb(
                    ray,
                    vox_x,
                    vox_y,
                    vox_z,
                    vox_x + leaf_child_size,
                    vox_y + leaf_child_size,
                    vox_z + leaf_child_size,
                );
                let vt_enter = vt_enter.max(t_min);
                let vt_exit = vt_exit.min(best_t);
                if vt_enter < vt_exit {
                    let leaf_sparse = popcount_below(c_mask_lo, c_mask_hi, leaf_bit);
                    let value = read_data_u8(data, child_ptr + leaf_sparse);
                    best_t = vt_enter;
                    hit.t = vt_enter;
                    hit.value = value;
                    hit.normal = face_normal(ray, vface_axis);
                    hit.pos = [vox_x as u32, vox_y as u32, vox_z as u32];
                    found = true;
                }
                leaf_bit = next_set_bit(c_mask_lo, c_mask_hi, leaf_bit + 1);
            }
        } else if (c_mask_lo != 0 || c_mask_hi != 0) && sp < MAX_DEPTH {
            // Internal node — push onto stack
            let first = next_set_bit(c_mask_lo, c_mask_hi, 0);
            stack[sp] = StackEntry {
                node_index: child_node_index,
                x: cx,
                y: cy,
                z: cz,
                child_size: child_size / 4.0,
                next_child: first,
            };
            sp += 1;
        }
    }

    found
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
