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
