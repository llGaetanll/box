use crate::traverse::CHILD_MASK_SHIFT;
use crate::traverse::CHILD_PTR_MASK;
use crate::traverse::LEAF_BIT;

/// A node in the octree during construction.
enum Node {
    Empty,
    Leaf(u32),
    Internal(Box<[Node; 8]>),
}

/// Sparse voxel octree. Supports a world of 2^depth units per axis.
///
/// Build on CPU with `set`/`get`, then call `flatten()` to produce a packed
/// `Vec<u32>` for GPU upload.
pub struct Octree {
    root: Node,
    depth: u32,
}

impl Octree {
    /// Create an empty octree. The world spans `[0, 2^depth)` on each axis.
    pub fn new(depth: u32) -> Self {
        assert!(depth > 0 && depth <= 20, "depth must be in 1..=20");
        Self {
            root: Node::Empty,
            depth,
        }
    }

    /// The world size along each axis.
    pub fn size(&self) -> u32 {
        1 << self.depth
    }

    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Set a voxel at `(x, y, z)` to `value`. Value 0 is reserved for empty
    /// (setting to 0 effectively removes the voxel).
    pub fn set(&mut self, x: u32, y: u32, z: u32, value: u32) {
        let size = self.size();
        assert!(
            x < size && y < size && z < size,
            "coordinates out of bounds"
        );
        Self::set_recursive(&mut self.root, x, y, z, value, self.depth);
    }

    /// Get the voxel value at `(x, y, z)`, or `None` if empty.
    pub fn get(&self, x: u32, y: u32, z: u32) -> Option<u32> {
        let size = self.size();
        assert!(
            x < size && y < size && z < size,
            "coordinates out of bounds"
        );
        Self::get_recursive(&self.root, x, y, z, self.depth)
    }

    /// Serialize the octree into a packed `Vec<u32>` for GPU consumption.
    ///
    /// The packed format uses one `u32` per node:
    /// - **Leaf**: bit 31 set, bits 30..0 = voxel value
    /// - **Internal**: bit 31 clear, bits 30..24 = 8-bit child mask,
    ///   bits 23..0 = index of first child in the vec
    ///
    /// Children of an internal node are stored contiguously in child-mask
    /// order (only present children occupy a slot).
    ///
    /// Returns an empty vec if the octree is entirely empty.
    pub fn flatten(&self) -> Vec<u32> {
        let mut data = Vec::new();
        match &self.root {
            Node::Empty => {}
            Node::Leaf(v) => {
                data.push(LEAF_BIT | (*v & !LEAF_BIT));
            }
            Node::Internal(_) => {
                // Reserve slot 0 for root, then fill it.
                data.push(0);
                Self::flatten_internal(&self.root, 0, &mut data);
            }
        }
        data
    }

    // --- private helpers ---

    fn octant(x: u32, y: u32, z: u32, level: u32) -> usize {
        let bit = level - 1;
        let ox = ((x >> bit) & 1) as usize;
        let oy = ((y >> bit) & 1) as usize;
        let oz = ((z >> bit) & 1) as usize;
        ox | (oy << 1) | (oz << 2)
    }

    fn set_recursive(node: &mut Node, x: u32, y: u32, z: u32, value: u32, level: u32) {
        if level == 0 {
            if value == 0 {
                *node = Node::Empty;
            } else {
                *node = Node::Leaf(value);
            }
            return;
        }

        // Ensure this is an internal node
        if !matches!(node, Node::Internal(_)) {
            let children: [Node; 8] = core::array::from_fn(|_| match node {
                Node::Leaf(v) => Node::Leaf(*v),
                _ => Node::Empty,
            });
            *node = Node::Internal(Box::new(children));
        }

        if let Node::Internal(children) = node {
            let oct = Self::octant(x, y, z, level);
            Self::set_recursive(&mut children[oct], x, y, z, value, level - 1);
            Self::try_collapse(node);
        }
    }

    fn try_collapse(node: &mut Node) {
        if let Node::Internal(children) = node {
            let first = match &children[0] {
                Node::Empty => None,
                Node::Leaf(v) => Some(*v),
                Node::Internal(_) => return,
            };

            for child in children.iter().skip(1) {
                let this = match child {
                    Node::Empty => None,
                    Node::Leaf(v) => Some(*v),
                    Node::Internal(_) => return,
                };
                if this != first {
                    return;
                }
            }

            *node = match first {
                None => Node::Empty,
                Some(v) => Node::Leaf(v),
            };
        }
    }

    fn get_recursive(node: &Node, x: u32, y: u32, z: u32, level: u32) -> Option<u32> {
        match node {
            Node::Empty => None,
            Node::Leaf(v) => Some(*v),
            Node::Internal(children) => {
                if level == 0 {
                    return None;
                }
                let oct = Self::octant(x, y, z, level);
                Self::get_recursive(&children[oct], x, y, z, level - 1)
            }
        }
    }

    /// Write an internal node's header into `data[header_idx]` and append its
    /// children (and their subtrees) to `data`.
    fn flatten_internal(node: &Node, header_idx: usize, data: &mut Vec<u32>) {
        let Node::Internal(children) = node else {
            panic!("flatten_internal called on non-internal node");
        };

        // Build child mask.
        let mut child_mask: u8 = 0;
        for (i, child) in children.iter().enumerate() {
            if !matches!(child, Node::Empty) {
                child_mask |= 1 << i;
            }
        }
        let child_count = child_mask.count_ones() as usize;

        // Reserve contiguous slots for child entries.
        let first_child = data.len();
        data.resize(data.len() + child_count, 0);

        // Write this node's header.
        data[header_idx] =
            ((child_mask as u32) << CHILD_MASK_SHIFT) | (first_child as u32 & CHILD_PTR_MASK);

        // Fill each child slot.
        let mut slot = 0;
        for (i, child) in children.iter().enumerate() {
            if child_mask & (1 << i) == 0 {
                continue;
            }
            match child {
                Node::Empty => unreachable!(),
                Node::Leaf(v) => {
                    data[first_child + slot] = LEAF_BIT | (*v & !LEAF_BIT);
                }
                Node::Internal(_) => {
                    Self::flatten_internal(child, first_child + slot, data);
                }
            }
            slot += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_octree_returns_none() {
        let tree = Octree::new(4);
        assert_eq!(tree.get(0, 0, 0), None);
        assert_eq!(tree.get(7, 7, 7), None);
        assert_eq!(tree.get(15, 15, 15), None);
    }

    #[test]
    fn single_voxel() {
        let mut tree = Octree::new(4);
        tree.set(3, 5, 7, 42);
        assert_eq!(tree.get(3, 5, 7), Some(42));
        assert_eq!(tree.get(0, 0, 0), None);
    }

    #[test]
    fn multiple_voxels() {
        let mut tree = Octree::new(4);
        tree.set(0, 0, 0, 1);
        tree.set(15, 15, 15, 2);
        tree.set(8, 4, 2, 3);

        assert_eq!(tree.get(0, 0, 0), Some(1));
        assert_eq!(tree.get(15, 15, 15), Some(2));
        assert_eq!(tree.get(8, 4, 2), Some(3));
        assert_eq!(tree.get(1, 1, 1), None);
    }

    #[test]
    fn overwrite_voxel() {
        let mut tree = Octree::new(4);
        tree.set(5, 5, 5, 10);
        assert_eq!(tree.get(5, 5, 5), Some(10));
        tree.set(5, 5, 5, 20);
        assert_eq!(tree.get(5, 5, 5), Some(20));
    }

    #[test]
    fn remove_voxel_with_zero() {
        let mut tree = Octree::new(4);
        tree.set(3, 3, 3, 5);
        assert_eq!(tree.get(3, 3, 3), Some(5));
        tree.set(3, 3, 3, 0);
        assert_eq!(tree.get(3, 3, 3), None);
    }

    #[test]
    fn flatten_empty() {
        let tree = Octree::new(4);
        let data = tree.flatten();
        assert!(data.is_empty());
    }

    #[test]
    fn flatten_single_leaf() {
        let mut tree = Octree::new(1); // 2x2x2 world
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    tree.set(x, y, z, 7);
                }
            }
        }
        let data = tree.flatten();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0], LEAF_BIT | 7);
    }

    #[test]
    fn flatten_and_structure() {
        let mut tree = Octree::new(1); // 2x2x2
        tree.set(0, 0, 0, 1);
        let data = tree.flatten();
        assert!(!data.is_empty());
        // Root should be internal (bit 31 clear)
        assert_eq!(data[0] & LEAF_BIT, 0);
    }

    #[test]
    fn memory_efficiency() {
        let mut tree = Octree::new(10); // 1024^3 world
        tree.set(500, 500, 500, 1);
        let data = tree.flatten();
        // depth=10: ~10 internal nodes, each with 1 child = ~20 entries
        assert!(
            data.len() < 100,
            "flatten produced {} entries for 1 voxel in 1024^3 world",
            data.len()
        );
    }

    #[test]
    fn flatten_traverse_roundtrip() {
        use gpu_prim::Ray;
        use gpu_prim::Vec3;

        use crate::traverse::VoxelHit;
        use crate::traverse::trace_octree;

        let mut tree = Octree::new(4); // 16x16x16
        // Place a voxel at (3, 0, 3)
        tree.set(3, 0, 3, 42);

        let data = tree.flatten();
        assert!(!data.is_empty());

        // Shoot a ray straight down at (3.5, ?, 3.5) — should hit the voxel
        let ray = Ray::new(Vec3::new(3.5, 10.0, 3.5), Vec3::new(0.0, -1.0, 0.0), 0.0);
        let mut hit = VoxelHit::default();
        let found = trace_octree(&data, 4, &ray, 0.001, 1000.0, &mut hit);
        assert!(found, "ray should hit the voxel");
        assert_eq!(hit.value, 42);
        // Normal should be +Y (top face, ray coming from above)
        assert!(
            (hit.normal - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6,
            "expected +Y normal, got {:?}",
            hit.normal
        );

        // Shoot a ray that misses (off to the side)
        let ray_miss = Ray::new(Vec3::new(0.5, 10.0, 0.5), Vec3::new(0.0, -1.0, 0.0), 0.0);
        let mut hit2 = VoxelHit::default();
        let found2 = trace_octree(&data, 4, &ray_miss, 0.001, 1000.0, &mut hit2);
        assert!(!found2, "ray should miss");
    }
}

#[test]
fn test_menger_traversal() {
    use gpu_prim::Ray;
    use gpu_prim::Vec3;

    use crate::traverse::VoxelHit;
    use crate::traverse::trace_octree;

    // First: simple test - single voxel in a depth-6 world
    {
        let mut octree = Octree::new(6);
        octree.set(32, 32, 32, 5);
        let data = octree.flatten();
        eprintln!("Simple: {} entries", data.len());

        let ray = Ray::new(Vec3::new(32.5, 32.5, 40.0), Vec3::new(0.0, 0.0, -1.0), 0.0);
        let mut hit = VoxelHit::default();
        let found = trace_octree(&data, 6, &ray, 0.001, 1000.0, &mut hit);
        eprintln!("Simple hit: {found}, t={}, value={}", hit.t, hit.value);
        assert!(found, "Should hit single voxel in depth-6 world");
        assert_eq!(hit.value, 5);
    }

    // Second: small filled cube in depth-6 world
    {
        let mut octree = Octree::new(6);
        for x in 30..34 {
            for y in 30..34 {
                for z in 30..34 {
                    octree.set(x, y, z, 3);
                }
            }
        }
        let data = octree.flatten();
        eprintln!("Cube: {} entries", data.len());

        let ray = Ray::new(Vec3::new(32.0, 32.0, 50.0), Vec3::new(0.0, 0.0, -1.0), 0.0);
        let mut hit = VoxelHit::default();
        let found = trace_octree(&data, 6, &ray, 0.001, 1000.0, &mut hit);
        eprintln!("Cube hit: {found}, t={}, value={}", hit.t, hit.value);
        assert!(found, "Should hit cube in depth-6 world");
    }

    // Third: menger sponge
    {
        fn is_menger(mut x: u32, mut y: u32, mut z: u32, size: u32) -> bool {
            let mut s = size;
            while s > 1 {
                s /= 3;
                let cx = (x / s) % 3;
                let cy = (y / s) % 3;
                let cz = (z / s) % 3;
                let center_count = u32::from(cx == 1) + u32::from(cy == 1) + u32::from(cz == 1);
                if center_count >= 2 {
                    return false;
                }
                x %= s;
                y %= s;
                z %= s;
            }
            true
        }

        let depth = 6u32;
        let size = 1u32 << depth;
        let mut octree = Octree::new(depth);
        let sponge_size = 27u32;
        let offset = (size - sponge_size) / 2;
        for x in 0..sponge_size {
            for y in 0..sponge_size {
                for z in 0..sponge_size {
                    if is_menger(x, y, z, sponge_size) {
                        octree.set(x + offset, y + offset, z + offset, 3);
                    }
                }
            }
        }

        // Verify some voxels exist
        assert_eq!(octree.get(offset, offset, offset), Some(3));
        assert_eq!(octree.get(offset + 13, offset + 13, offset + 13), None); // center hole

        let data = octree.flatten();
        eprintln!("Menger: {} entries", data.len());

        // Ray straight at a known-filled corner
        let tx = (offset as f32) + 0.5;
        let ty = (offset as f32) + 0.5;
        let tz = (offset as f32) + 0.5;
        let ray = Ray::new(Vec3::new(tx, ty, tz + 20.0), Vec3::new(0.0, 0.0, -1.0), 0.0);
        let mut hit = VoxelHit::default();
        let found = trace_octree(&data, depth, &ray, 0.001, 1000.0, &mut hit);
        eprintln!(
            "Menger corner hit: {found}, t={}, value={}, pos={:?}",
            hit.t, hit.value, hit.pos
        );
        assert!(found, "Ray should hit Menger sponge corner");
    }
}
