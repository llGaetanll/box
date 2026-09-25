A sparse 64-tree (tetrahexacontree) for voxel storage. Where a standard octree
splits space into 2x2x2 = 8 children per node, a 64-tree splits into 4x4x4 =
64 children. The 64-bit child bitmask maps directly to a machine word, so
hardware popcount gives O(1) sparse child indexing. The tree is half as deep as
an equivalent octree (fewer pointer indirections on the GPU) and uses roughly 3x
less memory (~0.19 bytes/voxel vs ~0.57 for a standard SVO).

Based on https://github.com/expenses/tree64.

**Key components:**

- `Node` -- 12-byte packed struct (1-bit leaf flag, 31-bit child pointer,
  64-bit child bitmask). Children are stored contiguously; the pointer addresses
  the first, and `popcount(mask & ((1 << index) - 1))` gives the offset to any
  child.

- `Tree64<T>` -- the tree itself, generic over voxel type `T`. Two flat
  arrays (`nodes: Vec<Node>`, `data: Vec<T>`) plus an undo/redo history.
  Supports construction from a `VoxelModel`, point lookup, single-voxel and box
  modification, and binary serialization.

- `PopMaskedData<T>` -- helper for working with sparse 64-element arrays.
  Expands a compact slice + bitmask into a flat `[T; 64]` for random access,
  then compresses back.

- `Edits` -- append-only undo/redo. Because modifications never mutate existing
  nodes (they append new ones and push a new root), previous tree states remain
  valid and can be restored by rewinding the root pointer.

**Note on memory growth:** Modifications are append-only, so dead nodes and data
accumulate in the arrays over time. There is no compaction. In practice this is
fine because mutable chunks have bounded editing lifetimes -- when a chunk
transitions to a compressed distant representation the entire `Tree64` is
dropped. If long editing sessions ever cause memory pressure, the simplest fix
is to rebuild the tree from scratch via `Tree64::new`, which produces a
perfectly compact tree (at the cost of losing undo history).
