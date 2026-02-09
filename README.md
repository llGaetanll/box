A ray-traced voxel engine using rust-gpu.

**Requirements**:
- Efficiency before all
  - Render large worlds on the screen at high FPS
  - Worlds load quickly
- Ray Tracing
- Let's assume a maximum of **256** kinds of voxel (including the "nothing" voxel)

  This rule is the most arbitrary, we just fix a number for now to see what we can do

- Larger than RAM worlds
- No assumptions on the contents of the world

  Some might be dense, some sparse. Some small, some massive.

- Dynamic world: the user can interact with their *nearby* surroundings

**Implementation**
- Everything in the world is a voxel, the type of which is uniquely identifiable with a number
- Different voxel types render as different colors
- Cubic world chunking

  Probably necessary to support larger than RAM worlds

- Sparse 64-tree for within-chunk voxel storage

  A 4^3 branching factor octree variant. Each node is 12 bytes: a 1-bit leaf
  flag, a 31-bit child pointer, and a 64-bit child mask. Compared to a standard
  octree, this halves the tree depth (fewer pointer indirections on the GPU) and
  uses roughly 3x less memory (~0.19 bytes/voxel vs ~0.57). The 64-bit child
  mask maps directly to hardware popcount for sparse child indexing. Since voxel
  types fit in a byte (max 256), leaf data is compact.

- Two-tier chunk representation: mutable nearby, compressed distant

  The dynamic world requirement means nearby chunks need to support fast single-
  voxel edits, which rules out DAG deduplication for those chunks (DAGs are
  read-only once built). A natural split: chunks within interaction range stay
  as plain 64-trees that can be modified in place, while distant chunks get DAG-
  compressed for memory savings. When the player moves, newly-nearby chunks are
  "unpacked" from DAG back to tree (cheap, since the DAG is already a tree with
  shared nodes -- you just copy the relevant path), and newly-distant chunks get
  compressed. This keeps the working set of editable chunks small while still
  getting 10-100x compression on the bulk of the world.

- GPU streaming with a chunk cache

  For larger-than-RAM worlds, only chunks near the camera need to be in VRAM.
  A chunk cache holds the active working set, streaming chunks in from CPU
  memory (or disk) as the camera moves and evicting the least recently used.
  Aokana (2025) demonstrated that only ~5% of a large world's data needs to
  be resident at any time. The coarse chunk grid acts as the top-level spatial
  index for ray traversal, dispatching into per-chunk 64-trees.

- Breadth-first node layout in GPU buffers

  When flattening the 64-tree for GPU upload, ordering nodes breadth-first
  (all level-0 nodes, then level-1, etc.) improves cache coherence for ray
  traversal, since nearby rays tend to visit the same upper tree levels.
  Benchmarks in the literature show ~25ms/frame improvement over depth-first.

See docs/notes/voxel-data-structures.md for the research behind these.
