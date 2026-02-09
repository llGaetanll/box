Research on voxel data structures, with a focus on memory representation and
GPU-friendliness. The goal is to figure out the best structure for our engine,
which needs to handle worlds ranging from small and dense to massive and sparse,
all ray-traced on the GPU.

Our current implementation uses a sparse voxel octree (SVO) with a flattened
Vec<u32> for GPU upload, where each node is a single u32 encoding a child mask
and pointer (internal nodes) or a voxel value with a leaf bit set. This is a
reasonable starting point, but there's a rich landscape of alternatives worth
understanding before committing to a direction.


**The Sparse Voxel Octree (SVO)**

The SVO is the most studied voxel data structure. It recursively subdivides a
cubic volume into 8 octants, pruning branches that are entirely empty or
uniform. The seminal work is "Efficient Sparse Voxel Octrees" by Laine and
Karras at NVIDIA (2010), which established the standard GPU encoding: each node
is packed into as little as 4 bytes containing an 8-bit valid mask (which
children exist), an 8-bit non-leaf mask (which children are internal vs leaf),
and a 15-bit child pointer plus a far-pointer bit for when children are too
distant in memory for the relative offset to fit.
(https://research.nvidia.com/publication/2010-02_efficient-sparse-voxel-octrees)

Our current encoding is similar in spirit. The SVO's main strength is that it
naturally handles sparsity: empty space costs nothing, and uniform regions
collapse into single leaves. Memory usage scales with surface complexity rather
than volume. For a model with N surface voxels, a well-constructed SVO uses
roughly O(8N/7) storage for the tree structure (since each internal node has at
most 8 children, and the geometric series converges). In practice, Eisenwave's
compression analysis found SVOs use around 0.57 bytes per voxel for typical
scenes.
(https://eisenwave.github.io/voxel-compression-docs/svo/svo.html)

The main weaknesses of the SVO are:

Traversal requires pointer chasing. Each level of the tree is a memory
indirection, and GPU caches don't love this. For a 1024^3 world (depth 10),
every ray-voxel test requires up to 10 pointer dereferences.

Modification is expensive. Changing a single voxel requires walking from root
to leaf and potentially reallocating along the way. This matters for dynamic
worlds.

The branching factor of 8 is somewhat small. Each traversal step only
eliminates 7/8 of the remaining space. Wider trees can skip more space per step.


**The 64-Tree (Tetrahexacontree)**

A compelling alternative to the standard octree is to use a branching factor of
4^3 = 64 instead of 2^3 = 8. This is sometimes called a "squashed octree"
(combining two octree levels into one) or a sparse 64-tree. The key insight is
that a 64-bit integer perfectly represents the child mask for 64 children, and
modern CPUs/GPUs have native popcount instructions to efficiently index into the
sparse child array.

The best practical writeup of this approach is by dubiousconst282, who built a
high-performance voxel ray tracer around this structure. Each node is 12 bytes:
a 1-bit leaf flag, a 31-bit child pointer, and a 64-bit child mask. Despite the
larger per-node size, the tree is shallower (half the depth of an octree for the
same resolution), which means fewer pointer dereferences per traversal. In
benchmarks, the 64-tree achieved approximately 0.19 bytes per voxel, compared to
roughly 0.57 for a standard SVO, a 3x improvement. On real voxelized scenes, the
64-tree was about 60% smaller than an equivalent ESVO.
(https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/)

There's a clever float-bit trick used for traversal: the tree operates in a
normalized coordinate space [1.0, 2.0), and since IEEE754 floats encode the
fractional part in the mantissa, you can directly manipulate the mantissa in
2-bit chunks to address cells at each tree level. This eliminates a lot of
arithmetic that traditional octree traversal needs.

A Rust implementation of this structure exists (https://github.com/expenses/tree64)
and the VoxelRT renderer (https://github.com/dubiousconst282/VoxelRT) demonstrates
it in practice. This is a direct conceptual upgrade from a standard octree: same model, better
memory efficiency, fewer indirections, and well-suited to GPU traversal.


**Sparse Voxel DAGs (SVDAGs)**

The next step beyond SVOs is to notice that many subtrees in a voxel scene are
identical. An octree stores each subtree independently even if two regions of
space contain the same pattern. A Directed Acyclic Graph (DAG) merges identical
subtrees so they share a single copy in memory.

The foundational paper is "High Resolution Sparse Voxel DAGs" by Kämpe, Sintorn,
and Assarsson (2013). The construction algorithm is elegant: starting from a
built SVO, you work bottom-up, hashing each subtree and merging duplicates. At
the leaf level, identical leaf nodes are merged. Then at the next level up,
nodes whose child masks and child pointers (after merging) are identical get
merged too, and so on up to the root.
(https://www.cse.chalmers.se/~uffe/HighResolutionSparseVoxelDAGs.pdf)

The compression is dramatic. In all tested scenes, node counts were reduced by
1 to 3 orders of magnitude compared to SVOs. The EPICCITADEL scene at 128K^3
resolution required 945MB as a DAG vs 5.1GB as an SVO, and this is for binary
geometry (occupied or not) without material data.

Traversal is essentially the same as an SVO: you walk from root to leaf,
following child pointers. The DAG structure is transparent to the ray traversal
algorithm. Performance is on par with or slightly faster than SVOs, because the
smaller memory footprint improves cache behavior.

The main limitation of basic SVDAGs is that they only merge *identical*
subtrees. Two subtrees that are mirror images or rotations of each other remain
separate.


**SSVDAGs (Symmetry-Aware Sparse Voxel DAGs)**

Villanueva and Marton (2016) extended SVDAGs by also merging subtrees that are
related by axis-aligned reflections. Since there are 3 axes and each can be
independently reflected, this gives up to 8 symmetry variants of each subtree
that can share storage. The transformation is encoded in 3 extra bits in the
child pointer.
(https://dl.acm.org/doi/10.1145/2856400.2856420)

The results are striking: the PowerPlant scene at 64K^3 resolution with nearly
6 billion non-empty voxels was stored in under 86MB at 0.123 bits per voxel. A
sparse voxel octree would need 16.2GB (200x more), and a plain SVDAG would need
167MB (nearly double). SSVDAGs also use variable bit-rate encoding for child
pointers, exploiting the skewed frequency distribution of node references (some
nodes are referenced far more often than others, so they get shorter pointers).

The catch is that SSVDAGs, like SVDAGs, are essentially static. You build them
once from an SVO and the result is a highly compressed, read-only structure.
This is perfect for static geometry but doesn't work for dynamic worlds.

The most recent extension is Transform-Aware SVDAGs (2025), which generalize
beyond reflections to include translations and other transforms.
(https://dl.acm.org/doi/full/10.1145/3728301)


**HashDAGs**

The HashDAG (Careil et al., 2020) addresses the biggest limitation of SVDAGs:
they can't be edited. The idea is to embed the DAG into a hash table, where
each unique subtree is stored once and identified by its hash. When you modify a
voxel, you create new nodes along the path from root to leaf, reusing existing
nodes from the hash table wherever possible. This is essentially a persistent
data structure, similar to how functional programming languages handle immutable
trees.
(https://github.com/Phyronnaz/HashDAG)

Each modification produces a new root node while maximizing reuse of historical
data. This also gives you free undo: old roots still reference the old version
of the world.

The tradeoff is performance. As the tree gets deeper, the number of hash table
lookups per query increases, reducing cache hit rates. For scenes above 32K
resolution, some newer approaches (like Aokana) achieve 2-4x faster rendering.
HashDAGs also use more memory than static SVDAGs because the hash table has
overhead, though still far less than an SVO.

HashDAGs are the main option for dynamic editing of large worlds while
maintaining compression.


**VDB and NanoVDB**

VDB is a data structure designed by Ken Museth at DreamWorks Animation for
volumetric simulation (fluids, smoke, etc). Unlike an octree which has a fixed
branching factor of 8, VDB uses a tree with configurable branching factors at
each level, typically something like 32^3 at the root, 16^3 at internal levels,
and 8^3 at the leaves. The key is that the branching factors are fixed at
compile time, not runtime, so the compiler can optimize aggressively.
(https://www.museth.org/Ken/Publications_files/Museth_TOG13.pdf)

The leaf nodes are dense 8x8x8 grids. Internal nodes use bitmasks to track
which children are active, and a secondary bitmask to distinguish between tile
values (uniform regions) and pointers to child nodes. This means VDB can
efficiently represent both sparse regions (empty space) and dense regions
(solid volumes) within the same structure.

NanoVDB is the GPU-friendly version: it linearizes the VDB tree into a single
contiguous memory block with no pointers (all references are offsets), making it
suitable for GPU storage buffers. It also bakes in min/max values and AABBs at
each node for accelerating ray tracing.
(https://dl.acm.org/doi/fullHtml/10.1145/3450623.3464653)

VDB excels at dynamic topology changes, which is why it dominates in simulation.
For static voxel worlds with ray tracing, VDB is likely over-engineered: its
multi-level branching factor adds complexity, and its compression ratios aren't
as good as DAG-based approaches for static scenes. VDB's strength is frequent
topology changes (adding/removing large regions of voxels).


**Brickmaps**

A brickmap is a two-level structure: a coarse grid where each cell either is
empty or points to a "brick" (typically an 8x8x8 dense voxel grid). All bricks
are stored in a large pool (a 3D texture on the GPU, sometimes called a "brick
pool").
(https://github.com/stijnherfst/BrickMap)

The advantages are simplicity and cache coherence. Nearby voxels within a brick
are stored contiguously in memory, which means traversal within a brick has
excellent locality. The coarse grid lets you skip empty space at the macro
level. There's no deep tree traversal, no pointer chasing across many levels,
just one grid lookup plus a dense brick read.

GigaVoxels (Crassin et al., 2009) extended brickmaps with a full octree of
bricks, a streaming system that loads bricks on demand based on what the camera
can see, and LOD through the octree hierarchy. The brick pool acts like a cache,
with timestamps for eviction. This enabled interactive rendering of billions of
voxels by only keeping visible bricks in GPU memory.
(https://maverick.inria.fr/Publications/2009/CNLE09/CNLE09.pdf)

The downside is that brickmaps have relatively poor compression for sparse
scenes. Every allocated brick costs 8^3 = 512 voxels of storage regardless of
how many are actually filled. They work best when the world has large contiguous
solid regions.


**Grid Hierarchies**

A grid hierarchy is a stack of 3D grids at progressively lower resolutions.
The finest grid stores the actual voxel data. Each coarser grid stores a single
bit per cell: "is there anything in the corresponding region of the finer
grid?" This is conceptually like a mipmap chain for occupancy.
(https://bink.eu.org/fast-voxel-datastructures/)

For traversal, you start at the coarsest grid and use it to skip large empty
regions, then descend to finer grids as needed. This gives octree-like
space-skipping without any pointers: the grids are flat arrays, and the
position in a coarser grid directly determines the position in the finer grid
through simple arithmetic.

The appeal of grid hierarchies is simplicity and speed. No tree construction,
no pointer management, trivially parallelizable on the GPU. They're so fast to
rebuild that some implementations reconstruct the entire hierarchy every frame.
The memory cost is fixed at O(n^3) for the finest grid, plus negligible overhead
for the coarser levels (each level is 8x smaller).

The limitation is obvious: the finest grid is dense, so memory scales cubically.
A 1024^3 world with 1-byte voxels is 1GB. This makes grid hierarchies
impractical for very large or very sparse worlds, but excellent for small to
medium dense worlds.


**Hybrid and Emerging Approaches**

Recent research (2024-2025) suggests that the best results come from combining
structures rather than picking a single one.

"Hybrid Voxel Formats for Efficient Ray Tracing" (2024) systematically
evaluated combinations of flat grids, SDFs, SVOs, and SVDAGs at different
levels of the hierarchy. The consistently best configurations were patterns like
R(3^3) G(8) or R(4^3) G(7), meaning a coarse grid with resolution 3^3 or 4^3
at the top, followed by a finer grid with resolution 8 or 7 below. These
hybrid formats achieved Pareto-optimal tradeoffs between memory and rendering
speed that no single format could match.
(https://arxiv.org/html/2410.14128v1)

Aokana (2025) is the most impressive recent system, a GPU-driven framework for
open-world voxel games. Rather than one deep SVDAG, Aokana divides the world
into 256^3-voxel chunks, each stored as a separate shallow SVDAG. The deepest
three leaf layers use a 64-bit bitmap to represent 4x4x4 voxel blocks. This
chunk-based approach dramatically reduces cache misses (since each chunk's DAG
fits better in cache than one world-spanning DAG), and enables streaming where
only about 5% of the scene data needs to be in VRAM at any time. At 64K
resolution, Aokana renders ten-billion-voxel scenes in about 6ms per frame,
2-4x faster than HashDAG with up to 9x less memory.
(https://arxiv.org/html/2505.02017v1)


**GPU Memory Layout Considerations**

For any tree-like structure on the GPU, how you lay out nodes in memory matters
as much as the tree structure itself.

Breadth-first layout stores all nodes at level 0, then all at level 1, etc.
This means sibling nodes are always adjacent in memory, which is good for GPU
coherence since nearby rays tend to traverse similar tree paths. Benchmarks
show breadth-first is roughly 25ms faster than depth-first per frame, with
better worst-case performance.
(https://bcmpinc.wordpress.com/2015/08/09/moving-towards-a-gpu-implementation/)

Depth-first layout is better for CPU traversal (you only need a stack, and
spatial locality follows the recursion), but causes scattered memory access on
the GPU.

The takeaway is that children of the same parent should be stored contiguously,
and ordering nodes breadth-first within the buffer improves GPU cache behavior.

Morton order (Z-order curves) is another option: it maps 3D coordinates to 1D
while preserving spatial locality. This is particularly useful for flat grids
and brickmaps, less critical for tree structures that already have spatial
locality through their hierarchy.


**Signed Distance Fields (SDFs)**

Worth mentioning even though they're a different paradigm: instead of storing
occupancy, each voxel stores its distance to the nearest surface. During ray
marching, you can safely advance the ray by the distance value at the current
position, since you know there's no surface within that radius. This gives the
fastest possible ray traversal, only one lookup per step, and the step sizes
adapt automatically to geometry density.

The catch is that SDFs are expensive to construct and update, they only
represent surfaces (not arbitrary voxel data like materials), and they have
limited precision at sharp features. For a pure voxel engine where each voxel
has a material ID, SDFs don't directly apply. But a hybrid approach where we
store distance-to-surface in empty voxels alongside our octree could accelerate
traversal through large empty regions.

