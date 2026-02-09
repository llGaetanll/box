Research on improving the performance of our GPU voxel ray tracer. The current
implementation is extremely naive: a stack-based DFS traversal of the tree64
that tests every populated child's AABB independently (no ordering, no early
termination on first hit), combined with a pure Monte Carlo path tracer doing
random Lambertian scattering with a hard 8-bounce cap, 1 sample per pixel per
frame, no accumulation, and a raw clamp to [0,1] for output. There's a lot of
room to improve.

The optimizations split naturally into two categories: making the tree traversal
faster (fewer nodes visited, less work per node), and making the path tracing
less noisy (more signal per sample, fewer wasted samples). Both matter, but
traversal is the inner loop — every bounce of every path calls it — so it has
the highest leverage.


**The core traversal problem**

Our current trace_tree64 works by DFS over the tree, iterating through every
set bit in each node's pop_mask and doing an independent ray-AABB intersection
test for each child. This has two major problems:

First, it visits children in bitmask order (0, 1, 2, ... 63), not in
ray-distance order. A child at bit 0 might be behind the camera while the
actual nearest hit is at bit 55. We test all of them anyway, keeping track of
the best hit found so far. We do cull children whose AABB is entirely beyond
best_t, but only after computing the intersection — we can't skip the work of
testing them.

Second, when we exit a child without finding a hit, we pop the stack and resume
iterating through the parent's remaining children. This re-reads the parent
node from memory every iteration (re-calling read_node on the same index),
which is wasteful, and doesn't exploit the fact that we already know exactly
where the ray exits the child AABB.

The fix for both problems is the same: replace the unordered DFS with a
DDA-style traversal that steps through each node's 4x4x4 grid in ray order.


**DDA traversal within tree64 nodes**

The classic grid traversal algorithm by Amanatides and Woo (1987) steps a ray
through a uniform grid by maintaining parametric distances to the next cell
boundary on each axis (tMaxX, tMaxY, tMaxZ) and always stepping on whichever
axis has the smallest tMax. Each step advances exactly one cell, and cells are
visited in the order the ray passes through them. This means you can stop at the
first occupied cell — it's guaranteed to be the nearest one.
(http://www.cse.yorku.ca/~amana/research/grid.pdf)

For our tree64, each internal node is a 4x4x4 grid of children. We can run the
Amanatides-Woo algorithm within that grid: when the ray enters a node, compute
which of the 4x4x4 cells it enters first (from the entry point's position
relative to the node's AABB), set up tMax and tDelta for the 4-cell-per-axis
grid, and step through cells in ray order. At each cell, check the pop_mask
bit. If set, descend into the child (and if it's a leaf, you've found the
nearest hit — done). If not set, step to the next cell. When you step out of
the 4x4x4 grid, pop up to the parent and continue its DDA from where you left
off.

This eliminates redundant AABB tests (the DDA computes the next cell from the
exit point of the previous one, not from scratch) and gives us front-to-back
ordering for free, enabling early termination on first hit. For primary rays
this is a major win: instead of testing every populated child in the node, we
only test the ones the ray actually passes through, in order, and stop at the
first one.

dubiousconst282's VoxelRT implements exactly this for the tree64 structure.
Their traversal maintains parametric side distances for each axis and steps
through the 4x4x4 grid using DDA, descending into the tree hierarchy or
backtracking to ancestors as needed.
(https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/)


**The IEEE754 mantissa trick**

VoxelRT uses a clever encoding: coordinates are normalized to [1.0, 2.0). In
IEEE754, floats in this range have exponent 0 and the fractional part is stored
directly in the mantissa bits. Since each tree64 level subdivides by 4 (2 bits
per axis), the cell position at each level corresponds to a specific 2-bit
slice of the mantissa. Extracting the cell index at level L is just a bit shift
and mask:

  cellPos = asuint(pos) >> scaleExp & 3
  cellIndex = cellPos.x + cellPos.z*4 + cellPos.y*16

where scaleExp decreases by 2 for each level deeper in the tree. This avoids
floating-point division and floor operations that traditional DDA needs for
computing cell coordinates — it's pure integer bit manipulation on the float's
representation.

This trick works because the tree's branching factor (4 per axis) aligns with
powers of 2 in the mantissa. An octree (2 per axis) would use 1-bit slices, a
16-tree would use 4-bit slices, etc. For the 64-tree it's a natural fit.


**Ray-octant mirroring**

The DDA intersection math has branches for the sign of each ray direction
component: when dir.x is negative, the entry face is on the positive side and
the formulas are different. VoxelRT eliminates this by reflecting the ray into
a canonical octant at the start of traversal. A precomputed mirror mask flips
the relevant mantissa bits:

  mirrorMask = 0
  if dir.x > 0: mirrorMask |= 3 << 0
  if dir.y > 0: mirrorMask |= 3 << 4
  if dir.z > 0: mirrorMask |= 3 << 2

Cell indices are XORed with this mask during lookup: childIdx ^ mirrorMask.
Since flipping bits of a 2-bit integer bounded by [0,3] is equivalent to
reversing it (0↔3, 1↔2), this effectively mirrors the coordinate system so the
ray always travels in the negative direction on all axes, simplifying the
intersection formulas to a single case.

VoxelRT measured this at ~11% faster (6,358 vs 7,052 cycles per ray). It's a
small change with a reliable payoff.


**Ancestor memoization**

When the DDA steps out of a node's 4x4x4 grid, we need to continue traversal
in the parent node. The naive approach is to restart from the root and descend
again. VoxelRT instead maintains a small stack of node indices along the
current traversal path, indexed by tree level:

  stack[scaleExp >> 1] = nodeIdx

After stepping the ray to the next position, the algorithm detects which tree
level changed by XORing the old and new positions' mantissa bits:

  diffPos = asuint(pos) ^ asuint(cellMin)
  diffExp = firstbithigh((diffPos.x | diffPos.y | diffPos.z) & 0xFFAAAAAA)

If diffExp > scaleExp, the ray crossed a boundary at a coarser level, so
scaleExp is set to diffExp and the node index is recovered from the stack. This
avoids redundant root-to-leaf descents and was the single biggest optimization
in VoxelRT's pipeline: nearly 2x speedup (8,896 from 16,903 cycles/ray) on
integrated GPUs.


**Bitmask coalescing (single-cell skipping)**

When the DDA steps through an empty cell, it advances one cell at a time. But
if the node's pop_mask shows that all four cells along a particular axis slice
are empty, we can skip the entire slice in one step. VoxelRT checks this with:

  if (node.ChildMask >> (childIdx & 0b101010) & 0x00330033) == 0:
      advScaleExp++

This coalesces multiple empty-cell steps into a single larger step by
temporarily treating the node as being at a coarser level. The bit patterns
0b101010 and 0x00330033 select 2x2x2 sub-blocks within the 4x4x4 grid. If
the sub-block is entirely empty, we can skip it in one step rather than four.

VoxelRT measured 21% improvement from this (7,052 vs 8,896 cycles/ray). For
our tree64 with its u32-pair pop_mask, the same bit manipulation works — we just
need to handle the lo/hi split.


**Beam optimization (coarse-to-fine pre-pass)**

Before casting full-resolution rays, render a lower-resolution "beam" image
(1/4 or 1/8 resolution) where each pixel represents a group of final pixels.
These coarse rays only need to find the approximate nearest surface distance.
The full-resolution pass then reads this coarse depth and advances each ray's
starting t to skip the empty space between the camera and the nearest geometry.

This is described in the ESVO paper (Laine & Karras 2010) and used by Aokana
(2025). dubiousconst282's post mentions it as something their implementation is
"still missing" but would benefit from. For scenes with large empty regions
(most voxel worlds), the beam pre-pass can cut traversal work dramatically
because rays start near surfaces rather than from the camera.
(https://research.nvidia.com/publication/2010-02_efficient-sparse-voxel-octrees)
(https://arxiv.org/html/2505.02017v1)

Implementation-wise this is a separate draw call at reduced resolution, writing
to a depth texture that the main pass reads from. The coarse pass itself can
terminate early using pixel-size LOD (see below).


**Pixel-size LOD termination**

The tree hierarchy is a natural LOD structure. When a node is small enough on
screen that it covers less than one pixel, there's no point descending further.
The test is:

  node_size_in_pixels = node_world_size / (ray_distance * tan(half_fov_per_pixel))
  if node_size_in_pixels < 1.0: stop

This means distant geometry is rendered at coarser tree levels, visiting far
fewer nodes. For a tree64, each level is a 4x reduction, so stopping one level
early means visiting 64x fewer children for that subtree.

For this to produce reasonable visuals, non-leaf nodes need to store an
aggregate value (average color, dominant material, opacity). This is analogous
to texture mipmaps. The tree hierarchy already exists — we'd just be decorating
internal nodes with color data computed during tree construction. Aokana uses a
density threshold: "if the number of non-empty voxels is >= 2, create a new
voxel" with the average color.

dubiousconst282 specifically notes this as a missing feature that would combine
well with the beam pre-pass. It's probably the single most impactful
optimization for rendering large worlds where most geometry is distant.


**Shared-memory stack (compute shader)**

VoxelRT stores the traversal stack in GPU shared memory rather than registers
or local memory:

  uint gs_stack[64][11]  // 64 threads, 11 entries each

This yielded ~9% speedup because GPU "local arrays" that don't fit in registers
get spilled to global memory (very slow), while shared memory is explicitly
addressed on-chip memory. The tree64's shallow depth (≤11 entries) makes this
practical.

The catch: this requires a compute shader. Fragment shaders have no access to
shared memory (workgroup storage). Our current architecture uses a fragment
shader, so adopting this optimization would require switching to a compute shader
for the ray tracing pass. The pipeline change is:
  - fullscreen triangle draw → compute dispatch (width/8 × height/8 workgroups)
  - frag_coord → gl_GlobalInvocationID
  - output → imageStore to a storage image
  - push constants work the same way

rust-gpu supports compute shaders, so this is feasible. Beyond the stack
optimization, compute shaders give access to subgroup operations and explicit
workgroup control. For a serious ray tracer, this is the standard approach.

The VoxelRT guide also notes that tracking only node indices in the stack (not
full node payloads) and reloading node data from the buffer as needed is "much
faster" — the smaller stack entries mean less shared memory pressure and better
occupancy.


**Now for the path tracing side.**

The traversal improvements above make each individual ray faster. The following
optimizations reduce the number of rays needed or extract more information per
ray.


**Temporal accumulation**

When the camera is stationary, each frame produces 1 new sample per pixel.
Instead of displaying a noisy single sample, accumulate into a running average.
After N frames you have an N-spp image at zero additional per-frame cost:

  result = (frameCount * history + current) / (frameCount + 1)

Reset the counter when the camera moves (compare view matrices). This is
trivially simple: one extra texture (the history buffer), one frame counter, and
a blend operation. It's the single highest-value path tracing improvement
because it gives you arbitrarily clean images when the camera is still, and
most interaction time is spent looking at a scene, not moving.

For smoother camera motion, temporal reprojection maps history samples to their
new screen positions using the previous frame's depth and view matrix. This
avoids the "flash to noise" on every camera move. But even without reprojection,
accumulation alone is a massive improvement.

To get proper anti-aliasing during accumulation, apply a subpixel jitter to the
ray origin each frame. The R2 low-discrepancy sequence distributes samples
quasi-uniformly across the pixel:

  jitter = fract(vec2(0.7548776662 * frame, 0.5698402910 * frame))
(https://nelari.us/post/pathtracer_devlog/)
(https://cwyman.org/code/dxrTutors/tutors/Tutor6/tutorial06.md.html)


**Tone mapping and gamma correction**

The current shader clamps radiance to [0,1] and writes it directly. This loses
all highlight detail and produces incorrect brightness. The minimum fix is:

1. Apply tone mapping to compress HDR values into displayable range
2. Apply gamma correction (linear → sRGB) for correct display

The Narkowicz ACES approximation is 5 lines and the industry default:

  vec3 aces(vec3 x) {
    float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
  }

Followed by gamma: pow(color, 1.0/2.2). This must be the absolute last step —
all accumulation and blending must happen in linear space.
(https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/)
(https://64.github.io/tonemapping/)


**Cosine-weighted hemisphere sampling**

Our current Lambertian scatter direction is normal + random_unit_vector. This
produces a cosine-weighted distribution (which is actually correct for
Lambertian) but only by accident, and the derivation of the correct estimator
weight depends on it. An explicit cosine-weighted sample (Malley's method) is:

  r = sqrt(u1)
  theta = 2π * u2
  local = (r*cos(theta), r*sin(theta), sqrt(1 - u1))
  world = transform local from normal-aligned frame to world

The PDF is cos(θ)/π, which perfectly cancels the cos(θ) and 1/π terms in the
Lambertian BRDF, simplifying the estimator to just throughput *= albedo. No
division by PDF needed. This is both more numerically stable and gives roughly
2x variance reduction over uniform hemisphere sampling.
(https://ameye.dev/notes/sampling-the-hemisphere/)


**Russian roulette**

Instead of a hard 8-bounce cap, probabilistically terminate paths based on
throughput. After some minimum bounces (typically 3):

  float q = max(0.05, 1 - luminance(throughput));
  if random() < q: break;
  throughput /= (1 - q);

This is unbiased (the compensation factor preserves expected value) and saves
work on dark paths that contribute little to the image while allowing bright
paths to continue beyond 8 bounces. The minimum termination probability (5%)
prevents any path from running forever.
(https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Russian_Roulette_and_Splitting)


**Next event estimation (direct light sampling)**

Currently, a bounce only gathers light if the random scatter direction happens
to point at the sky. For a small, bright sun, almost no random rays hit it, so
direct illumination is extremely noisy (mostly black with rare firefly bright
spots). Next event estimation fixes this by explicitly sampling the light source
at each bounce:

  1. Sample a direction toward the sun disk (cone sampling)
  2. Cast a shadow ray to check visibility
  3. If visible: add throughput × sun_radiance × BRDF × cos(θ) / lightPDF
  4. Continue the normal BSDF-sampled bounce for indirect light

For a distant sun, the cone sample is:

  cos_theta = 1 - u1 * (1 - cos_theta_max)
  sin_theta = sqrt(1 - cos_theta^2)
  phi = 2π * u2
  pdf = 1 / (2π * (1 - cos_theta_max))

This dramatically reduces variance for direct illumination — the most important
lighting component. The shadow ray is cheap (just a binary hit/no-hit
traversal, no bouncing). When combining NEE with BSDF sampling, you should
either exclude emissive contribution from the BSDF-sampled indirect bounce (to
avoid double counting), or use multiple importance sampling (MIS) to weight
both contributions.
(https://developer.nvidia.com/blog/conquering-noisy-images-in-ray-tracing-with-next-event-estimation/)
(https://nelari.us/post/pathtracer_devlog/)


**Blue noise and stratified sampling**

Standard white noise RNG produces clumpy sample distributions. Blue noise
spreads samples more evenly across both space and time. The practical approach:

  1. Bake a 128×128 blue noise texture (or use NVIDIA's precomputed STBN textures)
  2. Index it by pixel position (tiled across screen)
  3. Offset by R2 low-discrepancy sequence per frame:
       blueNoise = fract(texture[pixel % 128] + vec2(0.7548 * frame, 0.5698 * frame))

Use the blue noise values as seeds for the first bounce (most visually
important) and fall back to white noise for deeper bounces. The visual
improvement is noticeable: noise looks like a uniform dither rather than clumps,
and spatial denoisers perform much better on blue noise input.
(https://developer.nvidia.com/blog/rendering-in-real-time-with-spatiotemporal-blue-noise-textures-part-1/)
(https://github.com/NVIDIA-RTX/STBN)


**Denoising**

For production-quality 1-spp rendering during camera motion, you need a
denoiser. The standard is SVGF (Spatiotemporal Variance-Guided Filtering),
which combines temporal reprojection with an edge-preserving spatial filter:

  1. Output G-buffer data (normals, depth, albedo) from the primary ray
  2. Reproject previous frame's denoised output to current frame using depth + motion
  3. Blend with adaptive alpha (high alpha near disocclusions)
  4. Run 3-5 iterations of a-trous wavelet bilateral filter with edge-stopping
     weights based on depth, normal, and luminance similarity

SVGF is what Quake II RTX uses (the A-SVGF variant). It requires ~5-6 shader
passes and a G-buffer, so it's significant infrastructure. But even a simple
spatial bilateral filter without temporal reprojection helps a lot at 1 spp.
(https://alain.xyz/blog/ray-tracing-denoising)
(https://research.nvidia.com/publication/2017-07_spatiotemporal-variance-guided-filtering-real-time-reconstruction-path-traced)


**GPU architecture considerations**

Currently we run the ray tracer as a fragment shader. This works but has
limitations:

Fragment shaders have no access to shared memory (workgroup storage), which
prevents the shared-memory stack optimization. They also lack subgroup
operations and workgroup barriers on some platforms.

A compute shader would allow shared-memory stacks, explicit workgroup sizing
(8×8 = 64 threads is standard for ray tracing), and subgroup operations. The
pipeline change is straightforward: replace the fullscreen triangle with a
compute dispatch, use imageStore for output, keep push constants as-is.

The main argument for fragment shaders is simplicity and automatic 2×2 quad
derivatives (useful for LOD via dFdx/dFdy). But for a ray tracer that does its
own LOD via tree-level termination, this isn't needed.

Warp divergence is the fundamental GPU performance problem for ray tracing:
rays within a warp (32 threads on NVIDIA, 64 on AMD) hit different geometry
and take different code paths. Studies show ~20 active threads per warp on
average. Mitigations include keeping the traversal loop simple (fewer branches),
minimizing register pressure (to maximize occupancy), and storing the stack in
shared memory rather than registers.
(https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/)


**Summary of optimizations, roughly ordered by impact and feasibility**

Traversal side:
- DDA within tree64 nodes (front-to-back ordering, early first-hit termination)
- Ancestor memoization (avoid redundant root-to-leaf descent) — ~2x measured
- Bitmask coalescing (skip groups of empty cells) — ~21% measured
- Ray-octant mirroring (simplify intersection math) — ~11% measured
- Beam optimization pre-pass (skip empty space before surfaces)
- Pixel-size LOD termination (coarser levels for distant geometry)
- Shared-memory stack via compute shader — ~9% measured
- IEEE754 mantissa trick (bit manipulation instead of float math for cell indexing)

Path tracing side:
- Temporal accumulation (free convergence when camera is still)
- Tone mapping + gamma correction (correct display of HDR values)
- Cosine-weighted hemisphere sampling (~2x variance reduction)
- Russian roulette (probabilistic termination, saves work on dark paths)
- Next event estimation (explicit sun sampling, huge noise reduction)
- Blue noise / stratified sampling (better sample distribution)
- Spatial/temporal denoising (SVGF or simpler bilateral filter)

The traversal changes are architecturally deeper — replacing the core traversal
loop — but the individual path tracing improvements are mostly independent and
can be added incrementally. The VoxelRT benchmarks suggest the traversal
improvements compound to roughly 2.5x overall (16,903 → 6,358 cycles/ray),
which translates directly to framerate since traversal is the bottleneck.
