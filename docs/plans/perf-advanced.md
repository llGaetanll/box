Advanced performance improvement plan. These are larger changes that require
more design work than the quick wins in perf-easy.md.

**Key constraint**: the world will be editable at runtime. Any optimization
that assumes a static world must either support incremental updates at
interactive rates or be skippable when edits are happening. Do not trade
edit performance for render performance.


**Phase 1 — Rendering pipeline**

Step 1a: Compute shader migration
- Move path tracing from fragment shader to a compute shader
- Enables workgroup shared memory, flexible dispatch, decouples tracing
  from screen resolution
- Prerequisite for most advanced techniques (denoising, ray queues)
- The accumulation buffer (already a storage buffer) transfers directly
- No impact on edit performance — purely a rendering change

Step 1b: Importance-sampled direct lighting
- Sample sun/sky direction explicitly each bounce instead of hoping a
  random scatter hits the light
- Biggest single-sample quality improvement (less noise per frame)
- Combine with the existing indirect bounce via multiple importance
  sampling (MIS)
- No impact on edit performance

Step 1c: Temporal denoising (SVGF-style)
- Add G-buffer outputs: depth, normal, albedo
- Reproject previous frame using motion vectors (camera delta already
  tracked via frame_count reset)
- Spatiotemporal variance-guided filter to denoise 1-4 spp
- Must handle disocclusion from camera movement and world edits — when
  a voxel changes, invalidate the history for affected pixels
- Edit constraint: reprojection naturally handles small edits (new
  geometry = disoccluded = no history = falls back to current sample).
  Large edits may cause a brief noisy frame, which is acceptable.

Step 1d: Better RNG (blue noise / Sobol)
- Replace PCG-based PRNG with blue noise texture or Sobol sequence
- Better stratification = less variance at low sample counts
- Complements denoising — fewer samples needed for same quality
- No impact on edit performance


**Phase 2 — Data structure optimizations**

Step 2a: LOD via early termination
- Stop descending the tree64 when a node subtends less than ~1 pixel
  on screen
- Store an average color per internal node for distant rendering
- Edit constraint: when a voxel is edited, propagate the average color
  up to ancestor nodes. This is O(tree_depth) per edit — fast enough
  for interactive single-voxel edits. Bulk edits may defer propagation
  to end-of-batch.
- Requires: screen-space size test in traversal (needs pixel footprint
  from push constants)

Step 2b: Chunk streaming
- Split the world into chunks (already defined in the voxel-store crate:
  64^3 per chunk). Only upload visible/nearby chunks to GPU.
- Maintain a chunk atlas in a large GPU buffer, LRU-evict distant chunks
- Edit constraint: edited chunks are re-uploaded. The hot path (single
  chunk rebuild + upload) must stay under 1ms.
- Requires: frustum/distance culling on CPU, atlas management

Step 2c: DAG compression (deduplication)
- Deduplicate identical tree64 subtrees to reduce memory and improve
  GPU cache hit rates
- The Menger sponge has massive repetition; real terrain will have less
  but still benefits
- Edit constraint: this is the hardest one. Editing a shared subtree
  requires copy-on-write (unshare the path from root to edited node).
  Options:
  - Rebuild DAG periodically in background, edit the uncompressed tree
    in the foreground
  - Use COW with a dirty flag — compress on idle, decompress on edit
  - Skip DAG entirely if edit rate is high (just use tree64 as-is)
- May want to defer this until chunk streaming is in place, so DAG
  compression operates per-chunk


**Phase 3 — GPU architecture**

Step 3a: Persistent threads with ray queues
- Instead of one ray per thread, maintain a queue of active rays
- Threads pull from the queue, trace one step, push back if not done
- Better occupancy for incoherent rays (bounced rays diverge)
- Requires compute shader (Phase 1 step 1a)
- No impact on edit performance

Step 3b: Shared memory BVH caching
- Cache frequently-accessed tree64 nodes in workgroup shared memory
- Rays in the same tile tend to traverse similar nodes
- Requires compute shader
- No impact on edit performance


**Priority order**

For maximum impact with the edit constraint respected:
1. Compute shader migration (1a) — unlocks everything else
2. Direct lighting (1b) — biggest visual quality win
3. LOD (2a) — biggest performance win for larger worlds
4. Chunk streaming (2b) — required to scale beyond single chunk
5. Denoising (1c) — makes low spp look great
6. DAG compression (2c) — memory savings, but complex with edits
7. Better RNG (1d), ray queues (3a), shared memory (3b) — polish
