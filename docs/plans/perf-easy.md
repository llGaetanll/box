Performance improvement plan. Each step is a single commit on the `perf`
branch. Run the live renderer before and after each step for visual comparison.
Run `cargo run -- bench menger_sponge` after each step and record results.

**Step 0 — End-to-end path tracing tests** (done)

Tests live in `gpu-shader/tests/trace.rs`. The core path tracing loop was extracted
into `shader::trace_color` so it can be called from CPU tests with a fixed RNG
seed. Run `cargo test -p gpu-shader` to verify.

Tests:
- Exact color snapshots for a voxel hit from all 6 axis directions
- Front-to-back ordering with two voxels of different materials
- Ray missing the tree returns exact sky color
- Ray originating inside the tree
- 2-level deep tree (16^3 grid)
- Determinism: same seed produces identical output


**Phase 1 — Path tracing quick wins (shader-only)**

Each of these is a small change to `gpu-shader/src/lib.rs`. No host or data
structure changes. Visual correctness is verified by running the live renderer.

Step 1a: Tone mapping + gamma correction
- Add ACES tonemapping after accumulation
- Apply gamma: pow(color, 1/2.2)
- Test: visual — highlights should compress instead of clipping, overall
  brightness should look more natural
- Benchmark after

Commit: "Add tone mapping and gamma correction"

Step 1b: Cosine-weighted hemisphere sampling
- Replace `normal + random_unit_vector` with Malley's method
- Simplifies estimator to throughput *= albedo (no PDF division needed)
- Test: visual — less variance per sample, slightly different brightness
  distribution. Existing traversal tests still pass (sampling is independent).
- Benchmark after

Commit: "Use cosine-weighted hemisphere sampling"

Step 1c: Russian roulette
- After bounce 3, probabilistically terminate based on throughput luminance
- Compensate surviving paths by dividing throughput by survival probability
- Remove the hard MAX_BOUNCES=8 cap (or raise it to something high like 64)
- Test: visual — should look identical to before (unbiased), but dark corners
  converge faster because we stop wasting work on absorbed paths
- Benchmark after

Commit: "Add Russian roulette path termination"


**Phase 2 — Temporal accumulation (host + shader)**

Step 2a: Frame counter in push constants
- Add `frame_count: u32` to ShaderConstants
- Host tracks frame count, increments each frame, resets to 0 when camera moves
- Compare previous cam_pos/cam_dir/cam_vup to detect movement
- Test: `cargo test -p shared` (Pod/Zeroable derive still works), run live
  renderer and verify frame_count resets on movement (add a debug log)
- Benchmark after (should be no-op performance-wise)

Commit: "Add frame counter to push constants"

Step 2b: Accumulation buffer
- Host: create a storage texture (Rgba32Float, same size as window)
- Add it to the bind group as a read-write storage texture
- Shader: read history, blend with current sample using
  mix(history, current, 1.0 / (frame_count + 1)), write result back
- On frame_count == 0, just write the current sample (reset)
- Test: visual — when camera is still, image should converge to a clean
  render over a few seconds. When camera moves, should reset cleanly
  without ghosting.
- Benchmark after

Commit: "Add temporal accumulation"

Step 2c: Subpixel jitter
- Use R2 low-discrepancy sequence indexed by frame_count to offset ray
  within the pixel: jitter = fract(vec2(0.7548776662, 0.5698402910) * frame)
- Test: visual — edges should anti-alias as frames accumulate. Without
  jitter, accumulated image converges but edges stay aliased.
- Benchmark after

Commit: "Add subpixel jitter for anti-aliased accumulation"


**Phase 3 — DDA traversal rewrite**

This replaces the core traversal loop. The step 0 tests are the safety net.
Run `cargo test -p gpu-shader` after every sub-step to verify snapshot colors
are unchanged.

Step 3a: DDA within tree64 nodes
- Replace unordered bitmask iteration with Amanatides-Woo stepping through
  each 4x4x4 grid in ray order
- Early termination on first hit (the ray visits cells front-to-back)
- Test: all snapshot tests must pass. Visual output should be identical.
- Benchmark after — this is the biggest expected improvement

Commit: "Replace bitmask iteration with DDA traversal"

Step 3b: Ray-octant mirroring — **skipped**
- Implemented and benchmarked; ~5% regression. The XOR on every bitmask
  lookup and un-mirroring hit positions cost more than the branch removal
  saved. GPU warps tend to have coherent ray directions so the branches
  were already well-predicted. Reverted.

Step 3c: Ancestor memoization — **skipped**
- Depends on a restart-from-root traversal pattern that the DDA rewrite
  in 3a eliminated. DDA already maintains per-level stack state, so
  ancestor memoization has nothing to recover. Not applicable.

Step 3d: Bitmask coalescing — **skipped**
- The DDA in 3a already skips empty cells by checking the bitmask per
  step. Coalescing 2x2x2 blocks adds bookkeeping that isn't justified
  at the current 64^3 scene size. May revisit for larger worlds.


**Re-measured** (2026-09-25)

The numbers recorded while working through this plan were taken with vsync
on and without waiting on the GPU, in a window whose size the window manager
picked. Some frames measured queue submission only (76µs at 3800x2075), and
the rest were pulled toward multiples of the refresh interval. They are not
comparable to each other.

Each commit's shader, prim and world crates were rebuilt against the current
harness and run with `bench --headless` at 1920x1080 on an RTX 5090, three
runs each. Median frame time (p50):

- 2dbea7a (baseline): 8.6ms
- 6a43069 tone mapping: 8.5ms, no change
- 2f080e6 cosine sampling: 8.5ms, no change
- 5ffa9e1 Russian roulette: 5.5ms, -35%. The only real win
- 1b37f32, e240146, 55e6976 accumulation, seed, jitter: 5.5ms, no change
- 8ac4a3e DDA (a91bd5c does not compile to SPIR-V on its own): 5.4ms, about
  -2% on the median and -6% on the mean

DDA was expected to be the biggest improvement and was a small one. Step 3b's
5% regression was measured with the old harness and is within its noise, so
the reason for skipping it does not hold. It was never committed, so it
cannot be re-measured without redoing it.

Headless runs of one binary are not always stable either: 5ffa9e1 gave
medians of 5.5ms in some runs and 8-9ms in others, with a deterministic
shader. Most likely GPU clocks. Compare runs repeated at least three times.


**Traversal, second attempt** (2026-09-25, commit 7e176c9)

The DDA of step 3a kept a stack of 24 entries of 14 words each, indexed
by the current depth. That is 1.3KB per ray, which the GPU cannot keep in
registers and spills to local memory, and every step read and wrote it.
That, not the order children were visited in, was where the time went.

The rewrite tracks the ray as an integer voxel coordinate. The cell at any
level is a two-bit slice of it, the node containing it is the bits above,
and the level a step leaves is the highest bit the step changed. The only
per-level state is a stack of node indices. Headless 1920x1080 on an RTX
5090, three runs each, p50: 5.4ms -> 0.40ms. Same picture to within
rounding, checked with `bench --headless --save-frames` before and after.

The remaining ideas from the notes (octant mirroring, bitmask
coalescing, beam pre-pass, LOD termination) are still untried on the new
traversal. At 0.4ms per 1080p frame the tracer is no longer what limits
live mode; the next cost is the history read and write that reprojection
added (0.67ms with it on).
