Performance improvement plan. Each step is a single commit on the `perf`
branch. Run the live renderer before and after each step for visual comparison.
Run `cargo run -- bench menger_sponge` after each step and record results.

**Step 0 — End-to-end path tracing tests** (done)

Tests live in `shader/tests/trace.rs`. The core path tracing loop was extracted
into `shader::trace_color` so it can be called from CPU tests with a fixed RNG
seed. Run `cargo test -p shader` to verify.

Tests:
- Exact color snapshots for a voxel hit from all 6 axis directions
- Front-to-back ordering with two voxels of different materials
- Ray missing the tree returns exact sky color
- Ray originating inside the tree
- 2-level deep tree (16^3 grid)
- Determinism: same seed produces identical output


**Phase 1 — Path tracing quick wins (shader-only)**

Each of these is a small change to `shader/src/lib.rs`. No host or data
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
Run `cargo test -p shader` after every sub-step to verify snapshot colors
are unchanged.

Step 3a: DDA within tree64 nodes
- Replace unordered bitmask iteration with Amanatides-Woo stepping through
  each 4x4x4 grid in ray order
- Early termination on first hit (the ray visits cells front-to-back)
- Test: all snapshot tests must pass. Visual output should be identical.
- Benchmark after — this is the biggest expected improvement

Commit: "Replace bitmask iteration with DDA traversal"

Step 3b: Ray-octant mirroring
- Reflect ray into canonical octant at traversal start
- XOR cell indices with mirror mask during lookup
- Eliminates direction-sign branches in intersection math
- Test: all snapshot tests must pass. Visual output identical.
- Benchmark after (~11% expected)

Commit: "Add ray-octant mirroring"

Step 3c: Ancestor memoization
- Maintain stack of node indices indexed by tree level
- XOR old/new positions to detect which level changed
- Recover node index from stack instead of descending from root
- Test: all snapshot tests must pass. Visual output identical.
- Benchmark after (~2x expected)

Commit: "Add ancestor memoization"

Step 3d: Bitmask coalescing
- When a 2x2x2 sub-block of the 4x4x4 grid is entirely empty, skip it
  in one DDA step instead of four
- Test: all snapshot tests must pass. Visual output identical.
- Benchmark after (~21% expected)

Commit: "Add bitmask coalescing for empty cell skipping"
