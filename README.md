A ray-traced voxel engine using rust-gpu.

Renders sparse voxel worlds via path tracing on the GPU. Shaders are written in
Rust, compiled to SPIR-V at build time by rust-gpu, and executed through wgpu.
The current scene is a Menger sponge at 64^3 resolution with Lambertian
scattering and up to 8 bounces.

Requires nightly Rust (see rust-toolchain.toml).

**Usage**:
```
cargo run             # open the live renderer (default)
cargo run -- live     # same as above, explicitly
cargo run -- live --samples 4  # trace 4 paths per pixel per frame instead of 1
cargo run -- bench    # run all benchmarks in bench/configs/
cargo run -- bench <name>  # run a specific benchmark (e.g. menger_sponge)
cargo run -- bench --headless  # render offscreen, no window
cargo run -- bench --headless --save-frames  # also save frames as PPM images
cargo run -- chart    # generate SVG charts from benchmark results
cargo run -- stats    # print frame time percentiles per commit
```

**Live renderer controls**: WASD to move, Space/C for up/down, mouse to look
around, Escape or Q to quit.

**Benchmarking**: configs live in bench/configs/ as TOML files. Each defines a
scene, frame count, render size, paths per pixel (`samples`, default 1), and a
camera path (position + look-at spline control points). Run a benchmark to
record frame timings to bench/results/, then use `chart` to produce an SVG in
bench/charts/. Add `--headless` to render offscreen at exactly the config's
`width` and `height`, where a window may be resized by the window manager.
Frame times vary run to run, so repeat a benchmark three times and compare
medians. Results are filed under the current commit's SHA, so commit before
measuring or the runs of different code end up in one place.

`--save-frames` writes every 100th frame and the last one to bench/frames/ as
PPM files, which ImageMagick can convert and diff:

```
magick bench/frames/<sha>/menger_sponge-0200.ppm frame.png
magick compare -metric MAE before.png after.png null:
```

A change meant to be faster should draw the same picture; one meant to be
less noisy should draw a cleaner one.

**Rendering**: one path per pixel per frame, accumulated over frames. While the
camera is still, each pixel sums its own samples without limit. While it moves,
each pixel reprojects its first hit into the previous frame and blends into
the history found there, if the same voxel face was seen, up to a cap of 32
samples so the resampling blur does not build up.

**Crates** (all under crates/, named `<category>-<role>`):
- app-main: desktop application -- window, input, camera, GPU setup, render loop
- gpu-shader: SPIR-V fragment/vertex shaders (no_std, runs on GPU)
- gpu-wire: CPU-GPU shared types (ShaderConstants via push constants)
- gpu-prim: math primitives, ray, AABB, PRNG (dual-compile)
- voxel-octree: sparse octree and GPU ray traversal (dual-compile: CPU std, GPU no_std)
- voxel-tree64: 64-tree spatial data structure (4^3 branching factor octree variant)
- voxel-store: region-file-based chunk persistence
- util-bench: benchmarking framework with camera paths and chart generation

**Docs** (docs/):
- requirements.md: project constraints and goals
- impl.md: implementation strategy (64-trees, chunk streaming, DAG compression)
- notes.md: formatting guidelines for documentation
- notes/voxel-memory-structures.md: research on voxel data structures
- notes/voxel-disk-storage.md: research on persistent voxel storage

The docs directory is useful context for understanding design decisions. impl.md
is the best starting point, it references the research notes for background.
