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
cargo run -- bench    # run all benchmarks in bench/configs/
cargo run -- bench <name>  # run a specific benchmark (e.g. menger_sponge)
cargo run -- chart    # generate SVG charts from benchmark results
```

**Live renderer controls**: WASD to move, Space/C for up/down, mouse to look
around, Escape or Q to quit.

**Benchmarking**: configs live in bench/configs/ as TOML files. Each defines a
scene, frame count, and a camera path (position + look-at spline control
points). Run a benchmark to record frame timings to bench/results/, then use
`chart` to produce an SVG in bench/charts/.

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
