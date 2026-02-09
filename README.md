A ray-traced voxel engine using rust-gpu.

Renders sparse voxel worlds via path tracing on the GPU. Shaders are written in
Rust, compiled to SPIR-V at build time by rust-gpu, and executed through wgpu.
The current scene is a Menger sponge at 64^3 resolution with Lambertian
scattering and up to 8 bounces.

**Crates** (all under crates/):
- host: desktop application -- window, input, camera, GPU setup, render loop
- shader: SPIR-V fragment/vertex shaders (no_std, runs on GPU)
- world: sparse octree and GPU ray traversal (dual-compile: CPU std, GPU no_std)
- prim: math primitives, ray, AABB, PRNG (dual-compile)
- shared: CPU-GPU shared types (ShaderConstants via push constants)

**Docs** (docs/):
- requirements.md: project constraints and goals
- impl.md: implementation strategy (64-trees, chunk streaming, DAG compression)
- notes.md: formatting guidelines for documentation
- notes/voxel-memory-structures.md: research on voxel data structures
- notes/voxel-disk-storage.md: research on persistent voxel storage

The docs directory is useful context for understanding design decisions. impl.md
is the best starting point, it references the research notes for background.

Requires nightly Rust (see rust-toolchain.toml).
