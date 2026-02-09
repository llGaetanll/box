Research on how to store voxel worlds on disk, with a focus on formats that
support streaming (loading chunks on demand), fast load times, and larger-than-
RAM worlds.


**Region Files**

The most proven approach to on-disk voxel storage is the region file, used by
Minecraft and several engines that followed. The idea is to group chunks
spatially into "regions" (e.g. 32x32x32 chunks per region) and store each
region as a single file on disk.

A region file has a fixed-size header containing an offset table: one entry per
chunk, giving the byte offset and size of that chunk's data within the file.
This makes loading a single chunk a matter of one header lookup plus one
sequential read, with no need to scan or parse the rest of the file. Chunks
that haven't been generated yet simply have a zero entry in the table.
(https://minecraft.wiki/w/Region_file_format)

The Voxel Tools engine uses a similar design (VXR format) with sector-based
allocation: the file is divided into fixed-size sectors, and each chunk occupies
one or more contiguous sectors. When a chunk is modified and its compressed size
changes, it can be rewritten into the same sectors if it still fits, or
relocated to new sectors at the end of the file. This avoids the need to shift
file contents on every save, which is critical for frequent incremental writes.
(https://voxel-tools.readthedocs.io/en/latest/specs/region_format_v3/)

Region files map naturally to a filesystem: the region coordinates become the
filename (e.g. r.0.3.-1.vxr), so locating the file for any world-space
coordinate is trivial arithmetic. For larger-than-RAM worlds, only region files
near the camera need to be open at any time.


**Chunk Serialization**

Within a region file, each chunk needs to be serialized to bytes. There are two
main strategies:

Linearized tree: serialize the 64-tree (or octree) directly. Depth-first
linearization is natural here since it only requires a stack, and the node
encoding is already compact (12 bytes per node for a 64-tree). The serialized
form is essentially the same as the in-memory flattened representation, which
means deserialization can be nearly zero-copy: read the bytes into a buffer and
use them directly. This is the simplest approach and preserves the tree's
spatial structure for immediate GPU upload.
(https://eisenwave.github.io/voxel-compression-docs/svo/svo.html)

Flat array with RLE: ignore the tree structure on disk and instead serialize
the chunk's voxels as a flat array in some spatial traversal order, then
run-length encode it. This exploits the fact that voxel data is highly regular:
large runs of empty space, solid regions of a single material, etc. The
traversal order matters for RLE effectiveness. Morton order (Z-curves)
preserves 3D spatial locality reasonably well and is trivial to compute with
bit interleaving. Hilbert curves preserve locality better but are more complex
to compute; Eisenwave's benchmarks found Hilbert and Morton both improved
bytewise RLE compression by about 60% compared to naive row-major order, with
Hilbert only marginally better than Morton.
(https://eisenwave.github.io/voxel-compression-docs/rle/rle.html)

The choice depends on the pipeline. If chunks go straight from disk to GPU
buffer, linearized tree is ideal (minimal processing). If chunks go through a
CPU-side representation first (e.g. for physics, editing), flat array + RLE can
be more compact on disk.


**Palette Compression**

With at most 256 voxel types, each voxel needs at most 8 bits. But most chunks
won't use all 256 types. Palette compression stores a per-chunk palette (a
small array mapping local indices to global voxel type IDs) and then encodes
each voxel as an index into that palette, using only as many bits as needed. A
chunk using 4 materials needs only 2 bits per voxel. A chunk that's entirely
one material needs zero bits for the voxel data, just the palette entry.
(https://voxel.wiki/wiki/palette-compression/)

Minecraft uses this approach in its chunk format: each 16x16x16 section has its
own palette and a packed array of indices. The bit width of each index is
ceil(log2(palette_size)).
(https://minecraft.wiki/w/Chunk_format)

Palette compression composes well with both tree and RLE serialization. For a
linearized tree, leaf nodes would store palette indices instead of raw voxel
type IDs. For RLE, runs are over palette indices, which are smaller and have
fewer distinct values, improving run lengths.


**General-Purpose Compression on Top**

After domain-specific encoding (tree linearization, RLE, palette), a general-
purpose compressor can squeeze out remaining redundancy. Benchmarks from Zeux's
voxel terrain engine on a 1-billion-voxel world show the effect clearly:

- RLE alone: 73 MB (0.07 bytes/voxel)
- RLE + LZ4:  50 MB (0.05 bytes/voxel)
- RLE + zstd: 38 MB (0.04 bytes/voxel)
(https://zeux.io/2017/03/27/voxel-terrain-storage/)

LZ4 is the standard choice for real-time applications. It decompresses at
multiple GB/s per core with minimal memory overhead. zstd compresses better
(roughly 25-30% smaller than LZ4 at default settings) but decompresses at
about 1 GB/s, still fast enough for chunk streaming but not as headroom-rich
as LZ4. For a world that needs to load fast, LZ4 is the safer default; zstd
makes sense for archival or when disk space is tighter.
(https://github.com/lz4/lz4)
(https://github.com/facebook/zstd)

NVIDIA's nvCOMP library provides GPU-accelerated LZ4 and zstd decompression,
which could allow decompressing chunks directly on the GPU without CPU
involvement, though this adds complexity.
(https://developer.nvidia.com/nvcomp)


**Dirty Chunk Tracking and Incremental Saves**

For a dynamic world, only modified chunks need to be written back to disk. A
simple dirty bit per chunk suffices: when a voxel is edited, mark that chunk
dirty. On save (periodic autosave or on exit), only dirty chunks are serialized
and written to their region file. The sector-based region format handles this
well since individual chunks can be rewritten without touching the rest of the
file.

For crash safety, the simplest approach is to write modified chunks to a
temporary file and atomically rename it over the region file, or to use a
write-ahead log. Minecraft's region format sidesteps this by having chunks
occupy independent sectors, so a partial write only corrupts one chunk rather
than the whole region.


**Out-of-Core Construction**

For worlds that are too large to build in memory, the out-of-core SVO builder
by Baert et al. demonstrates a practical approach: voxels are first sorted in
Morton order on disk, then the octree is built bottom-up by reading 8 voxels at
a time, constructing parent nodes, and flushing each level's buffer to disk
when full. This requires only O(depth) memory regardless of world size. The
same principle applies to 64-trees: sort in Morton order, read 64 at a time,
build parents.
(https://github.com/Forceflow/ooc_svo_builder)
(https://www.forceflow.be/2012/07/24/out-of-core-construction-of-sparse-voxel-octrees/)
