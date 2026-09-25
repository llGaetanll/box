Region-file-based chunk persistence for voxel worlds. Chunks are grouped
spatially into regions (16x16x16 = 4096 chunks per region), each stored as a
single file on disk. Within a region file, chunk data is laid out in fixed-size
sectors (4096 bytes), so loading a chunk is one header lookup plus one
sequential read. Chunk coordinates are signed 64-bit integers, supporting worlds
that extend in any direction from the origin.

Currently only the "hot" path is implemented: chunks are stored as serialized
64-trees (`Tree64<u8>::serialize`), which can be loaded and used directly with
minimal processing. The format tag in each header entry reserves space for a
future "cold" path (palette-compressed, RLE'd, zstd'd) for distant chunks.

**Key components:**

- `RegionFile` -- low-level handle to a single region file. Manages the binary
  format: an 8-byte file header (magic `VXRG`, version, reserved), a 4096-entry
  header table (12 bytes per entry: sector offset, byte size, format tag), and
  sector-aligned chunk data. All binary parsing uses explicit
  `to_le_bytes`/`from_le_bytes`.

- `ChunkStore` -- high-level interface managing a directory of region files.
  Translates chunk coordinates to region files and slot indices, lazily opens
  region files on demand, and bridges between `Tree64<u8>` and raw bytes.

- `ChunkPos` -- `[i64; 3]` chunk coordinate. Mapped to region coordinates via
  `div_euclid` and to slot indices via `rem_euclid`, which correctly handles
  negative coordinates.

**Note on sector allocation:** Like tree64's append-only modification, sector
allocation uses a simple high-water mark. When a chunk is overwritten and the
new data fits in the old sectors, they are reused. Otherwise new sectors are
allocated at the end of the file and the old ones become dead space. There is no
free list or compaction. This keeps the implementation simple and is fine for
typical usage patterns where region files are periodically rebuilt or replaced
during chunk promotion/demotion.
