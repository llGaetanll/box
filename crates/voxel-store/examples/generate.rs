use voxel_store::CHUNK_SIZE;
use voxel_store::ChunkPos;
use voxel_store::ChunkStore;

fn is_menger(mut x: u32, mut y: u32, mut z: u32, size: u32) -> bool {
    let mut s = size;
    while s > 1 {
        s /= 3;
        let cx = (x / s) % 3;
        let cy = (y / s) % 3;
        let cz = (z / s) % 3;
        let center_count = u32::from(cx == 1) + u32::from(cy == 1) + u32::from(cz == 1);
        if center_count >= 2 {
            return false;
        }
        x %= s;
        y %= s;
        z %= s;
    }
    true
}

fn main() {
    let size = CHUNK_SIZE;
    let sponge_size = 27_u32;
    let offset = (size - sponge_size) / 2;

    // Build flat voxel array
    let mut voxels = vec![0_u8; (size * size * size) as usize];
    let mut solid_count = 0_u32;

    for z in 0..sponge_size {
        for y in 0..sponge_size {
            for x in 0..sponge_size {
                if is_menger(x, y, z, sponge_size) {
                    let gx = (x + offset) as usize;
                    let gy = (y + offset) as usize;
                    let gz = (z + offset) as usize;
                    voxels[gx + gy * size as usize + gz * size as usize * size as usize] = 3;
                    solid_count += 1;
                }
            }
        }
    }

    println!(
        "Generated {0}^3 Menger sponge in {1}^3 chunk",
        sponge_size, size
    );
    println!("Solid voxels: {solid_count}");

    // Build tree
    let tree = voxel_tree64::Tree64::new((&voxels[..], [size, size, size]));
    println!(
        "Tree64: {} nodes, {} data entries",
        tree.nodes.len(),
        tree.data.len()
    );

    // Save to region file
    let world_dir = "world";
    let mut store = ChunkStore::new(world_dir).expect("failed to create chunk store");

    let pos: ChunkPos = [0, 0, 0];
    store.save_chunk(pos, &tree).expect("failed to save chunk");

    let region_path = std::path::Path::new(world_dir)
        .join(voxel_store::region_filename(voxel_store::chunk_to_region(pos)));
    let file_size = std::fs::metadata(&region_path)
        .expect("region file missing")
        .len();
    println!("Saved to {} ({} bytes)", region_path.display(), file_size);

    // Load back and verify
    let loaded = store
        .load_chunk(pos)
        .expect("failed to load chunk")
        .expect("chunk not found after saving");

    let mut mismatches = 0_u32;
    for x in 0..size {
        for y in 0..size {
            for z in 0..size {
                let original = tree.get_value_at([x, y, z]);
                let roundtrip = loaded.get_value_at([x, y, z]);
                if original != roundtrip {
                    mismatches += 1;
                }
            }
        }
    }

    if mismatches == 0 {
        println!("Loaded and verified: all voxels match");
    } else {
        eprintln!("VERIFICATION FAILED: {mismatches} mismatches");
        std::process::exit(1);
    }
}
