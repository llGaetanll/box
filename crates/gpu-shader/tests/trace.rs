use gpu_prim::{Ray, Vec3};
use gpu_shader::{sky_color, trace_color};

fn pack_tree(flat: &[u8], dims: [u32; 3]) -> (Vec<u32>, Vec<u32>, u32, u32) {
    let tree = voxel_tree64::Tree64::new((flat, dims));
    let root = tree.root_state();
    let nodes_u32: Vec<u32> = bytemuck::cast_slice(&tree.nodes).to_vec();
    let data_u32: Vec<u32> = tree
        .data
        .chunks(4)
        .map(|chunk| {
            let mut word = 0u32;
            for (i, &b) in chunk.iter().enumerate() {
                word |= (b as u32) << (i * 8);
            }
            word
        })
        .collect();
    (nodes_u32, data_u32, root.num_levels as u32, root.index)
}

/// Build a tree64 with a single voxel of the given value at position (x,y,z)
/// within a 4x4x4 grid.
/// Index into a flat 4x4x4 array.
fn idx4(x: u32, y: u32, z: u32) -> usize {
    (x + y * 4 + z * 16) as usize
}

fn single_voxel_tree(x: u32, y: u32, z: u32, value: u8) -> (Vec<u32>, Vec<u32>, u32, u32) {
    let mut flat = [0u8; 64];
    flat[idx4(x, y, z)] = value;
    pack_tree(&flat, [4, 4, 4])
}

#[test]
fn ray_hits_single_voxel_and_sees_sky() {
    // Place a stone voxel (value=3) at (1,1,1) in a 4^3 grid.
    // Shoot a ray from outside, straight at it along +X.
    // The ray should hit the voxel, scatter, and (with this seed)
    // eventually escape to the sky, producing a nonzero color.
    let (nodes, data, depth, root) = single_voxel_tree(1, 1, 1, 3);
    let ray = Ray::new(Vec3::new(-5.0, 1.5, 1.5), Vec3::new(1.0, 0.0, 0.0), 0.0);
    let mut state: gpu_prim::RandState = 42;

    let color = trace_color(&nodes, &data, depth, root, &ray, &mut state);

    // Should have some color contribution (hit stone -> sky)
    assert!(
        color.x > 0.0 || color.y > 0.0 || color.z > 0.0,
        "expected nonzero color after hitting voxel, got {color:?}"
    );
}

#[test]
fn ray_misses_everything_returns_sky() {
    // Same scene but ray goes in -X direction, away from the voxel.
    let (nodes, data, depth, root) = single_voxel_tree(1, 1, 1, 3);
    let ray = Ray::new(Vec3::new(-5.0, 1.5, 1.5), Vec3::new(-1.0, 0.0, 0.0), 0.0);
    let mut state: gpu_prim::RandState = 42;

    let color = trace_color(&nodes, &data, depth, root, &ray, &mut state);
    let expected_sky = sky_color(ray.dir());

    assert!((color.x - expected_sky.x).abs() < 1e-6);
    assert!((color.y - expected_sky.y).abs() < 1e-6);
    assert!((color.z - expected_sky.z).abs() < 1e-6);
}

#[test]
fn deterministic_with_same_seed() {
    let (nodes, data, depth, root) = single_voxel_tree(2, 0, 2, 1);
    let ray = Ray::new(Vec3::new(-5.0, 0.5, 2.5), Vec3::new(1.0, 0.0, 0.0), 0.0);

    let mut state1: gpu_prim::RandState = 123;
    let color1 = trace_color(&nodes, &data, depth, root, &ray, &mut state1);

    let mut state2: gpu_prim::RandState = 123;
    let color2 = trace_color(&nodes, &data, depth, root, &ray, &mut state2);

    assert_eq!(color1, color2, "same seed should produce identical results");
}

fn assert_color_eq(actual: Vec3, expected: Vec3, label: &str) {
    assert!(
        (actual.x - expected.x).abs() < 1e-6
            && (actual.y - expected.y).abs() < 1e-6
            && (actual.z - expected.z).abs() < 1e-6,
        "{label}: expected {expected:?}, got {actual:?}"
    );
}

#[test]
fn snapshot_all_axes() {
    // Stone voxel (value=3) at (1,1,1) in a 4^3 grid, hit from all 6 directions.
    // Exact colors captured with seed=42. Any change to traversal or shading
    // logic will break these.
    let (nodes, data, depth, root) = single_voxel_tree(1, 1, 1, 3);
    let cases: &[(&str, Vec3, Vec3, Vec3)] = &[
        ("plus_x",  Vec3::new(-5.0, 1.5, 1.5), Vec3::new(1.0, 0.0, 0.0),
         Vec3::new(0.45202476, 0.51121485, 0.6)),
        ("minus_x", Vec3::new(9.0, 1.5, 1.5),  Vec3::new(-1.0, 0.0, 0.0),
         Vec3::new(0.45202476, 0.51121485, 0.6)),
        ("plus_y",  Vec3::new(1.5, -5.0, 1.5), Vec3::new(0.0, 1.0, 0.0),
         Vec3::new(0.53742415, 0.5624545, 0.6)),
        ("minus_y", Vec3::new(1.5, 9.0, 1.5),  Vec3::new(0.0, -1.0, 0.0),
         Vec3::new(0.3625759, 0.45754555, 0.6)),
        ("plus_z",  Vec3::new(1.5, 1.5, -5.0), Vec3::new(0.0, 0.0, 1.0),
         Vec3::new(0.5718726, 0.58312356, 0.6)),
        ("minus_z", Vec3::new(1.5, 1.5, 9.0),  Vec3::new(0.0, 0.0, -1.0),
         Vec3::new(0.32812744, 0.4368765, 0.6)),
    ];
    for (name, orig, dir, expected) in cases {
        let ray = Ray::new(*orig, *dir, 0.0);
        let mut state: gpu_prim::RandState = 42;
        let color = trace_color(&nodes, &data, depth, root, &ray, &mut state);
        assert_color_eq(color, *expected, name);
    }
}

#[test]
fn two_voxels_front_to_back() {
    // Grass (value=1) at (1,1,1), red (value=4) at (3,1,1).
    // Ray from -X hits grass first; ray from +X hits red first.
    // The colors differ because the materials differ.
    let mut flat = [0u8; 64];
    flat[idx4(1, 1, 1)] = 1;
    flat[idx4(3, 1, 1)] = 4;
    let (n, d, dep, r) = pack_tree(&flat, [4, 4, 4]);

    let ray_fwd = Ray::new(Vec3::new(-5.0, 1.5, 1.5), Vec3::new(1.0, 0.0, 0.0), 0.0);
    let mut state: gpu_prim::RandState = 42;
    let near = trace_color(&n, &d, dep, r, &ray_fwd, &mut state);
    assert_color_eq(near, Vec3::new(0.22601238, 0.5964173, 0.2), "near (grass)");

    let ray_bwd = Ray::new(Vec3::new(9.0, 1.5, 1.5), Vec3::new(-1.0, 0.0, 0.0), 0.0);
    let mut state: gpu_prim::RandState = 42;
    let far = trace_color(&n, &d, dep, r, &ray_bwd, &mut state);
    assert_color_eq(far, Vec3::new(0.6026997, 0.17040496, 0.2), "far (red)");
}

#[test]
fn ray_from_inside_tree() {
    // Red voxel (value=4) at (3,1,1). Ray originates inside the grid at (2,1.5,1.5)
    // pointing +X, should hit the voxel.
    let mut flat = [0u8; 64];
    flat[idx4(3, 1, 1)] = 4;
    let (n, d, dep, r) = pack_tree(&flat, [4, 4, 4]);

    let ray = Ray::new(Vec3::new(2.0, 1.5, 1.5), Vec3::new(1.0, 0.0, 0.0), 0.0);
    let mut state: gpu_prim::RandState = 42;
    let color = trace_color(&n, &d, dep, r, &ray, &mut state);
    assert_color_eq(color, Vec3::new(0.6026997, 0.17040496, 0.2), "inside tree");
}

#[test]
fn deep_tree_two_levels() {
    // 16^3 grid (2 tree levels). Wood voxel (value=2) at (5,5,5).
    let mut flat = vec![0u8; 16 * 16 * 16];
    flat[5 + 5 * 16 + 5 * 16 * 16] = 2;
    let (n, d, dep, r) = pack_tree(&flat, [16, 16, 16]);

    let ray = Ray::new(Vec3::new(-5.0, 5.5, 5.5), Vec3::new(1.0, 0.0, 0.0), 0.0);
    let mut state: gpu_prim::RandState = 42;
    let color = trace_color(&n, &d, dep, r, &ray, &mut state);
    assert_color_eq(color, Vec3::new(0.41435602, 0.29820865, 0.15), "deep tree");
}
