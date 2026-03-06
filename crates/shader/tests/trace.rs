use prim::{Ray, Vec3};
use shader::{sky_color, trace_color};

/// Build a tree64 with a single voxel of the given value at position (x,y,z)
/// within a 4x4x4 grid. Returns (nodes_u32, data_u32, num_levels, root_index).
fn single_voxel_tree(x: u32, y: u32, z: u32, value: u8) -> (Vec<u32>, Vec<u32>, u32, u32) {
    let mut flat = [0u8; 64];
    flat[(x + y * 4 + z * 16) as usize] = value;
    let tree = tree64::Tree64::new((&flat[..], [4, 4, 4]));
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

#[test]
fn ray_hits_single_voxel_and_sees_sky() {
    // Place a stone voxel (value=3) at (1,1,1) in a 4^3 grid.
    // Shoot a ray from outside, straight at it along +X.
    // The ray should hit the voxel, scatter, and (with this seed)
    // eventually escape to the sky, producing a nonzero color.
    let (nodes, data, depth, root) = single_voxel_tree(1, 1, 1, 3);
    let ray = Ray::new(Vec3::new(-5.0, 1.5, 1.5), Vec3::new(1.0, 0.0, 0.0), 0.0);
    let mut state: prim::RandState = 42;

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
    let mut state: prim::RandState = 42;

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

    let mut state1: prim::RandState = 123;
    let color1 = trace_color(&nodes, &data, depth, root, &ray, &mut state1);

    let mut state2: prim::RandState = 123;
    let color2 = trace_color(&nodes, &data, depth, root, &ray, &mut state2);

    assert_eq!(color1, color2, "same seed should produce identical results");
}
