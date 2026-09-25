use gpu_prim::Vec3;
use gpu_shader::Camera;

fn camera() -> Camera {
    let pos = Vec3::new(55.0, 45.0, 55.0);
    let dir = (Vec3::new(32.0, 32.0, 32.0) - pos).normalize();
    Camera::new(pos.into(), dir.into(), [0.0, 1.0, 0.0], 1920, 1080)
}

#[test]
fn projecting_a_point_on_a_pixel_ray_lands_on_that_pixel() {
    let cam = camera();
    for &(sx, sy) in &[
        (0.5, 0.5),
        (960.5, 540.5),
        (1919.5, 1079.5),
        (123.25, 987.75),
    ] {
        let ray = cam.ray(sx, sy);
        let p = ray.at(37.0);
        let (visible, x, y) = cam.project(p);
        assert!(visible, "({sx}, {sy}) should be in front of the camera");
        assert!((x - sx).abs() < 1e-2, "x: {x} vs {sx}");
        assert!((y - sy).abs() < 1e-2, "y: {y} vs {sy}");

        // A direction projects the same as any point along it
        let (visible, x, y) = cam.project_dir(ray.dir());
        assert!(visible);
        assert!((x - sx).abs() < 1e-2, "dir x: {x} vs {sx}");
        assert!((y - sy).abs() < 1e-2, "dir y: {y} vs {sy}");
    }
}

#[test]
fn points_behind_the_camera_are_not_visible() {
    let cam = camera();
    let ray = cam.ray(960.5, 540.5);
    let (visible, _, _) = cam.project(cam.pos - ray.dir() * 10.0);
    assert!(!visible);
}

#[test]
fn points_off_screen_project_outside_the_image() {
    let cam = camera();
    // Well to the right of the rightmost pixel's ray
    let right = cam.ray(1919.5, 540.5);
    let further = cam.ray(2500.0, 540.5);
    let (visible, x, _) = cam.project(further.at(20.0));
    assert!(visible);
    assert!(x > 1920.0, "x: {x}");
    let (_, x_edge, _) = cam.project(right.at(20.0));
    assert!(x_edge < x);
}
