use glam::Vec3;
use serde::Deserialize;
use serde::Serialize;

use crate::spline::CatmullRomSpline;

/// A camera path defined by position and look-at splines.
///
/// Interpolates camera position and look-at target along Catmull-Rom splines,
/// allowing smooth camera motion through a scene.
#[derive(Serialize, Deserialize, Clone)]
pub struct CameraPath {
    position: CatmullRomSpline,
    look_at: CatmullRomSpline,
    frame_count: u32,
}

/// The result of evaluating a camera path at a point in time.
pub struct CameraFrame {
    /// Camera position in world space.
    pub position: Vec3,
    /// Point the camera is looking at in world space.
    pub look_at: Vec3,
}

impl CameraFrame {
    /// Compute the camera's forward direction (normalized).
    pub fn direction(&self) -> Vec3 {
        (self.look_at - self.position).normalize()
    }

    /// Compute the camera's up vector, given a world up reference.
    pub fn up(&self, world_up: Vec3) -> Vec3 {
        let forward = self.direction();
        let right = forward.cross(world_up).normalize();
        right.cross(forward)
    }
}

impl CameraPath {
    /// Create a new camera path from position and look-at control points.
    ///
    /// Both splines must have at least 4 control points.
    /// Frame count must be at least 1.
    pub fn new(position_points: Vec<Vec3>, look_at_points: Vec<Vec3>, frame_count: u32) -> Self {
        Self {
            position: CatmullRomSpline::new(position_points),
            look_at: CatmullRomSpline::new(look_at_points),
            frame_count,
        }
    }

    /// Number of frames in this camera path.
    pub fn frame_count(&self) -> u32 {
        self.frame_count
    }

    /// Evaluate the camera path at a specific frame.
    pub fn evaluate_frame(&self, frame: u32) -> CameraFrame {
        let t = if self.frame_count <= 1 {
            0.0
        } else {
            (frame as f32 / (self.frame_count - 1) as f32).clamp(0.0, 1.0)
        };
        CameraFrame {
            position: self.position.evaluate(t),
            look_at: self.look_at.evaluate(t),
        }
    }

    /// Compute the normalized t value for a given frame index.
    pub fn frame_t(&self, frame: u32) -> f32 {
        if self.frame_count <= 1 {
            0.0
        } else {
            (frame as f32 / (self.frame_count - 1) as f32).clamp(0.0, 1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Vec3, b: Vec3, epsilon: f32) -> bool {
        (a - b).length() < epsilon
    }

    #[test]
    fn evaluate_at_start() {
        let position_points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::new(0.0, 0.0, 15.0),
        ];
        let look_at_points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
        ];
        let path = CameraPath::new(position_points, look_at_points, 100);

        let frame = path.evaluate_frame(0);
        assert!(
            approx_eq(frame.position, Vec3::new(0.0, 0.0, 5.0), 1e-6),
            "Expected position (0, 0, 5), got {:?}",
            frame.position
        );
        assert!(
            approx_eq(frame.look_at, Vec3::new(0.0, 1.0, 0.0), 1e-6),
            "Expected look_at (0, 1, 0), got {:?}",
            frame.look_at
        );
    }

    #[test]
    fn evaluate_at_end() {
        let position_points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::new(0.0, 0.0, 15.0),
        ];
        let look_at_points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
        ];
        let path = CameraPath::new(position_points, look_at_points, 100);

        let frame = path.evaluate_frame(99);
        assert!(
            approx_eq(frame.position, Vec3::new(0.0, 0.0, 10.0), 1e-6),
            "Expected position (0, 0, 10), got {:?}",
            frame.position
        );
        assert!(
            approx_eq(frame.look_at, Vec3::new(0.0, 2.0, 0.0), 1e-6),
            "Expected look_at (0, 2, 0), got {:?}",
            frame.look_at
        );
    }

    #[test]
    fn direction_and_up() {
        let frame = CameraFrame {
            position: Vec3::new(0.0, 0.0, 5.0),
            look_at: Vec3::new(0.0, 0.0, 0.0),
        };

        let dir = frame.direction();
        assert!(
            approx_eq(dir, Vec3::new(0.0, 0.0, -1.0), 1e-6),
            "Expected direction (0, 0, -1), got {:?}",
            dir
        );

        let up = frame.up(Vec3::Y);
        assert!(
            approx_eq(up, Vec3::new(0.0, 1.0, 0.0), 1e-6),
            "Expected up (0, 1, 0), got {:?}",
            up
        );
    }

    #[test]
    fn frame_t_values() {
        let position_points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];
        let look_at_points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, 3.0),
        ];
        let path = CameraPath::new(position_points, look_at_points, 5);

        assert!((path.frame_t(0) - 0.0).abs() < 1e-6);
        assert!((path.frame_t(1) - 0.25).abs() < 1e-6);
        assert!((path.frame_t(2) - 0.5).abs() < 1e-6);
        assert!((path.frame_t(3) - 0.75).abs() < 1e-6);
        assert!((path.frame_t(4) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn single_frame_path() {
        let position_points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];
        let look_at_points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, 3.0),
        ];
        let path = CameraPath::new(position_points, look_at_points, 1);

        assert!((path.frame_t(0) - 0.0).abs() < 1e-6);
    }
}
