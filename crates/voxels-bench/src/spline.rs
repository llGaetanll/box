use glam::Vec3;
use serde::Deserialize;
use serde::Serialize;

/// A Catmull-Rom spline that interpolates through a series of control points.
///
/// Catmull-Rom splines pass through all control points (except the first and last,
/// which only influence the curve's tangents at the endpoints). The curve is C1
/// continuous (smooth first derivative) at all interior points.
#[derive(Serialize, Deserialize, Clone)]
pub struct CatmullRomSpline {
    points: Vec<Vec3>,
}

impl CatmullRomSpline {
    /// Create a new Catmull-Rom spline from control points.
    ///
    /// Requires at least 4 points. The curve will pass through points[1] to
    /// points[n-2], with points[0] and points[n-1] influencing the tangents
    /// at the endpoints.
    pub fn new(points: Vec<Vec3>) -> Self {
        debug_assert!(
            points.len() >= 4,
            "Catmull-Rom spline requires at least 4 control points"
        );
        Self { points }
    }

    /// Evaluate the spline at parameter t in [0, 1].
    ///
    /// Returns the interpolated position. At t=0, returns points[1].
    /// At t=1, returns points[n-2].
    pub fn evaluate(&self, t: f32) -> Vec3 {
        let n = self.points.len();
        // Number of segments is n - 3 (each segment needs 4 points)
        let num_segments = n - 3;

        // Clamp t to [0, 1]
        let t = t.clamp(0.0, 1.0);

        // Map t to segment index and local parameter
        let scaled = t * num_segments as f32;
        let segment = (scaled.floor() as usize).min(num_segments - 1);
        let local_t = scaled - segment as f32;

        // Get the 4 control points for this segment
        let p0 = self.points[segment];
        let p1 = self.points[segment + 1];
        let p2 = self.points[segment + 2];
        let p3 = self.points[segment + 3];

        catmull_rom(p0, p1, p2, p3, local_t)
    }
}

/// Evaluate Catmull-Rom basis for a single segment.
///
/// Given 4 control points and parameter t in [0, 1], returns the interpolated
/// point between p1 and p2. Points p0 and p3 influence the tangents.
fn catmull_rom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;

    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Vec3, b: Vec3, epsilon: f32) -> bool {
        (a - b).length() < epsilon
    }

    #[test]
    fn evaluate_at_zero_returns_second_point() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];
        let spline = CatmullRomSpline::new(points);

        let result = spline.evaluate(0.0);
        assert!(
            approx_eq(result, Vec3::new(1.0, 0.0, 0.0), 1e-6),
            "Expected (1, 0, 0), got {:?}",
            result
        );
    }

    #[test]
    fn evaluate_at_one_returns_second_to_last_point() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];
        let spline = CatmullRomSpline::new(points);

        let result = spline.evaluate(1.0);
        assert!(
            approx_eq(result, Vec3::new(2.0, 0.0, 0.0), 1e-6),
            "Expected (2, 0, 0), got {:?}",
            result
        );
    }

    #[test]
    fn straight_line_produces_straight_output() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];
        let spline = CatmullRomSpline::new(points);

        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let result = spline.evaluate(t);
            let expected_x = 1.0 + t;
            assert!(
                approx_eq(result, Vec3::new(expected_x, 0.0, 0.0), 1e-5),
                "At t={}, expected x={}, got {:?}",
                t,
                expected_x,
                result
            );
        }
    }

    #[test]
    fn midpoint_evaluation() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];
        let spline = CatmullRomSpline::new(points);

        let result = spline.evaluate(0.5);
        assert!(
            approx_eq(result, Vec3::new(1.5, 0.0, 0.0), 1e-6),
            "Expected (1.5, 0, 0), got {:?}",
            result
        );
    }

    #[test]
    fn multiple_segments() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
        ];
        let spline = CatmullRomSpline::new(points);

        let at_start = spline.evaluate(0.0);
        assert!(
            approx_eq(at_start, Vec3::new(1.0, 0.0, 0.0), 1e-6),
            "Expected (1, 0, 0) at t=0, got {:?}",
            at_start
        );

        let at_end = spline.evaluate(1.0);
        assert!(
            approx_eq(at_end, Vec3::new(3.0, 0.0, 0.0), 1e-6),
            "Expected (3, 0, 0) at t=1, got {:?}",
            at_end
        );

        let at_mid = spline.evaluate(0.5);
        assert!(
            approx_eq(at_mid, Vec3::new(2.0, 1.0, 0.0), 1e-6),
            "Expected (2, 1, 0) at t=0.5, got {:?}",
            at_mid
        );
    }

    #[test]
    fn clamps_out_of_range_t() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];
        let spline = CatmullRomSpline::new(points);

        let below = spline.evaluate(-0.5);
        let at_zero = spline.evaluate(0.0);
        assert!(
            approx_eq(below, at_zero, 1e-6),
            "t=-0.5 should clamp to t=0"
        );

        let above = spline.evaluate(1.5);
        let at_one = spline.evaluate(1.0);
        assert!(approx_eq(above, at_one, 1e-6), "t=1.5 should clamp to t=1");
    }
}
