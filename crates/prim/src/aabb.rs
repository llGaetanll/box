use core::ops::Add;
use core::ops::Index;

use crate::Point3;
use crate::Range;
use crate::Ray;
use crate::Vec3;
use crate::F;

/// An Axis-Aligned Bounding Box defined using the slab method.
#[derive(Clone, Default)]
#[repr(C)]
pub struct Aabb {
    x: Range<F>,
    y: Range<F>,
    z: Range<F>,
}

impl Aabb {
    pub fn empty() -> Self {
        Self {
            x: Range { start: 0., end: 0. },
            y: Range { start: 0., end: 0. },
            z: Range { start: 0., end: 0. },
        }
    }

    pub const fn from_slabs(x: Range<F>, y: Range<F>, z: Range<F>) -> Self {
        let mut aabb = Self { x, y, z };

        aabb.pad_to_minimums();

        aabb
    }

    pub const fn from_points(p1: Point3, p2: Point3) -> Self {
        let (x1, y1, z1) = (p1.x, p1.y, p1.z);
        let (x2, y2, z2) = (p2.x, p2.y, p2.z);

        let (x_lo, x_hi) = (x1.min(x2), x1.max(x2));
        let (y_lo, y_hi) = (y1.min(y2), y1.max(y2));
        let (z_lo, z_hi) = (z1.min(z2), z1.max(z2));

        Self::from_slabs(
            Range::new(x_lo, x_hi),
            Range::new(y_lo, y_hi),
            Range::new(z_lo, z_hi),
        )
    }

    pub fn from_aabbs(b1: &Aabb, b2: &Aabb) -> Self {
        let Aabb {
            x: x1,
            y: y1,
            z: z1,
        } = b1;

        let Aabb {
            x: x2,
            y: y2,
            z: z2,
        } = b2;

        let (x_lo, x_hi) = (x1.start.min(x2.start), x1.end.max(x2.end));
        let (y_lo, y_hi) = (y1.start.min(y2.start), y1.end.max(y2.end));
        let (z_lo, z_hi) = (z1.start.min(z2.start), z1.end.max(z2.end));

        Self::from_slabs(
            Range::new(x_lo, x_hi),
            Range::new(y_lo, y_hi),
            Range::new(z_lo, z_hi),
        )
    }

    pub fn x(&self) -> &Range<F> {
        &self.x
    }

    pub fn y(&self) -> &Range<F> {
        &self.y
    }

    pub fn z(&self) -> &Range<F> {
        &self.z
    }

    pub fn hit(&self, ray: &Ray, t_int: &mut Range<F>) -> bool {
        let orig = ray.orig();
        let dir = ray.dir();

        for axis in 0..3 {
            let Range { start: lo, end: hi } = self[axis];

            let t0 = (lo - orig[axis]) / dir[axis];
            let t1 = (hi - orig[axis]) / dir[axis];

            if t0 < t1 {
                if t0 > t_int.start {
                    t_int.start = t0
                };

                if t1 < t_int.end {
                    t_int.end = t1
                };
            } else {
                if t1 > t_int.start {
                    t_int.start = t1
                };

                if t0 < t_int.end {
                    t_int.end = t0
                };
            }

            if t_int.end <= t_int.start {
                return false;
            }
        }

        true
    }

    pub fn union(&self, other: &Self) -> Aabb {
        Self::from_aabbs(self, other)
    }

    pub fn union_mut(&mut self, other: &Self) {
        let bbox = Self::from_aabbs(self, other);

        *self = bbox;
    }

    pub fn longest_axis(&self) -> usize {
        let mut axis = 0;
        let mut max = 0.;

        for i in 0..3 {
            let Range { start, end } = self[i];

            if end - start > max {
                max = end - start;
                axis = i;
            }
        }

        axis
    }

    /// Adjust the Aabb so that no side is narrower than some delta, padding if necessary
    const fn pad_to_minimums(&mut self) {
        const DELTA: F = 0.0001;

        const fn size(range: &Range<F>) -> F {
            (range.end - range.start).max(0.)
        }

        if size(&self.x) < DELTA {
            pad_range(&mut self.x, DELTA)
        }

        if size(&self.y) < DELTA {
            pad_range(&mut self.y, DELTA)
        }

        if size(&self.z) < DELTA {
            pad_range(&mut self.z, DELTA)
        }
    }
}

impl Index<usize> for Aabb {
    type Output = Range<F>;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            n => panic!("Axis \"{n}\" out of bounds"),
        }
    }
}

impl Add<Vec3> for Aabb {
    type Output = Self;

    fn add(self, rhs: Vec3) -> Self::Output {
        let (dx, dy, dz) = (rhs.x, rhs.y, rhs.z);

        let Aabb {
            x: Range {
                start: x_lo,
                end: x_hi,
            },
            y: Range {
                start: y_lo,
                end: y_hi,
            },
            z: Range {
                start: z_lo,
                end: z_hi,
            },
        } = self;

        Self::from_slabs(
            Range::new(x_lo + dx, x_hi + dx),
            Range::new(y_lo + dy, y_hi + dy),
            Range::new(z_lo + dz, z_hi + dz),
        )
    }
}

const fn pad_range(range: &mut Range<F>, delta: F) {
    let pad = delta / 2.;

    range.start -= pad;
    range.end += pad;
}
