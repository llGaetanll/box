#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::rand;
use crate::RandState;
use crate::Range;
use crate::Vec3;
use crate::F;
use crate::PI;

pub trait Vec3Ext {
    fn rand_range(state: &mut RandState, range: Range<F>) -> Self;

    fn rand_unit(state: &mut RandState) -> Self;

    fn rand_hemisphere(state: &mut RandState, normal: Vec3) -> Self;

    fn rand_unit_disk(state: &mut RandState) -> Self;

    fn near_zero(&self) -> bool;
}

impl Vec3Ext for Vec3 {
    fn rand_range(state: &mut RandState, range: Range<F>) -> Self {
        let x = rand::rand_f_range(state, range);
        let y = rand::rand_f_range(state, range);
        let z = rand::rand_f_range(state, range);

        Self::new(x, y, z)
    }

    fn rand_unit(state: &mut RandState) -> Self {
        let theta = 2.0 * PI * rand::rand_f(state);
        let z = rand::rand_f_range(state, Range::new(-1.0, 1.0));
        let r = (1.0 - z * z).sqrt();

        Vec3::new(r * theta.cos(), r * theta.sin(), z)
    }

    fn rand_hemisphere(state: &mut RandState, normal: Vec3) -> Self {
        let v = Self::rand_unit(state);

        if v.dot(normal) > 0. {
            v
        } else {
            -v
        }
    }

    fn rand_unit_disk(state: &mut RandState) -> Self {
        let theta = 2.0 * PI * rand::rand_f(state);
        let r = rand::rand_f(state).sqrt();

        Vec3::new(r * theta.cos(), r * theta.sin(), 0.0)
    }

    fn near_zero(&self) -> bool {
        const ERR: F = 1e-8;

        (self.x.abs() < ERR) && (self.y.abs() < ERR) && (self.z.abs() < ERR)
    }
}
