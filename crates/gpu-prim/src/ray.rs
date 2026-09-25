use crate::Point3;
use crate::Vec3;
use crate::F;

#[derive(Default, Clone, Copy)]
#[repr(C)]
pub struct Ray {
    orig: Point3,
    dir: Vec3,
    time: F,
}

impl Ray {
    pub const fn new(orig: Point3, dir: Vec3, time: F) -> Self {
        Self { orig, dir, time }
    }

    pub fn orig(&self) -> Point3 {
        self.orig
    }

    pub fn dir(&self) -> Vec3 {
        self.dir
    }

    pub fn time(&self) -> F {
        self.time
    }

    pub fn at(&self, t: F) -> Point3 {
        self.orig + t * self.dir
    }
}
