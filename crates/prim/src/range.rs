#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Range<T> {
    pub start: T,
    pub end: T,
}

impl<T> Range<T> {
    pub const fn new(start: T, end: T) -> Self {
        Self { start, end }
    }
}

impl<T: PartialOrd<T>> Range<T> {
    pub fn contains(&self, item: &T) -> bool {
        (self.start <= *item) && (*item < self.end)
    }
}
