use ouroboros::self_referencing;
use winit::window::Window;

#[self_referencing(pub_extras)]
pub struct WindowSurface {
    pub window: Box<Window>,
    #[borrows(window)]
    #[covariant]
    pub surface: wgpu::Surface<'this>,
}
