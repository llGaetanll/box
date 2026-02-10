use std::error::Error;
use std::time::Instant;

use futures::executor::block_on;
use glam::Quat;
use glam::Vec3;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::ElementState;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::event_loop::ControlFlow;
use winit::event_loop::EventLoop;
use winit::keyboard::Key;
use winit::keyboard::NamedKey;
use winit::window::WindowAttributes;
use winit::window::WindowId;

use crate::gpu::GpuContext;
use crate::window_surface::WindowSurface;
use crate::window_surface::WindowSurfaceBuilder;

const CHUNK_SIZE: u32 = 64; // 4^3 = 64 per axis

/// Tracks which movement keys are currently held
#[derive(Default)]
struct KeysHeld {
    w: bool,
    a: bool,
    s: bool,
    d: bool,
    space: bool,
    c: bool,
}

pub struct LiveApp {
    gpu: Option<GpuContext>,
    window_surface: Option<WindowSurface>,
    config: Option<wgpu::SurfaceConfiguration>,
    render_pipeline: Option<wgpu::RenderPipeline>,
    bind_group: Option<wgpu::BindGroup>,
    close_requested: bool,
    start: Instant,
    cursor_x: f32,
    cursor_y: f32,
    cam_pos: Vec3,
    yaw: f32,
    pitch: f32,
    keys_held: KeysHeld,
    last_cursor_x: f32,
    last_cursor_y: f32,
    last_frame: Instant,
    tree_depth: u32,
    tree_root: u32,
}

impl LiveApp {
    pub fn new() -> Self {
        let cam_pos = Vec3::new(55.0, 45.0, 55.0);
        // Compute initial yaw/pitch to look toward the sponge center
        let target = Vec3::new(32.0, 32.0, 32.0);
        let delta = target - cam_pos;
        let yaw = delta.x.atan2(-delta.z);
        let pitch = (-delta.y / delta.length()).asin();

        Self {
            gpu: None,
            window_surface: None,
            config: None,
            render_pipeline: None,
            bind_group: None,
            close_requested: false,
            start: Instant::now(),
            cursor_x: 0.0,
            cursor_y: 0.0,
            cam_pos,
            yaw,
            pitch,
            keys_held: KeysHeld::default(),
            last_cursor_x: 0.0,
            last_cursor_y: 0.0,
            last_frame: Instant::now(),
            tree_depth: 0,
            tree_root: 0,
        }
    }

    fn cam_orientation(&self) -> Quat {
        Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(self.pitch)
    }

    fn update_camera(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;

        let first_frame = self.last_cursor_x == 0.0 && self.last_cursor_y == 0.0;
        let mouse_dx = if first_frame {
            0.0
        } else {
            self.cursor_x - self.last_cursor_x
        };
        let mouse_dy = if first_frame {
            0.0
        } else {
            self.cursor_y - self.last_cursor_y
        };
        self.last_cursor_x = self.cursor_x;
        self.last_cursor_y = self.cursor_y;

        let sensitivity = 0.003;
        let half_pi = std::f32::consts::FRAC_PI_2 - 0.01;

        self.yaw -= mouse_dx * sensitivity;
        self.pitch -= mouse_dy * sensitivity;
        self.pitch = self.pitch.clamp(-half_pi, half_pi);

        let orientation = self.cam_orientation();
        let forward = orientation * Vec3::NEG_Z;
        let right = orientation * Vec3::X;

        let forward_horizontal = Vec3::new(forward.x, 0.0, forward.z).normalize_or_zero();
        let right_horizontal = Vec3::new(right.x, 0.0, right.z).normalize_or_zero();

        let speed = 5.0 * dt;

        if self.keys_held.w {
            self.cam_pos += forward_horizontal * speed;
        }
        if self.keys_held.s {
            self.cam_pos -= forward_horizontal * speed;
        }
        if self.keys_held.a {
            self.cam_pos -= right_horizontal * speed;
        }
        if self.keys_held.d {
            self.cam_pos += right_horizontal * speed;
        }
        if self.keys_held.space {
            self.cam_pos.y += speed;
        }
        if self.keys_held.c {
            self.cam_pos.y -= speed;
        }
    }

    fn cam_dir(&self) -> Vec3 {
        self.cam_orientation() * Vec3::NEG_Z
    }

    fn cam_vup(&self) -> Vec3 {
        self.cam_orientation() * Vec3::Y
    }

    fn is_menger(mut x: u32, mut y: u32, mut z: u32, size: u32) -> bool {
        let mut s = size;
        while s > 1 {
            s /= 3;
            let cx = (x / s) % 3;
            let cy = (y / s) % 3;
            let cz = (z / s) % 3;
            let center_count = u32::from(cx == 1) + u32::from(cy == 1) + u32::from(cz == 1);
            if center_count >= 2 {
                return false;
            }
            x %= s;
            y %= s;
            z %= s;
        }
        true
    }

    pub fn build_tree64() -> (Vec<u32>, Vec<u32>, u32, u32) {
        let sponge_size = 27u32;
        let offset = (CHUNK_SIZE - sponge_size) / 2;

        let mut flat = vec![0u8; (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize];
        for z in 0..sponge_size {
            for y in 0..sponge_size {
                for x in 0..sponge_size {
                    if Self::is_menger(x, y, z, sponge_size) {
                        let idx = (x + offset) as usize
                            + (y + offset) as usize * CHUNK_SIZE as usize
                            + (z + offset) as usize * CHUNK_SIZE as usize * CHUNK_SIZE as usize;
                        flat[idx] = 3;
                    }
                }
            }
        }

        let tree = tree64::Tree64::new((&flat[..], [CHUNK_SIZE; 3]));
        let root_state = tree.root_state();

        let nodes_u32: Vec<u32> = bytemuck::cast_slice(&tree.nodes).to_vec();

        // Pack u8 data into u32s (little-endian, 4 bytes per u32)
        let data_u32: Vec<u32> = tree
            .data
            .chunks(4)
            .map(|chunk| {
                let mut word = 0u32;
                for (i, &byte) in chunk.iter().enumerate() {
                    word |= (byte as u32) << (i * 8);
                }
                word
            })
            .collect();

        log::debug!(
            "Tree64: {} nodes, {} data bytes, {} levels, root={}",
            tree.nodes.len(),
            tree.data.len(),
            root_state.num_levels,
            root_state.index
        );

        (
            nodes_u32,
            data_u32,
            root_state.num_levels as u32,
            root_state.index,
        )
    }

    async fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<(), Box<dyn Error>> {
        let window_attributes = WindowAttributes::default()
            .with_title("Voxels")
            .with_inner_size(LogicalSize::new(800.0, 600.0));
        let window_box = event_loop.create_window(window_attributes)?;

        let instance = GpuContext::create_instance();

        let window_surface = WindowSurfaceBuilder {
            window: Box::new(window_box),
            surface_builder: |window| {
                instance
                    .create_surface(window)
                    .expect("Failed to create surface")
            },
        }
        .build();

        let window = window_surface.borrow_window();
        window.set_cursor_visible(false);
        // Try confined first, fall back to locked
        if window
            .set_cursor_grab(winit::window::CursorGrabMode::Confined)
            .is_err()
        {
            let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Locked);
        }
        let window_size = window_surface.borrow_window().inner_size();
        let surface = window_surface.borrow_surface();

        let gpu = GpuContext::new(instance, Some(surface)).await?;

        // Build tree64 and upload to GPU
        let (nodes_u32, data_u32, tree_depth, tree_root) = Self::build_tree64();
        let node_buffer = gpu.create_storage_buffer(&nodes_u32);
        let data_buffer = gpu.create_storage_buffer(&data_u32);
        let bind_group = gpu.create_bind_group(&node_buffer, &data_buffer);

        let swapchain_format = surface.get_capabilities(&gpu.adapter).formats[0];

        let render_pipeline = gpu.create_pipeline(swapchain_format, "main_fs");

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: window_size.width,
            height: window_size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: Default::default(),
        };
        surface.configure(&gpu.device, &config);

        self.gpu = Some(gpu);
        self.window_surface = Some(window_surface);
        self.config = Some(config);
        self.render_pipeline = Some(render_pipeline);
        self.bind_group = Some(bind_group);
        self.start = Instant::now();
        self.tree_depth = tree_depth;
        self.tree_root = tree_root;
        Ok(())
    }

    fn render(&mut self) {
        self.update_camera();

        let window_surface = match &self.window_surface {
            Some(ws) => ws,
            None => return,
        };
        let gpu = match &self.gpu {
            Some(gpu) => gpu,
            None => return,
        };

        let window = window_surface.borrow_window();
        let current_size = window.inner_size();
        let surface = window_surface.borrow_surface();

        let frame = match surface.get_current_texture() {
            Ok(frame) => frame,
            Err(e) => {
                eprintln!("Error getting next frame: {e:?}");
                return;
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let cam_dir = self.cam_dir();
        let cam_vup = self.cam_vup();
        let push_constants = shared::ShaderConstants {
            width: current_size.width,
            height: current_size.height,
            time: self.start.elapsed().as_secs_f32(),
            cursor_x: self.cursor_x,
            cursor_y: self.cursor_y,
            cam_pos: self.cam_pos.into(),
            cam_dir: cam_dir.into(),
            cam_vup: cam_vup.into(),
            tree_depth: self.tree_depth,
            tree_root: self.tree_root,
        };

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(self.render_pipeline.as_ref().unwrap());
            rpass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            rpass.set_push_constants(
                wgpu::ShaderStages::VERTEX_FRAGMENT,
                0,
                bytemuck::bytes_of(&push_constants),
            );
            rpass.draw(0..3, 0..1);
        }

        gpu.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}

impl ApplicationHandler for LiveApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Err(e) = block_on(self.init(event_loop)) {
            eprintln!("Initialization error: {e}");
            event_loop.exit();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => self.close_requested = true,
            WindowEvent::Resized(new_size) => {
                if let Some(config) = self.config.as_mut() {
                    config.width = new_size.width;
                    config.height = new_size.height;
                    if let Some(ws) = &self.window_surface {
                        let surface = ws.borrow_surface();
                        if let Some(gpu) = &self.gpu {
                            surface.configure(&gpu.device, config);
                        }
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_x = position.x as f32;
                self.cursor_y = position.y as f32;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                match &event.logical_key {
                    Key::Named(NamedKey::Escape) => {
                        if pressed {
                            self.close_requested = true;
                        }
                    }
                    Key::Named(NamedKey::Space) => self.keys_held.space = pressed,
                    Key::Character(c) => match c.as_str() {
                        "w" | "W" => self.keys_held.w = pressed,
                        "a" | "A" => self.keys_held.a = pressed,
                        "s" | "S" => self.keys_held.s = pressed,
                        "d" | "D" => self.keys_held.d = pressed,
                        "c" | "C" => self.keys_held.c = pressed,
                        "q" | "Q" => {
                            if pressed {
                                self.close_requested = true;
                            }
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }

        if self.close_requested {
            event_loop.exit();
        } else if let Some(ws) = &self.window_surface {
            ws.borrow_window().request_redraw();
        }
        event_loop.set_control_flow(ControlFlow::Poll);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.close_requested {
            event_loop.exit();
        } else if let Some(ws) = &self.window_surface {
            ws.borrow_window().request_redraw();
        }
        event_loop.set_control_flow(ControlFlow::Poll);
    }
}

pub fn run_live() -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = LiveApp::new();
    event_loop.run_app(&mut app).map_err(Into::into)
}
