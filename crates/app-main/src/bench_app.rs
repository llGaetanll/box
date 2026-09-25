use std::error::Error;
use std::fs;
use std::io::BufWriter;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use chrono::DateTime;
use chrono::Utc;
use futures::executor::block_on;
use glam::Vec3;
use serde::Deserialize;
use util_bench::BenchmarkMetadata;
use util_bench::CameraPath;
use util_bench::FrameRecord;
use util_bench::GpuInfo;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::ElementState;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::event_loop::ControlFlow;
use winit::event_loop::EventLoop;
use winit::keyboard::Key;
use winit::keyboard::NamedKey;
use winit::window::WindowAttributes;
use winit::window::WindowId;

use crate::gpu::Accumulation;
use crate::gpu::GpuContext;
use crate::live_app::LiveApp;
use crate::window_surface::WindowSurface;
use crate::window_surface::WindowSurfaceBuilder;

/// Benchmark definition loaded from a TOML file.
#[derive(Deserialize)]
pub struct BenchmarkFile {
    pub scene: String,
    pub frame_count: u32,
    /// Size to render at. Headless runs use it exactly; a window asks for it, but
    /// the window manager may hand back something else.
    pub width: u32,
    pub height: u32,
    /// Paths traced per pixel per frame.
    #[serde(default = "default_samples")]
    pub samples: u32,
    pub position: Vec<[f32; 3]>,
    pub look_at: Vec<[f32; 3]>,
}

fn default_samples() -> u32 {
    1
}

impl BenchmarkFile {
    /// Load a benchmark definition from `bench/configs/<name>.toml`.
    pub fn load(name: &str) -> Result<Self, Box<dyn Error>> {
        let path = PathBuf::from("bench/configs").join(format!("{}.toml", name));
        let contents = fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        let def: BenchmarkFile = toml::from_str(&contents)
            .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))?;

        if def.frame_count < 1 {
            return Err(format!("Benchmark {} needs at least 1 frame", name).into());
        }
        if def.samples < 1 {
            return Err(format!("Benchmark {} needs at least 1 sample per pixel", name).into());
        }
        if def.position.len() < 4 {
            return Err(format!(
                "Benchmark {} needs at least 4 position points, got {}",
                name,
                def.position.len()
            )
            .into());
        }
        if def.look_at.len() < 4 {
            return Err(format!(
                "Benchmark {} needs at least 4 look_at points, got {}",
                name,
                def.look_at.len()
            )
            .into());
        }

        Ok(def)
    }

    pub fn camera_path(&self) -> CameraPath {
        CameraPath::new(
            self.position_points(),
            self.look_at_points(),
            self.frame_count,
        )
    }

    pub fn position_points(&self) -> Vec<Vec3> {
        self.position.iter().map(|&p| Vec3::from(p)).collect()
    }

    pub fn look_at_points(&self) -> Vec<Vec3> {
        self.look_at.iter().map(|&p| Vec3::from(p)).collect()
    }
}

/// Extract GPU info from a wgpu adapter.
fn gpu_info_from_adapter(adapter: &wgpu::Adapter) -> GpuInfo {
    let info = adapter.get_info();
    GpuInfo::new(info.name, info.driver, format!("{:?}", info.backend))
}

/// Git SHA baked in at build time via build.rs.
const GIT_SHA: &str = env!("GIT_SHA");

/// Format of the offscreen target a headless benchmark draws into. The same format
/// a window's swapchain gets on the Vulkan drivers this runs on, so the fragment
/// shader writes the same bytes either way.
const HEADLESS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

/// A queued benchmark with its name and definition.
struct QueuedBenchmark {
    name: String,
    def: BenchmarkFile,
}

/// A run through the queued benchmarks, independent of where the frames end up.
/// The window and headless modes both drive one of these, handing it a view to
/// draw each frame into.
struct Session {
    gpu: GpuContext,
    gpu_info: GpuInfo,
    format: wgpu::TextureFormat,
    render_pipeline: wgpu::RenderPipeline,
    node_buffer: wgpu::Buffer,
    data_buffer: wgpu::Buffer,
    /// The shader writes one accumulation entry per pixel, so the buffers have
    /// to match the size being drawn. Rebuilt whenever that size changes.
    accumulation: Accumulation,
    /// The camera of the previous frame of the current benchmark, which the
    /// history buffer holds the view from. `None` on its first frame.
    prev_camera: Option<(Vec3, Vec3, Vec3)>,
    tree_depth: u32,
    tree_root: u32,
    current: QueuedBenchmark,
    camera_path: CameraPath,
    frame_records: Vec<FrameRecord>,
    queue: Vec<QueuedBenchmark>,
    timestamp: DateTime<Utc>,
    headless: bool,
}

impl Session {
    fn new(
        gpu: GpuContext,
        format: wgpu::TextureFormat,
        mut benchmarks: Vec<QueuedBenchmark>,
        timestamp: DateTime<Utc>,
        headless: bool,
    ) -> Self {
        let current = benchmarks.remove(0);
        let gpu_info = gpu_info_from_adapter(&gpu.adapter);
        let render_pipeline = gpu.create_pipeline(format, &current.def.scene);

        // Build tree64 and upload to GPU
        let (nodes_u32, data_u32, tree_depth, tree_root) = LiveApp::build_tree64();
        let node_buffer = gpu.create_storage_buffer(&nodes_u32);
        let data_buffer = gpu.create_storage_buffer(&data_u32);

        let accumulation = gpu.create_accumulation(
            &node_buffer,
            &data_buffer,
            current.def.width,
            current.def.height,
        );

        log::info!(
            "Running benchmark '{}' (scene: {})",
            current.name,
            current.def.scene
        );

        Self {
            camera_path: current.def.camera_path(),
            gpu,
            gpu_info,
            format,
            render_pipeline,
            node_buffer,
            data_buffer,
            accumulation,
            prev_camera: None,
            tree_depth,
            tree_root,
            current,
            frame_records: Vec::new(),
            queue: benchmarks,
            timestamp,
            headless,
        }
    }

    /// The size the current benchmark asks to be rendered at.
    fn size(&self) -> (u32, u32) {
        (self.current.def.width, self.current.def.height)
    }

    /// Whether every frame of the current benchmark has been rendered.
    fn current_done(&self) -> bool {
        self.frame_records.len() as u32 >= self.camera_path.frame_count()
    }

    /// Write the current benchmark's results and move on to the next one.
    /// Returns false once the queue is empty.
    fn finish_current(&mut self, resolution: [u32; 2]) -> bool {
        match self.write_results(resolution) {
            Ok(path) => log::info!("Benchmark results written to {}", path.display()),
            Err(e) => log::error!("Failed to write benchmark results: {e}"),
        }

        if self.queue.is_empty() {
            return false;
        }

        let next = self.queue.remove(0);
        log::info!(
            "Running benchmark '{}' (scene: {})",
            next.name,
            next.def.scene
        );

        // Scenes are fragment entry points, so each one gets its own pipeline
        self.render_pipeline = self.gpu.create_pipeline(self.format, &next.def.scene);
        self.camera_path = next.def.camera_path();
        self.frame_records.clear();
        self.prev_camera = None;
        self.current = next;

        true
    }

    /// Render the next frame of the current benchmark into `view` and record how
    /// long it took, counting from `frame_start`.
    fn render_frame(
        &mut self,
        frame_start: Instant,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        if self.accumulation.size != (width, height) {
            self.accumulation =
                self.gpu
                    .create_accumulation(&self.node_buffer, &self.data_buffer, width, height);
            self.prev_camera = None;
        }

        let frame_index = self.frame_records.len() as u32;

        // Evaluate camera path at current frame
        let t = self.camera_path.frame_t(frame_index);
        let pose = self.camera_path.evaluate_frame(frame_index);

        let cam_pos = pose.position;
        let cam_dir = pose.direction();
        let cam_vup = pose.up(Vec3::Y);

        // The camera moves every frame, so each frame builds on the previous
        // one's history by reprojection, as live mode does while moving
        let camera = (cam_pos, cam_dir, cam_vup);
        let (history, prev) = match self.prev_camera {
            None => (gpu_wire::HISTORY_NONE, camera),
            Some(prev) if prev == camera => (gpu_wire::HISTORY_STILL, prev),
            Some(prev) => (gpu_wire::HISTORY_MOVED, prev),
        };
        self.prev_camera = Some(camera);

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let push_constants = gpu_wire::ShaderConstants {
            width,
            height,
            time: t,
            cursor_x: 0.0,
            cursor_y: 0.0,
            cam_pos: cam_pos.into(),
            cam_dir: cam_dir.into(),
            cam_vup: cam_vup.into(),
            tree_depth: self.tree_depth,
            tree_root: self.tree_root,
            frame_count: frame_index,
            prev_cam_pos: prev.0.into(),
            prev_cam_dir: prev.1.into(),
            prev_cam_vup: prev.2.into(),
            history,
            samples: self.current.def.samples,
        };

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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

            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, self.accumulation.bind_group(), &[]);
            rpass.set_push_constants(
                wgpu::ShaderStages::VERTEX_FRAGMENT,
                0,
                bytemuck::bytes_of(&push_constants),
            );
            rpass.draw(0..3, 0..1);
        }

        self.gpu.queue.submit(Some(encoder.finish()));
        self.accumulation.swap();

        // Without vsync the CPU would otherwise race ahead and the elapsed time
        // would measure queue submission rather than the render itself
        self.gpu.device.poll(wgpu::PollType::Wait).ok();

        let frame_time_us = frame_start.elapsed().as_micros() as u64;

        self.frame_records.push(FrameRecord {
            frame: frame_index,
            t,
            time_us: frame_time_us,
            cam_pos: cam_pos.into(),
            cam_dir: cam_dir.into(),
            cam_vup: cam_vup.into(),
        });
    }

    /// Write benchmark results to a JSONL file.
    fn write_results(&self, resolution: [u32; 2]) -> Result<PathBuf, Box<dyn Error>> {
        let output_dir = PathBuf::from("bench/results").join(GIT_SHA);
        fs::create_dir_all(&output_dir)?;

        let filename_timestamp = self.timestamp.format("%Y-%m-%d-%H-%M-%S");
        let output_path = output_dir.join(format!(
            "{}-{}.jsonl",
            filename_timestamp, self.current.name
        ));
        let file = fs::File::create(&output_path)?;
        let mut writer = BufWriter::new(file);

        let metadata = BenchmarkMetadata {
            version: 1,
            timestamp: self.timestamp.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
            git_sha: GIT_SHA.to_string(),
            scene: self.current.def.scene.clone(),
            resolution,
            headless: self.headless,
            gpu: self.gpu_info.clone(),
            camera_path: self.camera_path.clone(),
        };
        serde_json::to_writer(&mut writer, &metadata)?;
        writeln!(writer)?;

        for record in &self.frame_records {
            serde_json::to_writer(&mut writer, record)?;
            writeln!(writer)?;
        }

        writer.flush()?;
        Ok(output_path)
    }
}

/// Application for benchmark mode with animated camera path.
pub struct BenchApp {
    /// Waiting for the window to exist, after which they move into the session.
    benchmarks: Vec<QueuedBenchmark>,
    timestamp: DateTime<Utc>,
    session: Option<Session>,
    config: Option<wgpu::SurfaceConfiguration>,
    /// Declared after everything holding a GPU handle. Fields drop in declaration
    /// order, and the surface has to go before the window it borrows.
    window_surface: Option<WindowSurface>,
    close_requested: bool,
}

impl BenchApp {
    fn new(benchmarks: Vec<QueuedBenchmark>, timestamp: DateTime<Utc>) -> Self {
        Self {
            benchmarks,
            timestamp,
            session: None,
            config: None,
            window_surface: None,
            close_requested: false,
        }
    }

    async fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<(), Box<dyn Error>> {
        let first = &self.benchmarks[0].def;
        let window_attributes = WindowAttributes::default()
            .with_title("Box Benchmark")
            .with_inner_size(PhysicalSize::new(first.width, first.height));
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

        let window_size = window_surface.borrow_window().inner_size();
        let surface = window_surface.borrow_surface();

        let gpu = GpuContext::new(instance, Some(surface)).await?;
        let swapchain_format = surface.get_capabilities(&gpu.adapter).formats[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: window_size.width,
            height: window_size.height,
            // Vsync would clamp every frame time to the refresh interval, hiding
            // any improvement that takes a frame below it
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: Default::default(),
        };
        surface.configure(&gpu.device, &config);

        let benchmarks = std::mem::take(&mut self.benchmarks);
        self.session = Some(Session::new(
            gpu,
            swapchain_format,
            benchmarks,
            self.timestamp,
            false,
        ));
        self.window_surface = Some(window_surface);
        self.config = Some(config);
        Ok(())
    }

    fn render(&mut self) {
        let frame_start = Instant::now();

        let (Some(window_surface), Some(session), Some(config)) =
            (&self.window_surface, &mut self.session, &self.config)
        else {
            return;
        };

        // When all frames are rendered, write results and advance to next benchmark
        if session.current_done() {
            if session.finish_current([config.width, config.height]) {
                // Benchmarks may render at different sizes, so resize before the next one
                let (width, height) = session.size();
                let _ = window_surface
                    .borrow_window()
                    .request_inner_size(PhysicalSize::new(width, height));
            } else {
                self.close_requested = true;
            }
            return;
        }

        let current_size = window_surface.borrow_window().inner_size();
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

        session.render_frame(frame_start, &view, current_size.width, current_size.height);

        frame.present();
    }
}

impl ApplicationHandler for BenchApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.session.is_some() {
            return;
        }
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
                    if let (Some(ws), Some(session)) = (&self.window_surface, &self.session) {
                        ws.borrow_surface().configure(&session.gpu.device, config);
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match &event.logical_key {
                        Key::Named(NamedKey::Escape) => self.close_requested = true,
                        Key::Character(c) if c.as_str() == "q" || c.as_str() == "Q" => {
                            self.close_requested = true
                        }
                        _ => {}
                    }
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

/// Run the benchmarks without a window, drawing each frame into an offscreen
/// texture at exactly the size its config asks for. Nothing is shown, so the
/// machine stays usable while it runs.
fn run_headless(
    benchmarks: Vec<QueuedBenchmark>,
    timestamp: DateTime<Utc>,
    save_frames: bool,
) -> Result<(), Box<dyn Error>> {
    let instance = GpuContext::create_instance();
    let gpu = block_on(GpuContext::new(instance, None))?;
    let mut session = Session::new(gpu, HEADLESS_FORMAT, benchmarks, timestamp, true);

    loop {
        let (width, height) = session.size();
        let target = session.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bench_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HEADLESS_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        let frame_count = session.camera_path.frame_count();
        while !session.current_done() {
            let frame = session.frame_records.len() as u32;
            session.render_frame(Instant::now(), &view, width, height);

            if save_frames && (frame % 100 == 0 || frame + 1 == frame_count) {
                match session.save_frame(&target, frame) {
                    Ok(path) => log::info!("Saved frame to {}", path.display()),
                    Err(e) => log::error!("Failed to save frame {frame}: {e}"),
                }
            }
        }

        if !session.finish_current([width, height]) {
            return Ok(());
        }
    }
}

impl Session {
    /// Read `target` back and write it as a binary PPM to
    /// `bench/frames/<sha>/<benchmark>-<frame>.ppm`. Blocks on the GPU, so it
    /// is only for frames whose timing has already been recorded.
    fn save_frame(&self, target: &wgpu::Texture, frame: u32) -> Result<PathBuf, Box<dyn Error>> {
        let (width, height) = self.size();
        let bytes_per_pixel = 4u32;
        // Buffer copies need rows padded to a 256 byte multiple
        let unpadded = width * bytes_per_pixel;
        let padded = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

        let readback = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_readback"),
            size: (padded * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.gpu.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.gpu.device.poll(wgpu::PollType::Wait)?;
        rx.recv()??;

        let output_dir = PathBuf::from("bench/frames").join(GIT_SHA);
        fs::create_dir_all(&output_dir)?;
        let path = output_dir.join(format!("{}-{frame:04}.ppm", self.current.name));

        // The target is BGRA and PPM wants RGB
        let mapped = slice.get_mapped_range();
        let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
        ppm.reserve((width * height * 3) as usize);
        for row in mapped.chunks(padded as usize) {
            for px in row[..unpadded as usize].chunks(4) {
                ppm.extend_from_slice(&[px[2], px[1], px[0]]);
            }
        }
        drop(mapped);
        readback.unmap();

        fs::write(&path, ppm)?;
        Ok(path)
    }
}

pub fn run_bench(name: String, headless: bool, save_frames: bool) -> Result<(), Box<dyn Error>> {
    let def = BenchmarkFile::load(&name)?;
    run_benchmarks(vec![QueuedBenchmark { name, def }], headless, save_frames)
}

pub fn run_all_benchmarks(headless: bool, save_frames: bool) -> Result<(), Box<dyn Error>> {
    let benchmarks_dir = PathBuf::from("bench/configs");
    let mut benchmark_names: Vec<String> = fs::read_dir(&benchmarks_dir)
        .map_err(|e| format!("Failed to read bench/configs directory: {}", e))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()? == "toml" {
                path.file_stem()?.to_str().map(|s| s.to_string())
            } else {
                None
            }
        })
        .collect();

    benchmark_names.sort();

    if benchmark_names.is_empty() {
        return Err("No benchmark files found in bench/configs/ directory".into());
    }

    log::info!(
        "Found {} benchmark(s): {}",
        benchmark_names.len(),
        benchmark_names.join(", ")
    );

    let mut benchmarks = Vec::new();
    for name in benchmark_names {
        let def = BenchmarkFile::load(&name)?;
        benchmarks.push(QueuedBenchmark { name, def });
    }

    run_benchmarks(benchmarks, headless, save_frames)
}

fn run_benchmarks(
    benchmarks: Vec<QueuedBenchmark>,
    headless: bool,
    save_frames: bool,
) -> Result<(), Box<dyn Error>> {
    if headless {
        return run_headless(benchmarks, Utc::now(), save_frames);
    }
    if save_frames {
        log::warn!("--save-frames only applies to headless runs; ignoring");
    }

    let event_loop = EventLoop::new()?;
    let mut app = BenchApp::new(benchmarks, Utc::now());
    event_loop.run_app(&mut app).map_err(Into::into)
}
