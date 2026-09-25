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
use box_bench::BenchmarkMetadata;
use box_bench::CameraPath;
use box_bench::FrameRecord;
use box_bench::GpuInfo;
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
use crate::live_app::LiveApp;
use crate::window_surface::WindowSurface;
use crate::window_surface::WindowSurfaceBuilder;

/// Benchmark definition loaded from a TOML file.
#[derive(Deserialize)]
pub struct BenchmarkFile {
    pub scene: String,
    pub frame_count: u32,
    pub position: Vec<[f32; 3]>,
    pub look_at: Vec<[f32; 3]>,
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

/// A queued benchmark with its name and definition.
struct QueuedBenchmark {
    name: String,
    def: BenchmarkFile,
}

/// Application for benchmark mode with animated camera path.
pub struct BenchApp {
    gpu: Option<GpuContext>,
    gpu_info: Option<GpuInfo>,
    window_surface: Option<WindowSurface>,
    config: Option<wgpu::SurfaceConfiguration>,
    render_pipeline: Option<wgpu::RenderPipeline>,
    bind_group: Option<wgpu::BindGroup>,
    swapchain_format: Option<wgpu::TextureFormat>,
    close_requested: bool,
    camera_path: CameraPath,
    frame_records: Vec<FrameRecord>,
    frame_count: u32,
    name: String,
    scene: String,
    timestamp: DateTime<Utc>,
    queue: Vec<QueuedBenchmark>,
    tree_depth: u32,
    tree_root: u32,
}

impl BenchApp {
    pub fn new(benchmarks: Vec<(String, BenchmarkFile)>, timestamp: DateTime<Utc>) -> Self {
        let mut queue: Vec<QueuedBenchmark> = benchmarks
            .into_iter()
            .map(|(name, def)| QueuedBenchmark { name, def })
            .collect();

        let first = queue.remove(0);
        let camera_path = CameraPath::new(
            first.def.position_points(),
            first.def.look_at_points(),
            first.def.frame_count,
        );

        Self {
            gpu: None,
            gpu_info: None,
            window_surface: None,
            config: None,
            render_pipeline: None,
            bind_group: None,
            swapchain_format: None,
            close_requested: false,
            camera_path,
            frame_records: Vec::new(),
            frame_count: 0,
            name: first.name,
            scene: first.def.scene,
            timestamp,
            queue,
            tree_depth: 0,
            tree_root: 0,
        }
    }

    /// Advance to the next benchmark in the queue. Returns false if queue is empty.
    fn advance_to_next(&mut self) -> bool {
        if self.queue.is_empty() {
            return false;
        }

        let next = self.queue.remove(0);
        log::info!(
            "Running benchmark '{}' (scene: {})",
            next.name,
            next.def.scene
        );

        self.camera_path = CameraPath::new(
            next.def.position_points(),
            next.def.look_at_points(),
            next.def.frame_count,
        );

        self.frame_records.clear();
        self.frame_count = 0;
        self.name = next.name;
        self.scene = next.def.scene.clone();

        if let (Some(gpu), Some(format)) = (&self.gpu, self.swapchain_format) {
            self.render_pipeline = Some(gpu.create_pipeline(format, &next.def.scene));
        }

        true
    }

    async fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<(), Box<dyn Error>> {
        let window_attributes = WindowAttributes::default()
            .with_title("Box Benchmark")
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

        let window_size = window_surface.borrow_window().inner_size();
        let surface = window_surface.borrow_surface();

        let gpu = GpuContext::new(instance, Some(surface)).await?;
        let gpu_info = gpu_info_from_adapter(&gpu.adapter);

        // Build tree64 and upload to GPU
        let (nodes_u32, data_u32, tree_depth, tree_root) = LiveApp::build_tree64();
        let node_buffer = gpu.create_storage_buffer(&nodes_u32);
        let data_buffer = gpu.create_storage_buffer(&data_u32);
        let accum_buffer = gpu.create_accum_buffer(window_size.width, window_size.height);
        let bind_group = gpu.create_bind_group(&node_buffer, &data_buffer, &accum_buffer);

        let swapchain_format = surface.get_capabilities(&gpu.adapter).formats[0];
        let render_pipeline = gpu.create_pipeline(swapchain_format, &self.scene);

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
        self.gpu_info = Some(gpu_info);
        self.window_surface = Some(window_surface);
        self.config = Some(config);
        self.render_pipeline = Some(render_pipeline);
        self.bind_group = Some(bind_group);
        self.swapchain_format = Some(swapchain_format);
        self.tree_depth = tree_depth;
        self.tree_root = tree_root;
        log::info!("Running benchmark '{}' (scene: {})", self.name, self.scene);
        Ok(())
    }

    fn render(&mut self) {
        let frame_start = Instant::now();

        let window_surface = match &self.window_surface {
            Some(ws) => ws,
            None => return,
        };
        let gpu = match &self.gpu {
            Some(gpu) => gpu,
            None => return,
        };

        // When all frames are rendered, write results and advance
        if self.frame_count >= self.camera_path.frame_count() {
            match self.write_results() {
                Ok(path) => log::info!("Benchmark results written to {}", path.display()),
                Err(e) => log::error!("Failed to write benchmark results: {e}"),
            }
            if !self.advance_to_next() {
                self.close_requested = true;
            }
            return;
        }

        // Evaluate camera path at current frame
        let t = self.camera_path.frame_t(self.frame_count);
        let pose = self.camera_path.evaluate_frame(self.frame_count);

        let cam_pos = pose.position;
        let cam_dir = pose.direction();
        let cam_vup = pose.up(Vec3::Y);

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

        let push_constants = shared::ShaderConstants {
            width: current_size.width,
            height: current_size.height,
            time: t,
            cursor_x: 0.0,
            cursor_y: 0.0,
            cam_pos: cam_pos.into(),
            cam_dir: cam_dir.into(),
            cam_vup: cam_vup.into(),
            tree_depth: self.tree_depth,
            tree_root: self.tree_root,
            frame_count: 0,
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

        // Record frame timing
        let frame_time_us = frame_start.elapsed().as_micros() as u64;
        self.frame_records.push(FrameRecord {
            frame: self.frame_count,
            t,
            time_us: frame_time_us,
            cam_pos: cam_pos.into(),
            cam_dir: cam_dir.into(),
            cam_vup: cam_vup.into(),
        });
        self.frame_count += 1;
    }

    /// Write benchmark results to a JSONL file.
    fn write_results(&self) -> Result<PathBuf, Box<dyn Error>> {
        let gpu_info = self.gpu_info.as_ref().ok_or("No GPU info")?;
        let config = self.config.as_ref().ok_or("No surface config")?;

        let output_dir = PathBuf::from("bench/results").join(GIT_SHA);
        fs::create_dir_all(&output_dir)?;

        let filename_timestamp = self.timestamp.format("%Y-%m-%d-%H-%M-%S");
        let output_path = output_dir.join(format!("{}-{}.jsonl", filename_timestamp, self.name));
        let file = fs::File::create(&output_path)?;
        let mut writer = BufWriter::new(file);

        let metadata = BenchmarkMetadata {
            version: 1,
            timestamp: self.timestamp.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
            git_sha: GIT_SHA.to_string(),
            scene: self.scene.clone(),
            resolution: [config.width, config.height],
            gpu: gpu_info.clone(),
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

impl ApplicationHandler for BenchApp {
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

pub fn run_bench(name: String) -> Result<(), Box<dyn Error>> {
    let def = BenchmarkFile::load(&name)?;
    let benchmarks = vec![(name, def)];
    run_benchmarks(benchmarks)
}

pub fn run_all_benchmarks() -> Result<(), Box<dyn Error>> {
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
        benchmarks.push((name, def));
    }

    run_benchmarks(benchmarks)
}

fn run_benchmarks(benchmarks: Vec<(String, BenchmarkFile)>) -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = BenchApp::new(benchmarks, Utc::now());
    event_loop.run_app(&mut app).map_err(Into::into)
}
