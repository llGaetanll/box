use std::error::Error;

use wgpu::InstanceDescriptor;
use wgpu::include_spirv;
use wgpu::include_spirv_raw;
use wgpu::{self};

pub struct GpuContext {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub shader_module: wgpu::ShaderModule,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuContext {
    pub fn create_instance() -> wgpu::Instance {
        let mut instance_flags = wgpu::InstanceFlags::default();
        instance_flags.remove(wgpu::InstanceFlags::VALIDATION);
        instance_flags.remove(wgpu::InstanceFlags::DEBUG);

        wgpu::Instance::new(&InstanceDescriptor {
            flags: instance_flags,
            ..Default::default()
        })
    }

    /// Create a new GPU context. If a surface is provided, the adapter will be
    /// selected for compatibility with that surface.
    pub async fn new(
        instance: wgpu::Instance,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Self, Box<dyn Error>> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface,
                force_fallback_adapter: false,
            })
            .await?;

        let mut required_features = wgpu::Features::PUSH_CONSTANTS;
        if adapter
            .features()
            .contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH)
        {
            required_features |= wgpu::Features::SPIRV_SHADER_PASSTHROUGH;
        }

        let required_limits = wgpu::Limits {
            max_push_constant_size: 256,
            ..Default::default()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features,
                required_limits,
                ..Default::default()
            })
            .await?;

        let shader_module = if device
            .features()
            .contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH)
        {
            let spirv = include_spirv_raw!(env!("gpu_shader.spv"));
            unsafe { device.create_shader_module_passthrough(spirv) }
        } else {
            device.create_shader_module(include_spirv!(env!("gpu_shader.spv")))
        };

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tree64_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        Ok(Self {
            adapter,
            device,
            queue,
            shader_module,
            bind_group_layout,
        })
    }

    /// Create a storage buffer from a slice of u32 data.
    pub fn create_storage_buffer(&self, data: &[u32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("storage_buffer"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    /// Create a bind group for the tree64 node and data buffers, the previous
    /// frame's accumulation buffer to read, and this frame's to write.
    pub fn create_bind_group(
        &self,
        node_buffer: &wgpu::Buffer,
        data_buffer: &wgpu::Buffer,
        history_buffer: &wgpu::Buffer,
        accum_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tree64_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: node_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: history_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: accum_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create a storage buffer for accumulation (read-write, zero-initialized).
    pub fn create_accum_buffer(&self, width: u32, height: u32) -> wgpu::Buffer {
        let size =
            (width * height * gpu_wire::PIXEL_WORDS) as u64 * std::mem::size_of::<u32>() as u64;
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("accum_buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }

    /// Two accumulation buffers and the bind groups that read one while
    /// writing the other, alternated frame by frame so each frame can blend
    /// into the previous one's result.
    pub fn create_accumulation(
        &self,
        node_buffer: &wgpu::Buffer,
        data_buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) -> Accumulation {
        let buffers = [
            self.create_accum_buffer(width, height),
            self.create_accum_buffer(width, height),
        ];
        let bind_groups = [
            self.create_bind_group(node_buffer, data_buffer, &buffers[1], &buffers[0]),
            self.create_bind_group(node_buffer, data_buffer, &buffers[0], &buffers[1]),
        ];
        Accumulation {
            _buffers: buffers,
            bind_groups,
            size: (width, height),
            writing: 0,
        }
    }
}

/// The pair of accumulation buffers a renderer alternates between. See
/// `GpuContext::create_accumulation`.
pub struct Accumulation {
    _buffers: [wgpu::Buffer; 2],
    bind_groups: [wgpu::BindGroup; 2],
    pub size: (u32, u32),
    /// Which buffer the next frame writes.
    writing: usize,
}

impl Accumulation {
    /// The bind group for the next frame: it reads the buffer the last frame
    /// wrote and writes the other one.
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_groups[self.writing]
    }

    /// The frame has been recorded; the next one reads what it wrote.
    pub fn swap(&mut self) {
        self.writing ^= 1;
    }
}

impl GpuContext {
    /// Create a render pipeline for the given texture format and fragment entry point.
    pub fn create_pipeline(
        &self,
        format: wgpu::TextureFormat,
        fragment_entry_point: &str,
    ) -> wgpu::RenderPipeline {
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&self.bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    range: 0..std::mem::size_of::<gpu_wire::ShaderConstants>() as u32,
                }],
            });

        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &self.shader_module,
                    entry_point: Some("main_vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &self.shader_module,
                    entry_point: Some(fragment_entry_point),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
    }
}
