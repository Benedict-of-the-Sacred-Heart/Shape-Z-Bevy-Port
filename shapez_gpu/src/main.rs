use argh::FromArgs;
use bytemuck::{Pod, Zeroable};
use shapez_gpu::{
    gpu_scene::GpuScene,
    scene::{SceneBuilder, Voxel},
};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[derive(FromArgs, Debug)]
/// Shape-Z GPU prototype runner.
pub struct Args {
    /// window width
    #[argh(option, default = "800")]
    pub width: u32,
    /// window height
    #[argh(option, default = "600")]
    pub height: u32,
}

fn main() -> anyhow::Result<()> {
    let args: Args = argh::from_env();
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Shape-Z GPU Prototype")
        .with_inner_size(winit::dpi::PhysicalSize::new(args.width, args.height))
        .build(&event_loop)?;

    let window = Arc::new(window);

    let instance = wgpu::Instance::default();
    let surface = instance.create_surface(window.as_ref())?;
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| anyhow::anyhow!("No suitable GPU adapters found."))?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("ShapeZ GPU Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))?;

    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(surface_caps.formats[0]);

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: args.width,
        height: args.height,
        present_mode: surface_caps.present_modes[0],
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    // Temporary native scene example; this will later feed GPU buffers.
    let scene = SceneBuilder::new([32, 32, 32])
        .with_voxel(|volume| {
            volume.set_voxel(
                16,
                16,
                16,
                Voxel {
                    material: 0,
                    volumetric_id: None,
                },
            );
        })
        .build();
    let gpu_scene = Arc::new(GpuScene::from_scene(&device, &queue, &scene));

    // ===================
    // GPU resources setup
    // ===================
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct SceneConstants {
        image_width: u32,
        image_height: u32,
        frame_index: u32,
        samples_per_dispatch: u32,
        voxel_dim_x: u32,
        voxel_dim_y: u32,
        voxel_dim_z: u32,
        material_count: u32,
    }

    fn build_scene_constants(
        width: u32,
        height: u32,
        frame_index: u32,
        samples_per_dispatch: u32,
        gpu_scene: &GpuScene,
    ) -> SceneConstants {
        SceneConstants {
            image_width: width,
            image_height: height,
            frame_index,
            samples_per_dispatch,
            voxel_dim_x: gpu_scene.voxel_dims[0],
            voxel_dim_y: gpu_scene.voxel_dims[1],
            voxel_dim_z: gpu_scene.voxel_dims[2],
            material_count: gpu_scene.material_count,
        }
    }

    fn write_scene_constants(
        queue: &wgpu::Queue,
        scene_ubo: &wgpu::Buffer,
        width: u32,
        height: u32,
        frame_index: u32,
        samples_per_dispatch: u32,
        gpu_scene: &GpuScene,
    ) {
        let constants = build_scene_constants(width, height, frame_index, samples_per_dispatch, gpu_scene);
        queue.write_buffer(scene_ubo, 0, bytemuck::bytes_of(&constants));
    }

    fn create_accum_texture(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Accumulation Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    // Buffers: uniform, per-pixel counts, RNG.
    let mut frame_index: u32 = 0;
    let samples_per_dispatch: u32 = 1;
    let scene_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Scene UBO"),
        contents: bytemuck::bytes_of(&build_scene_constants(
            config.width,
            config.height,
            frame_index,
            samples_per_dispatch,
            &gpu_scene,
        )),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Accumulation texture (rgba16float, storage + sampleable)
    let accum_format = wgpu::TextureFormat::Rgba16Float;
    let display_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Display Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    // Compute pipeline (path tracer)
    let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Path Tracer Compute"),
        source: wgpu::ShaderSource::Wgsl(include_str!("path_tracer.wgsl").into()),
    });

    let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Compute BGL"),
        entries: &[
            // Scene UBO
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Camera UBO
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Storage texture write
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: accum_format,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            // Sample counts buffer
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // RNG state buffer
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Materials buffer (read-only)
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Voxels buffer (read-only)
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Layout"),
        bind_group_layouts: &[&compute_bgl],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Path Trace Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &compute_shader,
        entry_point: "path_trace",
    });

    // Display pipeline (fullscreen sample of accumulation texture)
    let display_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Display Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("display.wgsl").into()),
    });

    let display_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Display BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    });

    let display_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Display Layout"),
        bind_group_layouts: &[&display_bgl],
        push_constant_ranges: &[],
    });

    let display_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Display Pipeline"),
        layout: Some(&display_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &display_shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &display_shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    struct FrameResources {
        _accum_tex: wgpu::Texture,
        _accum_view: wgpu::TextureView,
        _sample_counts: wgpu::Buffer,
        _rng_state: wgpu::Buffer,
        compute_bg: wgpu::BindGroup,
        display_bg: wgpu::BindGroup,
    }

    impl FrameResources {
        fn new(
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            config: &wgpu::SurfaceConfiguration,
            accum_format: wgpu::TextureFormat,
            display_bgl: &wgpu::BindGroupLayout,
            compute_bgl: &wgpu::BindGroupLayout,
            display_sampler: &wgpu::Sampler,
            scene_ubo: &wgpu::Buffer,
            gpu_scene: &GpuScene,
        ) -> Self {
            let pixel_count = (config.width * config.height) as usize;
            let sample_counts = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Sample Counts"),
                size: (pixel_count * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let rng_state = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RNG State"),
                size: (pixel_count * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let seeds: Vec<u32> = (0..pixel_count as u32).map(|i| i ^ 0xA2C2A3E9).collect();
            queue.write_buffer(&rng_state, 0, bytemuck::cast_slice(&seeds));
            let zeros = vec![0u32; pixel_count];
            queue.write_buffer(&sample_counts, 0, bytemuck::cast_slice(&zeros));

            let accum_tex = create_accum_texture(device, accum_format, config.width, config.height);
            let accum_view = accum_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let display_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Display BG"),
                layout: display_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&accum_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(display_sampler),
                    },
                ],
            });
            let compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Compute BG"),
                layout: compute_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: scene_ubo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: gpu_scene.camera_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&accum_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: sample_counts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: rng_state.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: gpu_scene.materials_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: gpu_scene.voxels_buffer.as_entire_binding(),
                    },
                ],
            });

            Self {
                _accum_tex: accum_tex,
                _accum_view: accum_view,
                _sample_counts: sample_counts,
                _rng_state: rng_state,
                compute_bg,
                display_bg,
            }
        }

        fn rebuild(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            config: &wgpu::SurfaceConfiguration,
            accum_format: wgpu::TextureFormat,
            display_bgl: &wgpu::BindGroupLayout,
            compute_bgl: &wgpu::BindGroupLayout,
            display_sampler: &wgpu::Sampler,
            scene_ubo: &wgpu::Buffer,
            gpu_scene: &GpuScene,
        ) {
            *self = Self::new(
                device,
                queue,
                config,
                accum_format,
                display_bgl,
                compute_bgl,
                display_sampler,
                scene_ubo,
                gpu_scene,
            );
        }
    }

    let mut frame_resources = FrameResources::new(
        &device,
        &queue,
        &config,
        accum_format,
        &display_bgl,
        &compute_bgl,
        &display_sampler,
        &scene_ubo,
        &gpu_scene,
    );
    let window_for_loop = Arc::clone(&window);
    let surface = Arc::new(surface);
    let gpu_scene_for_loop = Arc::clone(&gpu_scene);

    event_loop.run(move |event, target| {
        target.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { window_id, event } if window_id == window_for_loop.id() => {
                match event {
                    WindowEvent::CloseRequested => target.exit(),
                    WindowEvent::Resized(new_size) => {
                        if new_size.width > 0 && new_size.height > 0 {
                            config.width = new_size.width;
                            config.height = new_size.height;
                            surface.configure(&device, &config);
                            frame_index = 0;
                            write_scene_constants(
                                &queue,
                                &scene_ubo,
                                config.width,
                                config.height,
                                frame_index,
                                samples_per_dispatch,
                                &gpu_scene_for_loop,
                            );
                            frame_resources.rebuild(
                                &device,
                                &queue,
                                &config,
                                accum_format,
                                &display_bgl,
                                &compute_bgl,
                                &display_sampler,
                                &scene_ubo,
                                &gpu_scene_for_loop,
                            );
                        }
                    }
                    WindowEvent::ScaleFactorChanged {
                        scale_factor: _,
                        mut inner_size_writer,
                    } => {
                        let current_size = window_for_loop.inner_size();
                        let _ = inner_size_writer.request_inner_size(current_size);
                        if current_size.width > 0 && current_size.height > 0 {
                            config.width = current_size.width;
                            config.height = current_size.height;
                            surface.configure(&device, &config);
                            frame_index = 0;
                            write_scene_constants(
                                &queue,
                                &scene_ubo,
                                config.width,
                                config.height,
                                frame_index,
                                samples_per_dispatch,
                                &gpu_scene_for_loop,
                            );
                            frame_resources.rebuild(
                                &device,
                                &queue,
                                &config,
                                accum_format,
                                &display_bgl,
                                &compute_bgl,
                                &display_sampler,
                                &scene_ubo,
                                &gpu_scene_for_loop,
                            );
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        frame_index = frame_index.wrapping_add(1);
                        write_scene_constants(
                            &queue,
                            &scene_ubo,
                            config.width,
                            config.height,
                            frame_index,
                            samples_per_dispatch,
                            &gpu_scene_for_loop,
                        );

                        match surface.get_current_texture() {
                            Ok(frame) => {
                                let view = frame
                                    .texture
                                    .create_view(&wgpu::TextureViewDescriptor::default());
                                let mut encoder = device.create_command_encoder(
                                    &wgpu::CommandEncoderDescriptor {
                                        label: Some("Render Encoder"),
                                    },
                                );

                                {
                                    let mut cpass =
                                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                            label: Some("Path Trace Pass"),
                                            timestamp_writes: None,
                                        });
                                    cpass.set_pipeline(&compute_pipeline);
                                    cpass.set_bind_group(0, &frame_resources.compute_bg, &[]);
                                    let wg_x = config.width.div_ceil(8);
                                    let wg_y = config.height.div_ceil(8);
                                    cpass.dispatch_workgroups(wg_x, wg_y, 1);
                                }

                                {
                                    let mut pass =
                                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                            label: Some("Display Pass"),
                                            color_attachments: &[Some(
                                                wgpu::RenderPassColorAttachment {
                                                    view: &view,
                                                    resolve_target: None,
                                                    ops: wgpu::Operations {
                                                        load: wgpu::LoadOp::Clear(
                                                            wgpu::Color::BLACK,
                                                        ),
                                                        store: wgpu::StoreOp::Store,
                                                    },
                                                },
                                            )],
                                            depth_stencil_attachment: None,
                                            timestamp_writes: None,
                                            occlusion_query_set: None,
                                        });
                                    pass.set_pipeline(&display_pipeline);
                                    pass.set_bind_group(0, &frame_resources.display_bg, &[]);
                                    pass.draw(0..3, 0..1);
                                }

                                queue.submit(std::iter::once(encoder.finish()));
                                frame.present();
                            }
                            Err(wgpu::SurfaceError::Lost) => surface.configure(&device, &config),
                            Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                            Err(_) => {}
                        }
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                window_for_loop.request_redraw();
            }
            _ => {}
        }
    })?;

    Ok(())
}
