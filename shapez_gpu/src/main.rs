use std::{
    num::NonZeroU32,
    path::PathBuf,
    sync::{mpsc, Arc},
};

use argh::FromArgs;
use bytemuck::{Pod, Zeroable};
use image::{ImageBuffer, Rgba};
use shapez_gpu::{
    gpu_scene::GpuScene,
    scene::{SceneBuilder, Voxel},
};
use wgpu::util::DeviceExt;

#[derive(FromArgs, Debug)]
/// Shape-Z GPU prototype runner.
pub struct Args {
    /// output image width
    #[argh(option, default = "800")]
    pub width: u32,
    /// output image height
    #[argh(option, default = "600")]
    pub height: u32,
    /// where to write the rendered PNG
    #[argh(option)]
    pub output: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let args: Args = argh::from_env();
    let output_path = args
        .output
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("shapez_gpu.png"));

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
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

    let image_width = args.width;
    let image_height = args.height;

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
            image_width,
            image_height,
            frame_index,
            samples_per_dispatch,
            &gpu_scene,
        )),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Accumulation texture (rgba32float, storage)
    let accum_format = wgpu::TextureFormat::Rgba32Float;

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
    // Prepare storage buffers
    let pixel_count = (image_width * image_height) as usize;
    let sample_counts = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sample Counts"),
        size: (pixel_count * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
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

    let accum_tex = create_accum_texture(&device, accum_format, image_width, image_height);
    let accum_view = accum_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Compute BG"),
        layout: &compute_bgl,
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

    // Dispatch compute once for now
    write_scene_constants(
        &queue,
        &scene_ubo,
        image_width,
        image_height,
        frame_index,
        samples_per_dispatch,
        &gpu_scene,
    );

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Headless Encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Path Trace Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &compute_bg, &[]);
        let wg_x = image_width.div_ceil(8);
        let wg_y = image_height.div_ceil(8);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    // Copy accumulation texture into a staging buffer for readback
    let output_buffer_size = (pixel_count * std::mem::size_of::<[f32; 4]>()) as u64;
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &accum_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &readback_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(image_width * 16),
                rows_per_image: Some(image_height),
            },
        },
        wgpu::Extent3d {
            width: image_width,
            height: image_height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(std::iter::once(encoder.finish()));

    // Read back
    {
        let buffer_slice = readback_buffer.slice(..);
        let (tx, rx) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).expect("map_async callback send failed");
        });
        device.poll(wgpu::Maintain::Wait);
        let map_result = rx.recv().map_err(|_| anyhow::anyhow!("failed to receive map result"))?;
        map_result.map_err(|_| anyhow::anyhow!("failed to map readback buffer"))?;
        let data = buffer_slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let mut img = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(image_width, image_height);
        for y in 0..image_height {
            for x in 0..image_width {
                let idx = ((y * image_width + x) * 4) as usize;
                let r = floats[idx];
                let g = floats[idx + 1];
                let b = floats[idx + 2];
                let color = [
                    (r.clamp(0.0, 1.0) * 255.0) as u8,
                    (g.clamp(0.0, 1.0) * 255.0) as u8,
                    (b.clamp(0.0, 1.0) * 255.0) as u8,
                    255,
                ];
                img.put_pixel(x, y, Rgba(color));
            }
        }
        img.save(&output_path)?;
        drop(data);
        readback_buffer.unmap();
    }

    println!("wrote {:?}", output_path);

    Ok(())
}
