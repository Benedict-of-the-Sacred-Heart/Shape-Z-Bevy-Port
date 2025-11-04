use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::scene::{Camera, Material, SceneDescription, Voxel};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub origin: [f32; 4],
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub forward: [f32; 4],
    pub params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuMaterial {
    pub base_color_and_roughness: [f32; 4],
    pub emission_and_metallic: [f32; 4],
    pub misc: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuVoxel {
    pub data: [u32; 4],
}

pub struct GpuScene {
    pub camera_buffer: wgpu::Buffer,
    pub materials_buffer: wgpu::Buffer,
    pub voxels_buffer: wgpu::Buffer,
    pub voxel_dims: [u32; 3],
    pub material_count: u32,
}

impl GpuScene {
    pub fn from_scene(device: &wgpu::Device, queue: &wgpu::Queue, scene: &SceneDescription) -> Self {
        let camera_uniform = camera_uniform_from(&scene.camera);
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform Buffer"),
            contents: bytemuck::bytes_of(&camera_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let gpu_materials: Vec<GpuMaterial> = scene
            .materials
            .iter()
            .map(material_to_gpu)
            .collect();

        let materials_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Materials Buffer"),
            contents: bytemuck::cast_slice(&gpu_materials),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let voxel_data: Vec<GpuVoxel> = scene
            .volume
            .voxels()
            .iter()
            .copied()
            .map(voxel_to_gpu)
            .collect();

        let voxels_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Voxels Buffer"),
            contents: bytemuck::cast_slice(&voxel_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        queue.write_buffer(&camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));

        Self {
            camera_buffer,
            materials_buffer,
            voxels_buffer,
            voxel_dims: scene.volume.dims,
            material_count: gpu_materials.len() as u32,
        }
    }
}

fn camera_uniform_from(camera: &Camera) -> CameraUniform {
    let origin = camera.origin;
    let target = camera.target;
    let up = camera.up;

    let forward = normalize(sub(target, origin));
    let mut right = cross(forward, up);
    if length(right) == 0.0 {
        right = [1.0, 0.0, 0.0];
    } else {
        right = normalize(right);
    }
    let up = normalize(cross(right, forward));

    let vertical_fov = camera.vertical_fov_degrees.to_radians();
    let half_vertical = (vertical_fov * 0.5).tan();

    CameraUniform {
        origin: [origin[0], origin[1], origin[2], 1.0],
        right: [right[0], right[1], right[2], 0.0],
        up: [up[0], up[1], up[2], 0.0],
        forward: [forward[0], forward[1], forward[2], 0.0],
        params: [half_vertical, camera.aperture, camera.focus_distance, 0.0],
    }
}

fn material_to_gpu(mat: &Material) -> GpuMaterial {
    GpuMaterial {
        base_color_and_roughness: [mat.base_color[0], mat.base_color[1], mat.base_color[2], mat.roughness],
        emission_and_metallic: [mat.emission[0], mat.emission[1], mat.emission[2], mat.metallic],
        misc: [mat.shader_model as f32, 0.0, 0.0, 0.0],
    }
}

fn voxel_to_gpu(voxel: Voxel) -> GpuVoxel {
    let volumetric = voxel.volumetric_id.map(|v| v as u32).unwrap_or(u32::MAX);
    GpuVoxel {
        data: [voxel.material as u32, volumetric, 0, 0],
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = length(v);
    if len == 0.0 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}
