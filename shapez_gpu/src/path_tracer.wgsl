struct SceneConstants {
    image_width: u32,
    image_height: u32,
    frame_index: u32,
    samples_per_dispatch: u32,
    voxel_dim_x: u32,
    voxel_dim_y: u32,
    voxel_dim_z: u32,
    material_count: u32,
};

struct CameraUniform {
    origin: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    forward: vec4<f32>,
    params: vec4<f32>,
};

struct GpuMaterial {
    base_color_and_roughness: vec4<f32>,
    emission_and_metallic: vec4<f32>,
    misc: vec4<f32>,
};

struct GpuVoxel {
    data: vec4<u32>,
};

@group(0) @binding(0)
var<uniform> scene: SceneConstants;

@group(0) @binding(1)
var<uniform> camera: CameraUniform;

// Storage texture written by compute, sampled by render pass.
@group(0) @binding(2)
var accum_tex: texture_storage_2d<rgba16float, write>;

@group(0) @binding(3)
var<storage, read_write> sample_counts: array<u32>;

@group(0) @binding(4)
var<storage, read_write> rng_state: array<u32>;

@group(0) @binding(5)
var<storage, read> materials: array<GpuMaterial>;

@group(0) @binding(6)
var<storage, read> voxels: array<GpuVoxel>;

@compute @workgroup_size(8, 8, 1)
fn path_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Guard dispatch bounds.
    if (gid.x >= scene.image_width || gid.y >= scene.image_height) {
        return;
    }

    let pixel_index = gid.y * scene.image_width + gid.x;
    let seed = rng_state[pixel_index];
    let width_f = f32(scene.image_width);
    let height_f = f32(scene.image_height);
    let uv = vec2<f32>((f32(gid.x) + 0.5) / width_f, (f32(gid.y) + 0.5) / height_f);

    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let half_vertical = camera.params.x;
    let aspect = width_f / height_f;
    let half_horizontal = half_vertical * aspect;
    let forward = camera.forward.xyz;
    let right = camera.right.xyz;
    let up = camera.up.xyz;
    let dir = normalize(forward + right * ndc.x * half_horizontal - up * ndc.y * half_vertical);

    var base_color = vec3<f32>(0.8, 0.8, 0.8);
    if (scene.material_count > 0u) {
        base_color = materials[0u].base_color_and_roughness.xyz;
    }

    let gradient = 0.5 + 0.5 * dir.y;
    let contribution = base_color * gradient + vec3<f32>(0.0, 0.0, 0.5 * sin(f32(scene.frame_index) * 0.05));
    let count = sample_counts[pixel_index];
    let new_count = count + 1u;
    // Simple running average in-place would require a read; for now, just write the contribution.
    textureStore(accum_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(contribution, 1.0));
    sample_counts[pixel_index] = new_count;
    rng_state[pixel_index] = seed ^ 0x9E3779B9u;
}
