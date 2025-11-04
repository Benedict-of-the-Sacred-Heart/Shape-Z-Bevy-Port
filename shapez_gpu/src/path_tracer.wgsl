struct SceneConstants {
    image_size: vec2<u32>,
    frame_index: u32,
    samples_per_dispatch: u32,
};

@group(0) @binding(0)
var<uniform> scene: SceneConstants;

// Storage texture written by compute, sampled by render pass.
@group(0) @binding(1)
var accum_tex: texture_storage_2d<rgba16float, write>;

// Optional per-pixel state (stubbed for now)
@group(0) @binding(2)
var<storage, read_write> sample_counts: array<u32>;

@group(0) @binding(3)
var<storage, read_write> rng_state: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn path_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Guard dispatch bounds.
    if (gid.x >= scene.image_size.x || gid.y >= scene.image_size.y) {
        return;
    }

    let pixel_index = gid.y * scene.image_size.x + gid.x;
    let seed = rng_state[pixel_index];
    // Placeholder: gradient + frame-index flicker to prove compute is running.
    let uv = vec2<f32>(f32(gid.x) / f32(scene.image_size.x), f32(gid.y) / f32(scene.image_size.y));
    let contribution = vec3<f32>(uv.x, uv.y, 0.5 + 0.5 * sin(f32(scene.frame_index) * 0.05));
    let count = sample_counts[pixel_index];
    let new_count = count + 1u;
    // Simple running average in-place would require a read; for now, just write the contribution.
    textureStore(accum_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(contribution, 1.0));
    sample_counts[pixel_index] = new_count;
    rng_state[pixel_index] = seed ^ 0x9E3779B9u;
}
