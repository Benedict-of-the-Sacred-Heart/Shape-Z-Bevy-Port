@group(0) @binding(0)
var accum_tex: texture_2d<f32>;
@group(0) @binding(1)
var accum_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(3.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    let pos = positions[vertex_index];
    return vec4<f32>(pos, 0.0, 1.0);
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let a = 0.055;
    return select(c * 12.92, (1.0 + a) * pow(c, vec3<f32>(1.0 / 2.4)) - a, c > vec3<f32>(0.0031308));
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(accum_tex));
    let uv = pos.xy / dims;
    let color = textureSampleLevel(accum_tex, accum_sampler, uv, 0.0).xyz;
    let srgb = linear_to_srgb(color);
    return vec4<f32>(srgb, 1.0);
}
