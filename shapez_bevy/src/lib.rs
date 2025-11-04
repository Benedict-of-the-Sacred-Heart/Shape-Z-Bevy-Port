use bevy::pbr::MeshMaterial3d;
use bevy::prelude::Color as BevyColor;
use bevy::{
    asset::RenderAssetUsages, math::Vec3, prelude::*, render::render_resource::PrimitiveTopology,
};
use shapezlib::shapez::ShapeZ;

#[derive(Resource, Default)]
pub struct ShapeZSettings {
    pub enabled: bool,
}

/// Attach to an entity to generate a mesh from a `.shpz` file and insert a PBR bundle.
#[derive(Component, Debug)]
pub struct ShapeZVolume {
    pub path: std::path::PathBuf,
    pub spawned: bool,
}

impl ShapeZVolume {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            path: path.into(),
            spawned: false,
        }
    }
}

pub struct ShapeZPlugin;

impl Plugin for ShapeZPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ShapeZSettings>()
            .add_systems(Update, spawn_shapez_meshes);
    }
}

fn spawn_shapez_meshes(
    mut cmds: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut q: Query<(Entity, &mut ShapeZVolume)>,
) {
    for (entity, mut vol) in &mut q {
        if vol.spawned {
            continue;
        }

        let path = vol.path.clone();
        let mut engine = ShapeZ::default();
        match engine.parse(path.clone()) {
            Ok(module) => {
                if engine.compile(&module).is_ok() {
                    engine.execute();
                    let (positions, indices, face_mats) = engine.mesh_triangles();
                    if !positions.is_empty() && !indices.is_empty() {
                        // Group triangles by material id and spawn child entities per material
                        let grouped =
                            triangles_grouped_by_material(&positions, &indices, &face_mats);
                        let mut parent = cmds.entity(entity);
                        for (mat_id, (pos, norm)) in grouped {
                            let mesh = build_mesh(&pos, &norm);
                            let mesh_handle = meshes.add(mesh);
                            let color = material_color(mat_id);
                            let mat_handle = materials.add(StandardMaterial {
                                base_color: color,
                                perceptual_roughness: 0.6,
                                metallic: 0.0,
                                cull_mode: None,
                                ..Default::default()
                            });
                            parent.with_children(|c| {
                                c.spawn((
                                    Mesh3d(mesh_handle.clone()),
                                    MeshMaterial3d(mat_handle.clone()),
                                    Transform::from_translation(Vec3::Y * -0.2),
                                    GlobalTransform::default(),
                                    Visibility::default(),
                                    InheritedVisibility::default(),
                                ));
                            });
                        }
                        vol.spawned = true;
                    }
                }
            }
            Err(err) => {
                warn!("Shape-Z parse error for {:?}: {}", path, err.to_string());
            }
        }
    }
}

fn triangles_grouped_by_material(
    positions: &[[f32; 3]],
    indices: &[u32],
    face_mats: &[u8],
) -> std::collections::BTreeMap<u8, (Vec<[f32; 3]>, Vec<[f32; 3]>)> {
    use std::collections::BTreeMap;
    let mut map: BTreeMap<u8, (Vec<[f32; 3]>, Vec<[f32; 3]>)> = BTreeMap::new();
    for (tri_idx, tri) in indices.chunks_exact(3).enumerate() {
        let mat = face_mats.get(tri_idx).copied().unwrap_or(0);
        let (pos_out, norm_out) = map.entry(mat).or_insert_with(|| (Vec::new(), Vec::new()));
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        let v0 = Vec3::from_array(positions[i0]);
        let v1 = Vec3::from_array(positions[i1]);
        let v2 = Vec3::from_array(positions[i2]);
        let n = (v1 - v0).cross(v2 - v0).normalize_or_zero().to_array();
        pos_out.push(positions[i0]);
        pos_out.push(positions[i1]);
        pos_out.push(positions[i2]);
        norm_out.push(n);
        norm_out.push(n);
        norm_out.push(n);
    }
    map
}

fn build_mesh(out_positions: &[[f32; 3]], out_normals: &[[f32; 3]]) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, out_positions.to_vec());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, out_normals.to_vec());
    mesh
}

fn material_color(id: u8) -> BevyColor {
    // Deterministic palette based on id (simple HSV hue mapping)
    let h = (id as f32 * 0.1618) % 1.0; // golden ratio fraction for spread
    let s = 0.6;
    let v = 0.9;
    hsv_to_rgb(h, s, v)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> BevyColor {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match i as i32 % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    BevyColor::srgb(r, g, b)
}
