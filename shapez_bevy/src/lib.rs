use bevy::{
    prelude::*,
    render::mesh::{Indices, Mesh, PrimitiveTopology},
};
use shapezlib::prelude::*;

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
        Self { path: path.into(), spawned: false }
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
        if vol.spawned { continue; }

        let path = vol.path.clone();
        let mut engine = ShapeZ::default();
        match engine.parse(path.clone()) {
            Ok(module) => {
                if engine.compile(&module).is_ok() {
                    engine.execute();
                    let (positions, indices, _mats) = engine.mesh_triangles();
                    if !positions.is_empty() && !indices.is_empty() {
                        let mut mesh = triangles_to_mesh(&positions, &indices);
                        let mesh_handle = meshes.add(mesh);
                        let mat_handle = materials.add(StandardMaterial::default());
                        cmds.entity(entity).insert(PbrBundle {
                            mesh: mesh_handle,
                            material: mat_handle,
                            ..Default::default()
                        });
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

fn triangles_to_mesh(positions: &[[f32; 3]], indices: &[u32]) -> Mesh {
    let mut normals = vec![[0.0f32; 3]; positions.len()];

    for tri in indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        let v0 = glam::Vec3::from(positions[i0]);
        let v1 = glam::Vec3::from(positions[i1]);
        let v2 = glam::Vec3::from(positions[i2]);
        let n = (v1 - v0).cross(v2 - v0);
        for i in [i0, i1, i2] { 
            let a = &mut normals[i];
            a[0] += n.x; a[1] += n.y; a[2] += n.z; 
        }
    }
    for n in &mut normals {
        let v = glam::Vec3::from(*n);
        let nn = if v.length_squared() > 0.0 { v.normalize() } else { glam::Vec3::Z };
        *n = nn.into();
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions.to_vec());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.set_indices(Some(Indices::U32(indices.to_vec())));
    mesh
}
