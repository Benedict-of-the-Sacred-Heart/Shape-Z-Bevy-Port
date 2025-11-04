use bevy::prelude::*;
use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use shapez_bevy::{ShapeZPlugin, ShapeZVolume};
use std::f32::consts::FRAC_PI_2;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window { title: "Shape-Z Bevy Viewer".into(), ..default() }),
            ..default()
        }))
        .add_plugins(ShapeZPlugin)
        .add_systems(Startup, (spawn_camera_light, spawn_shapez))
        .add_systems(Update, orbit_camera_controls)
        .run();
}

fn spawn_camera_light(mut cmds: Commands) {
    let orbit = OrbitCamera::new(Vec3::ZERO, 8.0, -45_f32.to_radians(), 30_f32.to_radians());
    let transform = orbit.to_transform();

    cmds.spawn((
        Camera3d::default(),
        Transform::from(transform),
        GlobalTransform::from(transform),
        Visibility::default(),
        InheritedVisibility::default(),
        orbit,
    ));

    cmds.spawn((
        DirectionalLight { shadows_enabled: true, illuminance: 20000.0, ..default() },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, 0.8, 0.0)),
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
    ));
}

fn spawn_shapez(mut cmds: Commands) {
    // Use the existing example scene included in the repo
    let path = std::path::PathBuf::from("examples/circle.shpz");
    cmds.spawn((
        ShapeZVolume::new(path),
        Transform::default(),
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
    ));
}

#[derive(Component, Clone, Copy)]
struct OrbitCamera {
    focus: Vec3,
    radius: f32,
    yaw: f32,
    pitch: f32,
}

impl OrbitCamera {
    fn new(focus: Vec3, radius: f32, yaw: f32, pitch: f32) -> Self {
        Self { focus, radius, yaw, pitch }
    }

    fn to_transform(&self) -> Transform {
        let rotation = Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0);
        let offset = rotation * Vec3::new(0.0, 0.0, self.radius);
        let translation = self.focus + offset;
        let mut transform = Transform::from_translation(translation);
        transform.look_at(self.focus, Vec3::Y);
        transform
    }

    fn sync_transform(&self, transform: &mut Transform) {
        *transform = self.to_transform();
    }
}

fn orbit_camera_controls(
    mut motion_events: EventReader<MouseMotion>,
    mut scroll_events: EventReader<MouseWheel>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut query: Query<(&mut OrbitCamera, &mut Transform)>,
) {
    let mut rotation_delta = Vec2::ZERO;
    let mut pan_delta = Vec2::ZERO;

    for ev in motion_events.read() {
        if buttons.pressed(MouseButton::Right) {
            rotation_delta += ev.delta;
        } else if buttons.pressed(MouseButton::Middle) {
            pan_delta += ev.delta;
        }
    }

    let mut scroll_delta = 0.0;
    for ev in scroll_events.read() {
        let scroll_amount = match ev.unit {
            MouseScrollUnit::Line => ev.y * 0.25,
            MouseScrollUnit::Pixel => ev.y * 0.01,
        };
        scroll_delta += scroll_amount;
    }

    if rotation_delta == Vec2::ZERO && pan_delta == Vec2::ZERO && scroll_delta.abs() <= f32::EPSILON {
        return;
    }

    if let Ok((mut orbit, mut transform)) = query.single_mut() {
        if rotation_delta != Vec2::ZERO {
            let sensitivity = 0.005;
            orbit.yaw -= rotation_delta.x * sensitivity;
            orbit.pitch -= rotation_delta.y * sensitivity;
            orbit.pitch = orbit.pitch.clamp(-FRAC_PI_2 + 0.01, FRAC_PI_2 - 0.01);
        }

        if pan_delta != Vec2::ZERO {
            let pan_speed = orbit.radius * 0.0015;
            let right = transform.rotation * Vec3::X;
            let up = transform.rotation * Vec3::Y;
            orbit.focus += (-pan_delta.x * pan_speed) * right;
            orbit.focus += (pan_delta.y * pan_speed) * up;
        }

        if scroll_delta.abs() > f32::EPSILON {
            let zoom_factor = 1.0 - scroll_delta * 0.2;
            orbit.radius = (orbit.radius * zoom_factor).clamp(0.5, 200.0);
        }

        orbit.sync_transform(&mut transform);
    }
}
