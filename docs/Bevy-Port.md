# Shape‑Z → Bevy Port Plan

## Goals

- Provide a pure‑Rust scene API (no external DSL) that maps 1:1 to Shape‑Z concepts.
- First‑class Bevy integration: ECS components, systems, and a plugin to generate meshes or progressive renders in‑engine.
- Maintain or improve current performance and feature parity (patterns, recursion, volumetrics, materials).
- Enable gradual migration from existing `.shpz` assets.

## Scope & Targets

- Bevy 0.17/0.18 (PBR, Assets, RenderApp extract/prepare/queue).
- Platforms: macOS, Linux, Windows.
- Outputs: Bevy Mesh (polygonized), optional progressive renderer to Bevy Texture.

## Architecture Overview

- shapez_core (new): core data structures and algorithms
  - VoxelGrid, Tile storage, polygonization, math utils, noise.
  - Execution state and SceneBuilder (initially bridging to NodeOp, later direct Rust).
- shapez_bevy (new): Bevy plugin & ECS integration
  - Systems to execute scenes, convert grids → Mesh, and publish textures.
  - Components: `ShapeZScene`, `ShapeZMode`, `ShapeZSettings`.
- shapez_cli (existing): keep for backwards compat; deprecate DSL over time.

## Public API (Rust)

```rust
// High-level sketch (subject to change)
let scene = Scene::new()
    .config(Config::default().density(120))
    .camera(Camera::isometric().center([0.0, 0.6, 0.0]).scale(28.0))
    .material("CircleWhite", |m| m.albedo([0.92, 0.95, 1.0]).roughness(0.08))
    .voxel("Circle", |v| {
        v.size([3.0, 0.4, 3.0])
         .disc(|d| d.radius(1.15).floor_depth(0.35).fill(|c| c.use_material("CircleWhite")))
    })
    .place("Circle", [0.0, 0.0, 0.0]);
```

In Bevy:
```rust
app.add_plugins(DefaultPlugins)
   .add_plugins(ShapeZPlugin)
   .add_systems(Startup, |mut cmds: Commands, mut scenes: ResMut<Assets<ShapeZScene>>| {
        let handle = scenes.add(ShapeZScene::from(scene));
        cmds.spawn((ShapeZVolume { scene: handle, mode: ShapeZMode::Mesh }, Transform::default(), GlobalTransform::default()));
   });
```

## Bevy Integration

- `ShapeZPlugin` registers resources, asset types, and systems:
  - build_voxel_scene: executes scene when dirty → `VoxelGrid` cache
  - voxel_grid_to_mesh: converts grid into `bevy::render::mesh::Mesh`
  - sample_renderer (optional): progressive rendering into `Image`
- Extract/Prepare/Queue: rely on Bevy’s standard mesh/material pipelines.

## Migration Strategy

1. Phase 0: Skeleton plugin and docs (this file). Baseline benchmarks.
2. Phase 1: Rust `SceneBuilder` that generates NodeOps, invoking current Execution VM.
3. Phase 2: Bevy mesh export pipeline (VoxelGrid → Mesh) + example app.
4. Phase 3: Materials parity: map to `StandardMaterial` or custom shaders.
5. Phase 4: Progressive renderer ↔ Bevy texture pathway; editor/watch mode.
6. Phase 5: Replace VM with pure‑Rust direct execution; remove NodeOp.

## Testing & Parity

- Golden scenes: bathroom, lighthouse, bottle, circle.
- Regression: image diffs (renderer), OBJ diffs (polygonizer), voxel counts, timing.
- CI: run benchmarks; gate regressions beyond threshold.

## Contribution Workflow

- Branch naming: `bevy/<feature>`, `core/<refactor>`, `docs/<topic>`.
- PRs: small, focused, with screenshots/metrics. Link golden diffs.
- Issue labels: `phase-1`, `phase-2`, … for roadmap tracking.

## Risks & Mitigations

- Feature parity gaps → keep VM bridge until Phase 5 completes.
- Performance regressions → benchmark tiles/meshes per second; optimize hotspots.
- API churn → stabilize after Phase 2; version with `-alpha` tags.

## Initial Milestones

- M0: Docs + plugin skeleton proposal
- M1: `SceneBuilder` → NodeOp bridge, compile/run from Rust
- M2: VoxelGrid → Bevy Mesh converter, spawn from ECS
- M3: Materials mapping, lighting validation
- M4: Progressive renderer to texture
- M5: Pure‑Rust execution (no NodeOp), deprecate DSL

## Open Questions

- Material graphs: closures vs. node‑graph builder? (closures first; graph later)
- Asset hot‑reload: dynamic library scenes or serialized scene assets?
- Volumetrics: PBR integration path in Bevy (custom pipeline?)
