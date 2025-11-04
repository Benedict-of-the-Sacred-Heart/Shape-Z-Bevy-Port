use std::{
    collections::BTreeSet,
    env, fs, io,
    path::{Path, PathBuf},
    time::Instant,
};

use shapezlib::shapez::ShapeZ;

fn main() {
    let options = match parse_options() {
        Ok(opts) => opts,
        Err(ParseOutcome::Help) => {
            print_usage();
            return;
        }
        Err(ParseOutcome::Error(msg)) => {
            eprintln!("{msg}");
            print_usage();
            return;
        }
    };

    let Some(scene_path) = find_lighthouse_scene() else {
        eprintln!(
            "Could not locate 'examples/lighthouse.shpz'. Run the example from the workspace root."
        );
        return;
    };

    println!("Loading Shape-Z lighthouse scene from {scene_path:?}");

    let output_dir = output_dir();
    if let Err(err) = fs::create_dir_all(&output_dir) {
        eprintln!("Failed to create output directory {:?}: {err}", output_dir);
        return;
    }
    println!("Outputs will be written to {:?}", output_dir);

    let mut engine = ShapeZ::default();

    let parse_started = Instant::now();
    let module = match engine.parse(scene_path.clone()) {
        Ok(module) => module,
        Err(err) => {
            eprintln!("Parse error: {err}");
            return;
        }
    };
    println!("Parsed module in {:.2?}", parse_started.elapsed());

    let compile_started = Instant::now();
    if let Err(err) = engine.compile(&module) {
        eprintln!("Compile error: {err}");
        return;
    }
    println!("Compiled module in {:.2?}", compile_started.elapsed());

    let execute_started = Instant::now();
    engine.execute();
    println!(
        "Executed voxel program in {:.2?}",
        execute_started.elapsed()
    );

    let (voxels, memory) = engine.stats();
    println!("Scene stats: {voxels}, {memory}");

    let (positions, indices, face_materials) = engine.mesh_triangles();
    let triangle_count = indices.len() / 3;
    let unique_materials = face_materials
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .len();
    println!(
        "Mesh summary: {} vertices, {} triangles, {} materials",
        positions.len(),
        triangle_count,
        unique_materials
    );

    if options.write_obj {
        println!(
            "Writing OBJ + MTL and moving them into {:?} ...",
            output_dir
        );
        engine.write_obj();

        let obj_path = scene_path.with_extension("obj");
        let mtl_path = scene_path.with_extension("mtl");

        match move_into_dir(&obj_path, &output_dir) {
            Ok(dest) => println!("Moved OBJ to {:?}", dest),
            Err(err) => eprintln!("Failed to move {:?}: {err}", obj_path),
        }

        match move_into_dir(&mtl_path, &output_dir) {
            Ok(dest) => println!("Moved MTL to {:?}", dest),
            Err(err) => eprintln!("Failed to move {:?}: {err}", mtl_path),
        }
    }

    if let Some(iterations) = options.render_iterations {
        let png_path = scene_path.with_extension("png");
        println!(
            "Rendering {iterations} iteration(s) of the progressive renderer, output will end up in {:?} ...",
            output_dir
        );
        for i in 0..iterations {
            engine.sample();
            if iterations <= 10 || (i + 1) % 10 == 0 || i + 1 == iterations {
                println!("  sample {}/{}", i + 1, iterations);
            }
        }
        engine.write_image();

        match move_into_dir(&png_path, &output_dir) {
            Ok(dest) => println!("Moved PNG to {:?}", dest),
            Err(err) => eprintln!("Failed to move {:?}: {err}", png_path),
        }
    }

    println!("Done.");
}

struct Options {
    write_obj: bool,
    render_iterations: Option<usize>,
}

enum ParseOutcome {
    Help,
    Error(String),
}

fn parse_options() -> Result<Options, ParseOutcome> {
    let mut write_obj = false;
    let mut render_iterations = None;

    for arg in env::args().skip(1) {
        if matches!(arg.as_str(), "--help" | "-h") {
            return Err(ParseOutcome::Help);
        }

        if arg == "--write-obj" {
            write_obj = true;
            continue;
        }

        if let Some(value) = arg.strip_prefix("--render=") {
            match value.parse::<usize>() {
                Ok(iterations) if iterations > 0 => {
                    render_iterations = Some(iterations);
                }
                _ => {
                    return Err(ParseOutcome::Error(format!(
                        "Invalid value for --render: '{value}'. Use a positive integer."
                    )));
                }
            }
            continue;
        }

        return Err(ParseOutcome::Error(format!("Unrecognized argument: {arg}")));
    }

    Ok(Options {
        write_obj,
        render_iterations,
    })
}

fn find_lighthouse_scene() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("../examples/lighthouse.shpz"),
        manifest_dir.join("../../examples/lighthouse.shpz"),
        PathBuf::from("examples/lighthouse.shpz"),
    ];

    for candidate in candidates {
        if candidate.exists() {
            return Some(candidate);
        }
    }

    None
}

fn print_usage() {
    println!("Usage: cargo run -p shapezlib --example lighthouse [--write-obj] [--render=N]");
    println!("\nFlags:");
    println!(
        "  --write-obj   Export an OBJ+MTL next to the .shpz scene (overwrites existing file)."
    );
    println!(
        "  --render=N    Run N progressive render iterations and overwrite the PNG next to the scene."
    );
}

fn output_dir() -> PathBuf {
    workspace_root().join("pure-rust-example")
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

fn move_into_dir(src: &Path, dir: &Path) -> io::Result<PathBuf> {
    if !src.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Source file {:?} does not exist", src),
        ));
    }

    let Some(name) = src.file_name() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Source path {:?} has no filename", src),
        ));
    };

    let dest = dir.join(name);
    if dest.exists() {
        fs::remove_file(&dest)?;
    }

    fs::rename(src, &dest)?;
    Ok(dest)
}
