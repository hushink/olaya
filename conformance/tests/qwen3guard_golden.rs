use std::path::PathBuf;

use olaya::{BuiltinFamilies, Engine};
use olaya_conformance::{QWEN3GUARD_PROBABILITY_TOLERANCE, fixture_path, load_goldens, run_golden};

fn build_engine() -> Engine {
    let registry = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../registry");
    let manifests = olaya::load_manifests(registry).expect("registry loads");
    Engine::builder()
        .builtin_families(BuiltinFamilies::all())
        .manifests(manifests)
        .build()
        .expect("engine builds")
}

#[test]
#[ignore = "requires ~/.olaya/models/qwen3guard"]
fn qwen3guard_matches_the_upstream_goldens() {
    let engine = build_engine();
    let path = fixture_path("qwen3guard", "qwen3guard");
    let goldens = load_goldens(&path).expect("goldens load");
    assert!(!goldens.is_empty(), "{} is empty", path.display());
    for golden in &goldens {
        run_golden(&engine, golden, QWEN3GUARD_PROBABILITY_TOLERANCE)
            .unwrap_or_else(|error| panic!("{}: {error}", golden.id));
    }
    println!("qwen3guard: {} goldens", goldens.len());
}
