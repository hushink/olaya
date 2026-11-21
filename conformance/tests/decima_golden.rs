use std::path::PathBuf;

use olaya::{BuiltinFamilies, Engine};
use olaya_conformance::{PROBABILITY_TOLERANCE, fixture_path, load_goldens, run_golden};

const MODELS: [&str; 6] = [
    "decima-small",
    "decima-base",
    "decima-base-clean",
    "decima-agent",
    "decima-guard",
    "decima-guard-small",
];

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
#[ignore = "requires the decima models under ~/.olaya/models"]
fn decima_matches_the_upstream_goldens() {
    let engine = build_engine();
    for model in MODELS {
        let path = fixture_path("decima", model);
        let goldens = load_goldens(&path).expect("goldens load");
        assert!(!goldens.is_empty(), "{} is empty", path.display());
        for golden in &goldens {
            run_golden(&engine, golden, PROBABILITY_TOLERANCE)
                .unwrap_or_else(|error| panic!("{model} {}: {error}", golden.id));
        }
        println!("{model}: {} goldens", goldens.len());
    }
}
