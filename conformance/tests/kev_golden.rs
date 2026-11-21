use std::path::PathBuf;

use olaya::{BuiltinFamilies, Engine};
use olaya_conformance::{KEV_PROBABILITY_TOLERANCE, fixture_path, load_goldens, run_golden};

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
#[ignore = "requires ~/.olaya/models/kev-0.8b"]
fn kev_matches_the_upstream_goldens() {
    let engine = build_engine();
    let path = fixture_path("kev", "kev-0.8b");
    let goldens = load_goldens(&path).expect("goldens load");
    assert!(!goldens.is_empty(), "{} is empty", path.display());
    for golden in &goldens {
        run_golden(&engine, golden, KEV_PROBABILITY_TOLERANCE)
            .unwrap_or_else(|error| panic!("{}: {error}", golden.id));
    }
    println!("kev-0.8b: {} goldens", goldens.len());
}
