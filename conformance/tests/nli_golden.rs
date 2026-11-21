use std::path::PathBuf;

use olaya::{BuiltinFamilies, Engine};
use olaya_conformance::{NLI_PROBABILITY_TOLERANCE, fixture_path, load_goldens, run_golden};

const MODELS: [&str; 3] = ["nli-modernbert", "nli-modernbert-base", "nli-deberta-v3"];

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
#[ignore = "requires the NLI models under ~/.olaya/models"]
fn nli_matches_the_transformer_goldens() {
    let engine = build_engine();
    for model in MODELS {
        let path = fixture_path("nli", model);
        let goldens = load_goldens(&path).expect("goldens load");
        assert!(!goldens.is_empty(), "{} is empty", path.display());
        for golden in &goldens {
            run_golden(&engine, golden, NLI_PROBABILITY_TOLERANCE)
                .unwrap_or_else(|error| panic!("{model} {}: {error}", golden.id));
        }
        println!("{model}: {} goldens", goldens.len());
    }
}
