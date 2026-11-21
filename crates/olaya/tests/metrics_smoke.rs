//! Engine metrics and the shared runtime handle.

use std::path::PathBuf;

use olaya::{BuiltinFamilies, DecisionRuntime, Engine, Questions, State};

fn registry_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../registry")
}

#[test]
#[ignore = "requires ~/.olaya/models/laya"]
fn engine_records_request_load_and_unload_metrics() {
    let manifests = olaya::load_manifests(registry_dir()).expect("registry");
    let engine = Engine::builder()
        .builtin_families(BuiltinFamilies::all())
        .manifests(manifests)
        .build()
        .expect("engine");

    engine.load("laya").expect("load laya");
    let loaded = engine.metrics();
    assert_eq!(loaded.model_loads_total, 1);
    assert_eq!(loaded.loaded_models, 1);
    assert!(loaded.model_load_seconds.count >= 1);

    let state = State::text("I was charged twice and want a refund.");
    let questions = Questions::new().noul("billing", "Is this about billing?");
    engine.ask("laya", &state, &questions).expect("ask");
    let after = engine.metrics();
    assert_eq!(after.requests_total, 1);
    assert_eq!(after.request_errors_total, 0);
    assert!(after.input_tokens_total > 0);
    assert_eq!(after.inference_seconds.count, 1);

    // The serve layer uses this shared handle for HTTP counters.
    let handle = DecisionRuntime::runtime_metrics(&engine).expect("engine exposes metrics");
    let shared = handle.snapshot();
    assert_eq!(shared.requests_total, after.requests_total);
    handle.http_request_started();
    handle.http_request_finished(false, 0.01);
    let from_engine = engine.metrics();
    assert_eq!(from_engine.http_requests_total, 1);
    assert_eq!(from_engine.http_request_errors_total, 0);

    assert!(engine.unload("laya"));
    let unloaded = engine.metrics();
    assert_eq!(unloaded.model_unloads_total, 1);
    assert_eq!(unloaded.loaded_models, 0);
}
