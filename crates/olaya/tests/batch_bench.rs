//! Cross-request batching benchmark.
//!
//! ```console
//! OLAYA_BATCH_MODEL=qwen3guard OLAYA_BATCH_DEVICE=cuda OLAYA_BATCH_DTYPE=f16 `
//!   cargo test --release -p olaya --features cuda --test batch_bench -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use olaya::{BuiltinFamilies, Engine, Question, Questions, State};

fn registry_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../registry")
}

fn guard_questions() -> Questions {
    let question = |value: serde_json::Value| {
        serde_json::from_value::<Question>(value).expect("guard question")
    };
    Questions::new()
        .insert(
            "safety",
            question(serde_json::json!({
                "type": "choice",
                "instructions": "Qwen3Guard safety level of the user text",
                "criteria": {"safe": null, "controversial": null, "unsafe": null}
            })),
        )
        .insert(
            "category",
            question(serde_json::json!({
                "type": "choice",
                "instructions": "Qwen3Guard: first unsafe category, given unsafe",
                "criteria": {
                    "none": null, "violent": null, "non_violent_illegal": null,
                    "sexual": null, "pii": null, "suicide_self_harm": null,
                    "unethical": null, "politically_sensitive": null,
                    "copyright": null, "jailbreak": null
                }
            })),
        )
}

#[test]
#[ignore = "timing harness: requires a model under ~/.olaya/models"]
fn cross_request_batching_scales() {
    let model = std::env::var("OLAYA_BATCH_MODEL").unwrap_or_else(|_| "qwen3guard".to_owned());
    let device = std::env::var("OLAYA_BATCH_DEVICE").unwrap_or_else(|_| "cuda".to_owned());
    let dtype = match std::env::var("OLAYA_BATCH_DTYPE").as_deref() {
        Ok("f16") => olaya::candle_core::DType::F16,
        Ok("bf16") => olaya::candle_core::DType::BF16,
        _ => olaya::candle_core::DType::F32,
    };
    let count = std::env::var("OLAYA_BATCH_REQUESTS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16);
    let manifests = olaya::load_manifests(registry_dir()).expect("registry");
    let engine = Engine::builder()
        .builtin_families(BuiltinFamilies::all())
        .manifests(manifests)
        .device(olaya::olaya_candle::device::resolve(&device).expect("device"))
        .dtype(dtype)
        .build()
        .expect("engine");
    let questions = guard_questions();
    let states = (0..count)
        .map(|index| State::text(format!("probe {index}: how do I reset my password?")))
        .collect::<Vec<_>>();
    engine.load(&model).expect("load");
    let warm = states
        .iter()
        .take(2)
        .map(|state| (state, &questions))
        .collect::<Vec<_>>();
    let _ = engine.ask_batch_responses(&model, &warm);

    let start = Instant::now();
    for state in &states {
        engine.ask(&model, state, &questions).expect("ask");
    }
    let sequential = start.elapsed().as_secs_f64();

    let requests = states
        .iter()
        .map(|state| (state, &questions))
        .collect::<Vec<_>>();
    let start = Instant::now();
    let batched = engine.ask_batch_responses(&model, &requests);
    let batched_seconds = start.elapsed().as_secs_f64();
    assert!(batched.iter().all(Result::is_ok), "batched run failed");

    println!(
        "{model} ({device}, {dtype:?}): {count} requests sequential {sequential:.3}s, batched {batched_seconds:.3}s ({:.1}x)",
        sequential / batched_seconds
    );
}
