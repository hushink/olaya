//! Half-precision inference must stay close to the F32 reference.
//!
//! ```console
//! OLAYA_DTYPE_DEVICE=cuda OLAYA_DTYPE_DTYPE=f16 `
//!   cargo test --release -p olaya --features cuda --test dtype_parity -- --ignored --nocapture
//! ```
//!
//! On CPU both engines compute in F32 (candle's half-precision CPU kernels
//! widen to F32), so the test also documents the CPU fallback.

use std::path::PathBuf;
use std::time::Instant;

use olaya::candle_core::DType;
use olaya::{BuiltinFamilies, Engine, Question, Questions, State};
use serde_json::Value;

fn registry_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../registry")
}

fn guard_questions() -> Questions {
    let question = |value: Value| serde_json::from_value::<Question>(value).expect("guard");
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

fn questions(model: &str) -> Questions {
    if model.starts_with("qwen3guard") {
        return guard_questions();
    }
    Questions::new()
        .noul("billing", "Is this message about billing?")
        .choice_with_instructions(
            "team",
            "Which team should handle this?",
            [
                ("billing", "charges and refunds"),
                ("technical", "bugs and crashes"),
                ("account", "login and profile"),
            ],
        )
        .score_with_instructions("urgency", "How urgent is this?", ["low", "medium", "high"])
}

fn states() -> Vec<State> {
    vec![
        State::text("I was charged twice for my subscription this month. Please refund it."),
        State::text("The app crashes every time I upload a photo to a listing."),
    ]
}

fn max_diff(got: &Value, want: &Value) -> f64 {
    match (got, want) {
        (Value::Number(got), Value::Number(want)) => {
            (got.as_f64().unwrap() - want.as_f64().unwrap()).abs()
        }
        (Value::Object(got), Value::Object(want)) => {
            assert_eq!(got.len(), want.len(), "object keys");
            want.iter()
                .map(|(key, want)| {
                    max_diff(
                        got.get(key).unwrap_or_else(|| panic!("missing {key}")),
                        want,
                    )
                })
                .fold(0.0, f64::max)
        }
        (Value::Array(got), Value::Array(want)) => {
            assert_eq!(got.len(), want.len(), "array length");
            got.iter()
                .zip(want)
                .map(|(got, want)| max_diff(got, want))
                .fold(0.0, f64::max)
        }
        _ => {
            assert_eq!(got, want, "non-numeric answer");
            0.0
        }
    }
}

#[test]
#[ignore = "requires the comparison models under ~/.olaya/models and a CUDA/Metal device"]
fn half_precision_matches_f32() {
    let device = std::env::var("OLAYA_DTYPE_DEVICE").unwrap_or_else(|_| "cuda".to_owned());
    let dtype = match std::env::var("OLAYA_DTYPE_DTYPE").as_deref() {
        Ok("bf16") => DType::BF16,
        _ => DType::F16,
    };
    let tolerance = std::env::var("OLAYA_DTYPE_TOLERANCE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1e-2);
    let models = std::env::var("OLAYA_DTYPE_MODELS").unwrap_or_else(|_| {
        "laya,nli-modernbert,nli-modernbert-base,von,gliclass-instruct-edge,decima-small,decima-base,qwen3guard"
            .to_owned()
    });
    let device = olaya::olaya_candle::device::resolve(&device).expect("device");
    let manifests = olaya::load_manifests(registry_dir()).expect("registry");
    let build = |dtype| {
        Engine::builder()
            .builtin_families(BuiltinFamilies::all())
            .manifests(manifests.clone())
            .device(device.clone())
            .dtype(dtype)
            .build()
            .expect("engine")
    };
    let reference = build(DType::F32);
    let half = build(dtype);
    let states = states();

    for model in models.split(',').map(str::trim).filter(|m| !m.is_empty()) {
        let questions = questions(model);
        reference.load(model).expect("load f32");
        half.load(model).expect("load half");

        let mut reference_answers = Vec::new();
        let mut reference_seconds = f64::MAX;
        for state in &states {
            let start = Instant::now();
            let response = reference.ask(model, state, &questions).expect("ask f32");
            reference_seconds = reference_seconds.min(start.elapsed().as_secs_f64());
            reference_answers.push(serde_json::to_value(&response.answers).expect("json"));
        }

        let mut half_answers = Vec::new();
        let mut half_seconds = f64::MAX;
        for state in &states {
            let start = Instant::now();
            let response = half.ask(model, state, &questions).expect("ask half");
            half_seconds = half_seconds.min(start.elapsed().as_secs_f64());
            half_answers.push(serde_json::to_value(&response.answers).expect("json"));
        }

        let diff = reference_answers
            .iter()
            .zip(&half_answers)
            .map(|(reference, half)| max_diff(half, reference))
            .fold(0.0, f64::max);
        println!(
            "{model}: {dtype:?} max diff {diff:.3e}; f32 {reference_seconds:.3}s, {dtype:?} {half_seconds:.3}s"
        );
        assert!(
            diff <= tolerance,
            "{model}: {dtype:?} max diff {diff:.3e} exceeds {tolerance:.3e}"
        );
        reference.unload(model);
        half.unload(model);
    }
}
