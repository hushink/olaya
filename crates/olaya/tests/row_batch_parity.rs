//! Question rows batched into one encoder call must match one call per question.

use std::path::PathBuf;

use olaya::{BuiltinFamilies, Engine, Questions, State};
use serde_json::Value;

fn registry_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../registry")
}

fn engine() -> Engine {
    let manifests = olaya::load_manifests(registry_dir()).expect("registry loads");
    let device = std::env::var("OLAYA_ROW_BATCH_DEVICE").unwrap_or_else(|_| "cpu".to_owned());
    let dtype = match std::env::var("OLAYA_ROW_BATCH_DTYPE").as_deref() {
        Ok("f16") => olaya::candle_core::DType::F16,
        Ok("bf16") => olaya::candle_core::DType::BF16,
        _ => olaya::candle_core::DType::F32,
    };
    Engine::builder()
        .builtin_families(BuiltinFamilies::all())
        .manifests(manifests)
        .device(olaya::olaya_candle::device::resolve(&device).expect("device"))
        .dtype(dtype)
        .build()
        .expect("engine builds")
}

fn questions() -> Questions {
    Questions::new()
        .noul("billing", "Is this message about billing?")
        .choice_with_instructions(
            "tone",
            "What is the tone of this message?",
            [
                ("angry", "An upset or hostile message"),
                ("calm", "A neutral or polite message"),
            ],
        )
        .score_with_instructions(
            "urgency",
            "How urgent is this message?",
            [
                "Can wait",
                "Needs attention this week",
                "Needs attention today",
            ],
        )
}

fn state() -> State {
    State::text(
        "I was charged twice for my subscription this month. Please refund the extra charge.",
    )
}

fn close(label: &str, got: &Value, want: &Value, tolerance: f64) {
    match (got, want) {
        (Value::Number(got), Value::Number(want)) => {
            let got = got.as_f64().expect("number");
            let want = want.as_f64().expect("number");
            assert!(
                (got - want).abs() < tolerance,
                "{label}: {got} != {want} (tolerance {tolerance:.1e})"
            );
        }
        (Value::Object(got), Value::Object(want)) => {
            assert_eq!(got.len(), want.len(), "{label}: keys");
            for (key, want) in want {
                close(
                    &format!("{label}.{key}"),
                    got.get(key)
                        .unwrap_or_else(|| panic!("{label}: missing {key}")),
                    want,
                    tolerance,
                );
            }
        }
        (Value::Array(got), Value::Array(want)) => {
            assert_eq!(got.len(), want.len(), "{label}: length");
            for (index, (got, want)) in got.iter().zip(want).enumerate() {
                close(&format!("{label}[{index}]"), got, want, tolerance);
            }
        }
        _ => assert_eq!(got, want, "{label}"),
    }
}

#[test]
#[ignore = "requires laya, von, and gliclass checkpoints under ~/.olaya/models"]
fn batched_question_rows_match_individual_requests() {
    let engine = engine();
    let state = state();
    let questions = questions();
    // F16/BF16 kernels reduce in a different order for different batch shapes,
    // so half precision needs a wider tolerance than the F32 default.
    let tolerance = std::env::var("OLAYA_ROW_BATCH_TOLERANCE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1e-5);
    for model in ["laya", "von", "gliclass-v4-edge", "gliclass-instruct-edge"] {
        let together = engine
            .ask(model, &state, &questions)
            .unwrap_or_else(|error| panic!("{model}: {error}"));
        for (id, question) in questions.iter() {
            let single = Questions::new().insert(id.clone(), question.clone());
            let alone = engine
                .ask(model, &state, &single)
                .unwrap_or_else(|error| panic!("{model}/{id}: {error}"));
            close(
                &format!("{model}/{id}"),
                &serde_json::to_value(&together.answers[id]).expect("batched json"),
                &serde_json::to_value(&alone.answers[id]).expect("single json"),
                tolerance,
            );
        }
    }
}
