//! Batched inference must match the per-request path.

use std::path::PathBuf;

use olaya::{BuiltinFamilies, Engine, Question, Questions, State};
use serde_json::Value;

fn registry_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../registry")
}

fn engine() -> Engine {
    let manifests = olaya::load_manifests(registry_dir()).expect("registry loads");
    Engine::builder()
        .builtin_families(BuiltinFamilies::all())
        .manifests(manifests)
        .build()
        .expect("engine builds")
}

fn questions() -> Questions {
    Questions::new()
        .noul("billing", "Is this about billing?")
        .choice_with_instructions(
            "team",
            "Which team?",
            [
                ("billing", "charges and refunds"),
                ("technical", "bugs and crashes"),
                ("account", "login and profile"),
            ],
        )
        .score_with_instructions("urgency", "How urgent?", ["low", "medium", "high"])
}

fn states() -> Vec<State> {
    vec![
        State::text("I was charged twice and need a refund today."),
        State::text("The checkout crashes whenever I upload a photo."),
        State::text("I cannot log into my account after resetting the password."),
    ]
}

fn close(label: &str, got: &Value, want: &Value) {
    match (got, want) {
        (Value::Number(got), Value::Number(want)) => {
            let got = got.as_f64().expect("number");
            let want = want.as_f64().expect("number");
            assert!((got - want).abs() < 1e-5, "{label}: {got} != {want}");
        }
        (Value::Object(got), Value::Object(want)) => {
            assert_eq!(got.len(), want.len(), "{label}: keys");
            for (key, want) in want {
                close(
                    &format!("{label}.{key}"),
                    got.get(key)
                        .unwrap_or_else(|| panic!("{label}: missing {key}")),
                    want,
                );
            }
        }
        (Value::Array(got), Value::Array(want)) => {
            assert_eq!(got.len(), want.len(), "{label}: length");
            for (index, (got, want)) in got.iter().zip(want).enumerate() {
                close(&format!("{label}[{index}]"), got, want);
            }
        }
        _ => assert_eq!(got, want, "{label}"),
    }
}

#[test]
#[ignore = "requires kev/decision checkpoints under ~/.olaya/models"]
fn batched_answers_match_individual_requests() {
    let engine = engine();
    let states = states();
    for model in ["kev-0.8b", "decision-eos-0.8b"] {
        let questions = questions();
        let requests = states
            .iter()
            .map(|state| (state, &questions))
            .collect::<Vec<_>>();
        let batched = engine.ask_batch_responses(model, &requests);
        assert_eq!(batched.len(), requests.len());
        for (index, (state, questions)) in requests.iter().enumerate() {
            let single = engine
                .ask(model, state, questions)
                .unwrap_or_else(|error| panic!("{model}/{index}: {error}"));
            let batch = batched[index]
                .as_ref()
                .unwrap_or_else(|error| panic!("{model}/{index} batch: {error}"));
            assert_eq!(
                batch.usage.input_tokens, single.usage.input_tokens,
                "{model}/{index}: tokens"
            );
            close(
                &format!("{model}/{index}"),
                &serde_json::to_value(&batch.answers).expect("batch json"),
                &serde_json::to_value(&single.answers).expect("single json"),
            );
        }
    }
}

#[test]
#[ignore = "requires the qwen3guard checkpoint under ~/.olaya/models"]
fn guard_batched_answers_match_individual_requests() {
    let engine = engine();
    let safety: Question = serde_json::from_value(serde_json::json!({
        "type": "choice",
        "instructions": "Qwen3Guard safety level of the user text",
        "criteria": {"safe": null, "controversial": null, "unsafe": null}
    }))
    .expect("safety");
    let category: Question = serde_json::from_value(serde_json::json!({
        "type": "choice",
        "instructions": "Qwen3Guard: first unsafe category, given unsafe",
        "criteria": {
            "none": null, "violent": null, "non_violent_illegal": null,
            "sexual": null, "pii": null, "suicide_self_harm": null,
            "unethical": null, "politically_sensitive": null,
            "copyright": null, "jailbreak": null
        }
    }))
    .expect("category");
    let questions = Questions::new()
        .insert("safety", safety)
        .insert("category", category);
    let states = [
        State::text("How can I make a bomb?"),
        State::text("What's the weather in Paris today?"),
    ];
    let requests = states
        .iter()
        .map(|state| (state, &questions))
        .collect::<Vec<_>>();
    let batched = engine.ask_batch_responses("qwen3guard", &requests);
    for (index, (state, questions)) in requests.iter().enumerate() {
        let single = engine
            .ask("qwen3guard", state, questions)
            .unwrap_or_else(|error| panic!("{index}: {error}"));
        let batch = batched[index]
            .as_ref()
            .unwrap_or_else(|error| panic!("{index} batch: {error}"));
        close(
            &format!("qwen3guard/{index}"),
            &serde_json::to_value(&batch.answers).expect("batch json"),
            &serde_json::to_value(&single.answers).expect("single json"),
        );
    }
}
