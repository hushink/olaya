#![cfg(feature = "decision")]

use std::path::PathBuf;

use olaya::{Answers, BuiltinFamilies, Engine, Question, Questions, State};
use serde_json::Value;

fn registry_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../registry")
}

fn fixture() -> Vec<Value> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/decision-eos.json");
    serde_json::from_slice(&std::fs::read(path).expect("decision fixture")).expect("fixture json")
}

fn engine() -> Engine {
    let manifests = olaya::load_manifests(registry_dir()).expect("registry loads");
    Engine::builder()
        .builtin_families(BuiltinFamilies::all())
        .manifests(manifests)
        .build()
        .expect("engine builds")
}

fn close(label: &str, got: f64, want: f64) {
    assert!((got - want).abs() < 1e-4, "{label}: {got} != {want}");
}

#[test]
#[ignore = "requires ~/.olaya/models/decision-eos-0.8b"]
fn decision_eos_matches_the_upstream_head() {
    let engine = engine();
    for record in fixture() {
        let id = record["id"].as_str().expect("id");
        let state = State::from_value(record["state"].clone()).expect("state");
        let question: Question =
            serde_json::from_value(record["question"].clone()).expect("question");
        let questions = Questions::new().insert("q", question);
        let response = engine
            .ask("decision-eos-0.8b", &state, &questions)
            .unwrap_or_else(|error| panic!("{id}: {error}"));
        assert_eq!(
            response.usage.input_tokens,
            record["tokens"].as_u64().expect("tokens"),
            "{id}: token count"
        );
        let answers = Answers::new(response.answers);
        let answer = answers.get("q").expect("answer");
        let expected = &record["answer"];
        match answer {
            olaya::Answer::Noul(noul) => {
                close(id, noul.noul, expected["noul"].as_f64().expect("noul"));
            }
            olaya::Answer::Choice(choice) => {
                assert_eq!(
                    choice.choice,
                    expected["choice"].as_str().expect("choice"),
                    "{id}"
                );
                let probabilities = expected["probabilities"]
                    .as_object()
                    .expect("probabilities");
                assert_eq!(choice.probabilities.len(), probabilities.len(), "{id}");
                for (label, want) in probabilities {
                    close(
                        &format!("{id}/{label}"),
                        choice.probabilities[label.as_str()],
                        want.as_f64().expect("probability"),
                    );
                }
            }
            olaya::Answer::Score(score) => {
                close(id, score.score, expected["score"].as_f64().expect("score"));
                let probabilities = expected["probabilities"]
                    .as_object()
                    .expect("probabilities");
                for (label, want) in probabilities {
                    close(
                        &format!("{id}/{label}"),
                        score.probabilities[label.as_str()],
                        want.as_f64().expect("probability"),
                    );
                }
                let legend = expected["legend"].as_object().expect("legend");
                for (index, want) in legend {
                    let value = score.legend.get(index.as_str()).expect("legend entry");
                    assert_eq!(
                        serde_json::to_value(value).expect("legend value"),
                        *want,
                        "{id}/legend/{index}"
                    );
                }
            }
            olaya::Answer::Box(_) => panic!("{id}: box answer"),
        }
    }
}

#[test]
#[ignore = "requires ~/.olaya/models/decision-eos-0.8b"]
fn decision_rejects_invalid_questions_and_long_rows() {
    let engine = engine();
    let state = State::text("hi");

    let missing: Question = serde_json::from_value(serde_json::json!({
        "type": "noul"
    }))
    .expect("question");
    let error = engine
        .ask(
            "decision-eos-0.8b",
            &state,
            &Questions::new().insert("q", missing),
        )
        .expect_err("instructions is required");
    assert!(matches!(
        error,
        olaya::Error::Model {
            source: olaya::RuntimeError::InvalidRequest { .. },
            ..
        }
    ));

    let short_score: Question = serde_json::from_value(serde_json::json!({
        "type": "score", "instructions": "x", "criteria": ["only"]
    }))
    .expect("question");
    assert!(
        engine
            .ask(
                "decision-eos-0.8b",
                &state,
                &Questions::new().insert("q", short_score),
            )
            .is_err()
    );

    let long = State::text("refund ".repeat(40_000));
    let question: Question = serde_json::from_value(serde_json::json!({
        "type": "noul", "instructions": "Is this about billing?"
    }))
    .expect("question");
    let error = engine
        .ask(
            "decision-eos-0.8b",
            &long,
            &Questions::new().insert("q", question),
        )
        .expect_err("the row exceeds 16k tokens");
    assert!(matches!(
        error,
        olaya::Error::Model {
            source: olaya::RuntimeError::InputTooLong { .. },
            ..
        }
    ));
}
