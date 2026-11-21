#![cfg(feature = "kev")]

use std::path::PathBuf;

use olaya::{Answers, BuiltinFamilies, Engine, Question, Questions, State};
use serde_json::Value;

fn registry_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../registry")
}

fn fixture() -> Vec<Value> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kev.json");
    serde_json::from_slice(&std::fs::read(path).expect("kev fixture")).expect("fixture json")
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
#[ignore = "requires ~/.olaya/models/kev-0.8b"]
fn kev_matches_the_upstream_pointer_head() {
    let engine = engine();
    for record in fixture() {
        let id = record["id"].as_str().expect("id");
        let state = State::from_value(record["state"].clone()).expect("state");
        let mut questions = Questions::new();
        for (qid, question) in record["questions"].as_object().expect("questions") {
            let question: Question = serde_json::from_value(question.clone()).expect("question");
            questions = questions.insert(qid.clone(), question);
        }
        let expected_tokens = record["input_tokens"].as_u64().expect("input_tokens");
        let response = engine
            .ask("kev-0.8b", &state, &questions)
            .unwrap_or_else(|error| panic!("{id}: {error}"));
        assert_eq!(
            response.usage.input_tokens, expected_tokens,
            "{id}: token count"
        );
        let answers = Answers::new(response.answers);
        for (qid, expected) in record["answers"].as_object().expect("answers") {
            let answer = answers.get(qid).expect("answer");
            match answer {
                olaya::Answer::Noul(noul) => {
                    close(
                        &format!("{id}/{qid}"),
                        noul.noul,
                        expected["noul"].as_f64().expect("noul"),
                    );
                }
                olaya::Answer::Choice(choice) => {
                    assert_eq!(
                        choice.choice,
                        expected["choice"].as_str().expect("choice"),
                        "{id}/{qid}"
                    );
                    close(
                        &format!("{id}/{qid}/confidence"),
                        choice.confidence,
                        expected["confidence"].as_f64().expect("confidence"),
                    );
                    let probabilities = expected["probabilities"]
                        .as_object()
                        .expect("probabilities");
                    assert_eq!(
                        choice.probabilities.len(),
                        probabilities.len(),
                        "{id}/{qid}"
                    );
                    for (label, want) in probabilities {
                        close(
                            &format!("{id}/{qid}/{label}"),
                            choice.probabilities[label.as_str()],
                            want.as_f64().expect("probability"),
                        );
                    }
                }
                olaya::Answer::Score(score) => {
                    close(
                        &format!("{id}/{qid}/score"),
                        score.score,
                        expected["score"].as_f64().expect("score"),
                    );
                    close(
                        &format!("{id}/{qid}/confidence"),
                        score.confidence,
                        expected["confidence"].as_f64().expect("confidence"),
                    );
                    let probabilities = expected["probabilities"]
                        .as_object()
                        .expect("probabilities");
                    for (label, want) in probabilities {
                        close(
                            &format!("{id}/{qid}/{label}"),
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
                            "{id}/{qid}/legend/{index}"
                        );
                    }
                }
                olaya::Answer::Box(_) => panic!("{id}/{qid}: box answer"),
            }
        }
    }
}

#[test]
#[ignore = "requires ~/.olaya/models/kev-0.8b"]
fn kev_rejects_box_truncated_and_overlong_rows() {
    let engine = engine();
    let state = State::text("hi");
    let question: Question = serde_json::from_value(serde_json::json!({
        "type": "box", "instructions": "find it", "criteria": {"label": "thing"}
    }))
    .expect("question");
    let error = engine
        .ask("kev-0.8b", &state, &Questions::new().insert("q", question))
        .expect_err("box questions are unsupported");
    assert!(matches!(
        error,
        olaya::Error::Model {
            source: olaya::RuntimeError::InvalidRequest { .. },
            ..
        }
    ));

    let long_state = State::text("refund ".repeat(9_000));
    let question: Question = serde_json::from_value(serde_json::json!({
        "type": "noul", "instructions": "Is this about billing?"
    }))
    .expect("question");
    let error = engine
        .ask(
            "kev-0.8b",
            &long_state,
            &Questions::new().insert("q", question),
        )
        .expect_err("the state exceeds 8192 tokens");
    assert!(
        matches!(
            error,
            olaya::Error::Model {
                source: olaya::RuntimeError::StateTruncated { .. },
                ..
            }
        ),
        "unexpected error: {error}"
    );

    let question: Question = serde_json::from_value(serde_json::json!({
        "type": "noul", "instructions": "word ".repeat(9_000)
    }))
    .expect("question");
    let error = engine
        .ask("kev-0.8b", &state, &Questions::new().insert("q", question))
        .expect_err("the row exceeds 8192 tokens");
    assert!(matches!(
        error,
        olaya::Error::Model {
            source: olaya::RuntimeError::InputTooLong { .. },
            ..
        }
    ));
}
