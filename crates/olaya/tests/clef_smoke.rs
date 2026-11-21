#![cfg(feature = "clef")]

use std::path::PathBuf;

use olaya::{Answers, BuiltinFamilies, Engine, Question, Questions, State};
use serde_json::Value;

fn registry_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../registry")
}

fn fixture() -> Vec<Value> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/clef.json");
    serde_json::from_slice(&std::fs::read(path).expect("clef fixture")).expect("fixture json")
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
    // The upstream release is served in bf16; the candle trunk runs fp32.
    assert!((got - want).abs() < 5e-3, "{label}: {got} != {want}");
}

#[test]
#[ignore = "requires ~/.olaya/models/clef-flash (18 GB)"]
fn clef_matches_the_upstream_joint_head() {
    let engine = engine();
    for record in fixture() {
        let id = record["id"].as_str().expect("id");
        let state = State::from_value(record["state"].clone()).expect("state");
        let mut questions = Questions::new();
        for (qid, question) in record["questions"].as_object().expect("questions") {
            let question: Question = serde_json::from_value(question.clone()).expect("question");
            questions = questions.insert(qid.clone(), question);
        }
        let expected_tokens = record["tokens"].as_u64().expect("tokens");
        let response = engine
            .ask("clef-flash", &state, &questions)
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
                        expected["true"].as_f64().expect("true"),
                    );
                }
                olaya::Answer::Choice(choice) => {
                    let probabilities = expected.as_object().expect("probabilities");
                    let best = probabilities
                        .iter()
                        .max_by(|left, right| {
                            left.1
                                .as_f64()
                                .unwrap()
                                .total_cmp(&right.1.as_f64().unwrap())
                        })
                        .expect("probabilities")
                        .0;
                    assert_eq!(choice.choice, *best, "{id}/{qid}");
                    close(
                        &format!("{id}/{qid}/confidence"),
                        choice.confidence,
                        expected[best].as_f64().expect("probability"),
                    );
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
                    let probabilities = expected.as_object().expect("probabilities");
                    let mut want_score = 0.0;
                    for (index, want) in probabilities {
                        let probability = want.as_f64().expect("probability");
                        want_score += index.parse::<usize>().expect("index") as f64 * probability;
                        close(
                            &format!("{id}/{qid}/{index}"),
                            score.probabilities[index.as_str()],
                            probability,
                        );
                    }
                    close(&format!("{id}/{qid}/score"), score.score, want_score);
                }
                olaya::Answer::Box(_) => panic!("{id}/{qid}: box answer"),
            }
        }
    }
}

#[test]
#[ignore = "requires ~/.olaya/models/clef-flash (18 GB)"]
fn clef_rejects_box_truncated_and_overlong_schemas() {
    let engine = engine();
    let state = State::text("hi");
    let question: Question = serde_json::from_value(serde_json::json!({
        "type": "box", "instructions": "find it"
    }))
    .expect("question");
    let error = engine
        .ask(
            "clef-flash",
            &state,
            &Questions::new().insert("q", question),
        )
        .expect_err("box questions are unsupported");
    assert!(matches!(
        error,
        olaya::Error::Model {
            source: olaya::RuntimeError::InvalidRequest { .. },
            ..
        }
    ));

    let long_state = State::text("refund ".repeat(17_000));
    let question: Question = serde_json::from_value(serde_json::json!({
        "type": "noul", "instructions": "Is this about billing?"
    }))
    .expect("question");
    let error = engine
        .ask(
            "clef-flash",
            &long_state,
            &Questions::new().insert("q", question),
        )
        .expect_err("the state exceeds the 16384-token row");
    assert!(matches!(
        error,
        olaya::Error::Model {
            source: olaya::RuntimeError::StateTruncated { .. },
            ..
        }
    ));

    let question: Question = serde_json::from_value(serde_json::json!({
        "type": "noul", "instructions": "word ".repeat(20_000)
    }))
    .expect("question");
    let error = engine
        .ask(
            "clef-flash",
            &state,
            &Questions::new().insert("q", question),
        )
        .expect_err("the schema alone exceeds the row limit");
    assert!(matches!(
        error,
        olaya::Error::Model {
            source: olaya::RuntimeError::InputTooLong { .. },
            ..
        }
    ));
}
