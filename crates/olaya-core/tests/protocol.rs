use std::collections::BTreeMap;

use olaya_core::{
    Calibration, CalibrationFile, CalibrationMode, Capabilities, Content, FileRole, ManifestFile,
    Question, RepoManifest, RouterConfig, answer_from_logits, error_type, parse_request_value,
};
use serde_json::json;

fn example_request() -> serde_json::Value {
    json!({
        "model": "jev-latest",
        "state": "I was charged twice for my subscription this month.",
        "questions": {
            "billing": {
                "type": "noul",
                "instructions": "Is this message about billing?"
            },
            "tone": {
                "type": "choice",
                "instructions": "What is the tone of this message?",
                "criteria": {
                    "angry": "An upset or hostile message",
                    "calm": "A neutral or polite message"
                }
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this message?",
                "criteria": ["Can wait", "Needs attention this week", "Needs attention today"]
            }
        }
    })
}

#[test]
fn noul_criteria_keys_are_case_insensitive() {
    let question: Question = serde_json::from_value(json!({
        "type": "noul",
        "instructions": "Angry?",
        "criteria": {
            "TRUE": "yes",
            "False": "no",
            "true": "still yes"
        }
    }))
    .expect("question parses");
    let Question::Noul(noul) = question else {
        panic!("expected a noul question");
    };
    let criteria = noul.criteria.expect("criteria");
    assert_eq!(criteria.is_true, Some(Content::String("still yes".into())));
    assert_eq!(criteria.is_false, Some(Content::String("no".into())));
}

#[test]
fn parses_the_official_examples() {
    let request = parse_request_value(example_request()).expect("request parses");
    assert_eq!(request.model, "jev-latest");
    assert_eq!(request.questions.len(), 3);
    assert_eq!(request.questions["billing"].num_options(), 2);
    assert_eq!(request.questions["tone"].num_options(), 2);
    assert_eq!(request.questions["urgency"].num_options(), 3);
    assert!(request.validate().is_ok());
}

#[test]
fn missing_instructions_stay_none() {
    let request = parse_request_value(json!({
        "model": "m",
        "state": "s",
        "questions": {
            "topic": {"type": "choice", "criteria": {"a": null, "b": null}}
        }
    }))
    .expect("request parses");
    let Question::Choice(choice) = &request.questions["topic"] else {
        panic!("choice");
    };
    assert!(choice.instructions.is_none());
}

#[test]
fn instructions_fall_back_to_the_question_id() {
    let question = |value| -> Question { serde_json::from_value(value).expect("question") };
    assert_eq!(
        question(json!({"type": "noul"})).instructions_or_id("qid"),
        "qid"
    );
    assert_eq!(
        question(json!({"type": "noul", "instructions": null})).instructions_or_id("qid"),
        "qid"
    );
    assert_eq!(
        question(json!({"type": "noul", "instructions": " text "})).instructions_or_id("qid"),
        " text "
    );
    assert_eq!(
        question(json!({"type": "noul", "instructions": {"a": 1}})).instructions_or_id("qid"),
        "{\"a\": 1}"
    );
}

#[test]
fn ordinal_choice_is_opt_in() {
    let request = parse_request_value(json!({
        "model": "m",
        "state": "s",
        "questions": {
            "rank": {"type": "choice", "ordinal": true,
                     "criteria": {"low": null, "mid": null, "high": null}}
        }
    }))
    .expect("request parses");
    let Question::Choice(choice) = &request.questions["rank"] else {
        panic!("choice");
    };
    assert!(choice.ordinal);
    assert!(request.questions["rank"].is_ordinal());
    assert!(request.validate().is_ok());

    let plain = parse_request_value(json!({
        "model": "m",
        "state": "s",
        "questions": {"topic": {"type": "choice", "criteria": {"a": null, "b": null}}}
    }))
    .expect("request parses");
    assert!(!plain.questions["topic"].is_ordinal());
    assert_eq!(
        serde_json::to_value(&plain.questions["topic"]).expect("json"),
        json!({"type": "choice", "criteria": {"a": null, "b": null}})
    );
}

#[test]
fn rejects_ordinal_choices_outside_two_to_ten_labels() {
    let one = parse_request_value(json!({
        "model": "m",
        "state": "s",
        "questions": {"q": {"type": "choice", "ordinal": true, "criteria": {"a": null}}}
    }))
    .expect_err("one label is not ordinal");
    assert_eq!(one.errors()[0].kind, error_type::VALUE_ERROR);

    let eleven = (0..11)
        .map(|index| (format!("l{index}"), json!(null)))
        .collect::<serde_json::Map<_, _>>();
    let too_many = parse_request_value(json!({
        "model": "m",
        "state": "s",
        "questions": {"q": {"type": "choice", "ordinal": true, "criteria": eleven}}
    }))
    .expect_err("eleven labels are not ordinal");
    assert_eq!(too_many.errors()[0].kind, error_type::VALUE_ERROR);
}

#[test]
fn rejects_too_many_choice_options() {
    let criteria = (0..256)
        .map(|i| (format!("option-{i}"), json!(null)))
        .collect::<serde_json::Map<_, _>>();
    let err = parse_request_value(json!({
        "model": "m",
        "state": "s",
        "questions": {"q": {"type": "choice", "criteria": criteria}}
    }))
    .expect_err("256 options are rejected");
    assert_eq!(err.errors()[0].kind, error_type::TOO_LONG);
    assert_eq!(
        err.errors()[0].loc[2],
        olaya_core::LocSegment::Key("q".into())
    );
}

#[test]
fn rejects_too_many_score_levels() {
    let err = parse_request_value(json!({
        "model": "m",
        "state": "s",
        "questions": {"q": {"type": "score", "criteria": ["l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8", "l9", "l10"]}}
    }))
    .expect_err("11 levels are rejected");
    assert_eq!(err.errors()[0].kind, error_type::TOO_LONG);
}

#[test]
fn rejects_top_level_numbers_as_state() {
    let err = parse_request_value(json!({
        "model": "m",
        "state": 42,
        "questions": {"q": {"type": "noul", "instructions": "?"}}
    }))
    .expect_err("numbers are not state");
    assert_eq!(err.errors()[0].kind, error_type::STATE_TYPE);
}

#[test]
fn answers_serialize_in_typesafe_order() {
    let question: Question = serde_json::from_value(json!({
        "type": "choice",
        "instructions": "Which?",
        "criteria": {"a": null, "b": null, "c": null}
    }))
    .unwrap();
    let calibration = Calibration::from_file(&CalibrationFile::default(), CalibrationMode::Raw);
    let answer = answer_from_logits(
        &question,
        &["a".into(), "b".into(), "c".into()],
        &[],
        &[2.0, 0.5, -1.0],
        &calibration,
        10,
    )
    .unwrap();
    let value = serde_json::to_value(&answer).unwrap();
    assert_eq!(value["type"], "choice");
    assert_eq!(value["choice"], "a");
    assert!(value["probabilities"]["a"].as_f64().unwrap() > 0.7);
    assert!(value.get("confidence").is_some());
}

#[test]
fn noul_answer_is_the_true_probability() {
    let question: Question =
        serde_json::from_value(json!({"type": "noul", "instructions": "?"})).unwrap();
    let calibration = Calibration::from_file(
        &CalibrationFile {
            temperature: vec![json!(1.0)],
            ..CalibrationFile::default()
        },
        CalibrationMode::Clamp,
    );
    let answer = answer_from_logits(
        &question,
        &["false".into(), "true".into()],
        &[],
        &[-10.0, 10.0],
        &calibration,
        1,
    )
    .unwrap();
    let value = serde_json::to_value(&answer).unwrap();
    assert_eq!(value["type"], "noul");
    assert!(value["noul"].as_f64().unwrap() > 0.99);
    assert!(value.get("confidence").is_none());
}

#[test]
fn raw_and_clamped_calibration_differ() {
    let file: CalibrationFile = serde_json::from_value(json!({
        "temperature": [0.1, 1.0, 1.0]
    }))
    .unwrap();
    let raw = Calibration::from_file(&file, CalibrationMode::Raw);
    let clamped = Calibration::from_file(&file, CalibrationMode::Clamp);
    assert!((raw.temperature(olaya_core::QType::Choice, 2) - 0.1).abs() < 1e-12);
    assert!((clamped.temperature(olaya_core::QType::Choice, 2) - 0.5).abs() < 1e-12);
}

fn manifest() -> RepoManifest {
    RepoManifest {
        schema_version: 1,
        name: "laya".into(),
        aliases: vec!["laya:en".into()],
        family: "laya".into(),
        engine: "candle".into(),
        arch: "modernbert".into(),
        format: "safetensors".into(),
        layout: "laya-markers-v1".into(),
        subfolder: None,
        files: vec![ManifestFile {
            path: "model.safetensors".into(),
            role: FileRole::Weights,
            sha256: Some("a".repeat(64)),
            size: Some(842_609_210),
            url: None,
        }],
        calibration_mode: CalibrationMode::Clamp,
        capabilities: Capabilities::classification(),
        router: None,
        license: "Apache-2.0".into(),
        description: "Laya decision model".into(),
        release_date: Some("2026-09-15".into()),
        source: None,
        revision: None,
    }
}

#[test]
fn manifest_validates() {
    assert!(manifest().validate().is_ok());
}

#[test]
fn manifest_rejects_parent_paths() {
    let mut manifest = manifest();
    manifest.files[0].path = "../model.safetensors".into();
    assert!(manifest.validate().is_err());
}

#[test]
fn manifest_accepts_a_subfolder() {
    let mut manifest = manifest();
    manifest.subfolder = Some("multilingual".into());
    assert!(manifest.validate().is_ok());
}

#[test]
fn manifest_rejects_parent_subfolder() {
    let mut manifest = manifest();
    manifest.subfolder = Some("../multilingual".into());
    assert!(manifest.validate().is_err());
}

#[test]
fn manifest_rejects_unknown_engine() {
    let mut manifest = manifest();
    manifest.engine = "onnx".into();
    assert!(manifest.validate().is_err());
}

#[test]
fn manifest_rejects_alias_equal_to_name() {
    let mut manifest = manifest();
    manifest.aliases = vec!["laya".into()];
    assert!(manifest.validate().is_err());
}

fn router_manifest() -> RepoManifest {
    let mut manifest = manifest();
    manifest.name = "laya-latest".into();
    manifest.aliases = vec!["laya:latest".into()];
    manifest.family = "router".into();
    manifest.engine = "router".into();
    manifest.arch = "router".into();
    manifest.format = "router".into();
    manifest.layout = "router".into();
    manifest.files.clear();
    manifest.router = Some(RouterConfig {
        strategy: "script".into(),
        routes: BTreeMap::from([
            ("english".into(), "laya".into()),
            ("multilingual".into(), "laya-multilingual".into()),
        ]),
        default: "laya".into(),
    });
    manifest
}

#[test]
fn router_manifest_validates_without_files() {
    assert!(router_manifest().validate().is_ok());
}

#[test]
fn router_manifest_requires_routes() {
    let mut manifest = router_manifest();
    manifest.router.as_mut().expect("router").routes.clear();
    assert!(manifest.validate().is_err());
}

#[test]
fn router_engine_requires_a_config() {
    let mut manifest = router_manifest();
    manifest.router = None;
    assert!(manifest.validate().is_err());
}

#[test]
fn router_config_requires_the_router_engine() {
    let mut manifest = manifest();
    manifest.router = Some(RouterConfig {
        strategy: "script".into(),
        routes: BTreeMap::from([("english".into(), "laya".into())]),
        default: "laya".into(),
    });
    assert!(manifest.validate().is_err());
}
