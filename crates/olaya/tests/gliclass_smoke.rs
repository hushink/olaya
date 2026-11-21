use std::path::PathBuf;

use olaya::{Answers, BuiltinFamilies, Engine, Questions, State};

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

fn state() -> State {
    State::text(
        "I was charged twice for my subscription this month. Please refund the extra charge.",
    )
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

#[test]
#[ignore = "requires ~/.olaya/models/gliclass-v4-edge"]
fn gliclass_v4_edge_answers_three_primitives() {
    let response = engine()
        .ask("gliclass-v4-edge", &state(), &questions())
        .unwrap_or_else(|error| panic!("gliclass answers: {error}"));
    println!("gliclass: usage {:?}", response.usage);
    assert_eq!(response.usage.input_tokens, 116);
    let answers = Answers::new(response.answers);
    println!(
        "gliclass: {}",
        serde_json::to_string_pretty(answers.as_map()).unwrap()
    );

    // Reference: ollaya's GLiClass v4-edge ONNX graph (sha256 4c58ea...)
    // through onnxruntime 1.32, with the same tokenizer and rows.
    let noul = answers.noul("billing").expect("noul");
    assert!((noul.noul - 0.887_387_821_378_109_7).abs() < 1e-5);

    let choice = answers.choice("tone").expect("choice");
    assert_eq!(choice.choice, "calm");
    assert!((choice.probabilities["calm"] - 0.913_775_571_563_548_6).abs() < 1e-5);

    let score = answers.score("urgency").expect("score");
    assert!((score.score - 1.061_502_161_141_035_1).abs() < 1e-5);
    assert!((score.probabilities["2"] - 0.399_823_976_169_298_3).abs() < 1e-5);
}

#[test]
#[ignore = "requires ~/.olaya/models/gliclass-large"]
fn gliclass_large_answers_three_primitives() {
    let response = engine()
        .ask("gliclass-large", &state(), &questions())
        .unwrap_or_else(|error| panic!("gliclass-large answers: {error}"));
    println!("gliclass-large: usage {:?}", response.usage);
    assert_eq!(response.usage.input_tokens, 116);
    let answers = Answers::new(response.answers);
    println!(
        "gliclass-large: {}",
        serde_json::to_string_pretty(answers.as_map()).unwrap()
    );

    // Reference: ollaya's GLiClass instruct-large v1.0 ONNX graph
    // (sha256 4e3b56..., segment embeddings + average label pooling) through
    // onnxruntime 1.32, with the same tokenizer and rows.
    let noul = answers.noul("billing").expect("noul");
    assert!((noul.noul - 0.026_434_267_392_750_274).abs() < 1e-5);

    let choice = answers.choice("tone").expect("choice");
    assert_eq!(choice.choice, "angry");
    assert!((choice.probabilities["angry"] - 0.631_565_449_513_389_3).abs() < 1e-5);
    assert!((choice.probabilities["calm"] - 0.368_434_550_486_610_56).abs() < 1e-5);

    let score = answers.score("urgency").expect("score");
    assert!((score.score - 1.001_288_729_533_629_5).abs() < 1e-5);
    assert!((score.probabilities["1"] - 0.896_416_122_063_209_8).abs() < 1e-5);
}

#[test]
#[ignore = "requires ~/.olaya/models/gliclass-instruct-edge"]
fn gliclass_instruct_edge_answers_three_primitives() {
    let response = engine()
        .ask("gliclass-instruct-edge", &state(), &questions())
        .unwrap_or_else(|error| panic!("gliclass-instruct-edge answers: {error}"));
    println!("gliclass-instruct-edge: usage {:?}", response.usage);
    assert_eq!(response.usage.input_tokens, 119);
    let answers = Answers::new(response.answers);
    println!(
        "gliclass-instruct-edge: {}",
        serde_json::to_string_pretty(answers.as_map()).unwrap()
    );

    // Reference: gliclass 0.1.22 (GLiClassModel.from_pretrained, torch fp32)
    // on the same tokenizer ids and rows. The ModernBERT encoder is
    // `jhu-clsp/ettin-encoder-32m`; segment embeddings and average label
    // pooling are folded into the candle path.
    let noul = answers.noul("billing").expect("noul");
    assert!((noul.noul - 0.607_782_309_365_255_7).abs() < 1e-5);

    let choice = answers.choice("tone").expect("choice");
    assert_eq!(choice.choice, "calm");
    assert!((choice.probabilities["angry"] - 0.036_020_400_203_126_56).abs() < 1e-5);
    assert!((choice.probabilities["calm"] - 0.963_979_599_796_873_5).abs() < 1e-5);

    let score = answers.score("urgency").expect("score");
    assert!((score.score - 0.076_142_930_158_248_05).abs() < 1e-5);
    assert!((score.probabilities["0"] - 0.959_770_965_836_859_3).abs() < 1e-5);
    assert!((score.probabilities["1"] - 0.004_315_138_168_033_39).abs() < 1e-5);
    assert!((score.probabilities["2"] - 0.035_913_895_995_107_33).abs() < 1e-5);
}
