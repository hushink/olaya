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
#[ignore = "requires ~/.olaya/models/nli-modernbert"]
fn nli_answers_three_primitives() {
    let response = engine()
        .ask("nli-modernbert", &state(), &questions())
        .unwrap_or_else(|error| panic!("nli answers: {error}"));
    println!("nli: usage {:?}", response.usage);
    assert_eq!(response.usage.input_tokens, 141);
    let answers = Answers::new(response.answers);
    println!(
        "nli: {}",
        serde_json::to_string_pretty(answers.as_map()).unwrap()
    );

    // Reference: ollaya's ONNX graph (sha256 da223f...) through onnxruntime
    // 1.32, with the same tokenizer and rows. The candle probabilities agree
    // within 3.3e-6.
    let noul = answers.noul("billing").expect("noul");
    assert!((noul.noul - 0.963_568_838_585_794_2).abs() < 1e-5);

    let choice = answers.choice("tone").expect("choice");
    assert_eq!(choice.choice, "angry");
    assert!((choice.confidence - 0.354_393_329_898_636_6).abs() < 1e-5);
    assert_eq!(choice.probabilities.len(), 2);
    assert!((choice.probabilities["angry"] - 0.677_196_664_949_318_3).abs() < 1e-5);
    let sum: f64 = choice.probabilities.values().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "choice probabilities sum to {sum}"
    );

    let score = answers.score("urgency").expect("score");
    assert!((score.score - 1.404_680_618_058_097).abs() < 1e-5);
    assert_eq!(score.probabilities.len(), 3);
    assert!((score.probabilities["2"] - 0.492_181_292_579_042_3).abs() < 1e-5);
    let sum: f64 = score.probabilities.values().sum();
    assert!((sum - 1.0).abs() < 1e-6, "score probabilities sum to {sum}");
}

#[test]
#[ignore = "requires ~/.olaya/models/nli-deberta-v3"]
fn nli_deberta_answers_three_primitives() {
    let response = engine()
        .ask("nli-deberta-v3", &state(), &questions())
        .unwrap_or_else(|error| panic!("nli DeBERTa answers: {error}"));
    println!("nli-deberta: usage {:?}", response.usage);
    assert_eq!(response.usage.input_tokens, 213);
    let answers = Answers::new(response.answers);
    println!(
        "nli-deberta: {}",
        serde_json::to_string_pretty(answers.as_map()).unwrap()
    );

    // Reference: ollaya's DeBERTa ONNX graph (sha256 104fc2...) through
    // onnxruntime 1.32, with the same tokenizer and rows.
    let noul = answers.noul("billing").expect("noul");
    assert!((noul.noul - 0.989_392_700_505_586_1).abs() < 1e-5);

    let choice = answers.choice("tone").expect("choice");
    assert_eq!(choice.choice, "angry");
    assert!((choice.confidence - 0.698_538_250_572_816_6).abs() < 1e-5);
    assert!((choice.probabilities["angry"] - 0.849_269_125_286_408_3).abs() < 1e-5);

    let score = answers.score("urgency").expect("score");
    assert!((score.score - 1.065_667_988_949_624_6).abs() < 1e-5);
    assert!((score.probabilities["2"] - 0.408_832_802_338_881_53).abs() < 1e-5);
}
