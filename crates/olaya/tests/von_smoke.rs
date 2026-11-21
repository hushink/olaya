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
#[ignore = "requires ~/.olaya/models/von"]
fn von_answers_three_primitives() {
    let response = engine()
        .ask("von", &state(), &questions())
        .unwrap_or_else(|error| panic!("von answers: {error}"));
    println!("von: usage {:?}", response.usage);
    assert_eq!(response.usage.input_tokens, 144);
    let answers = Answers::new(response.answers);
    println!(
        "von: {}",
        serde_json::to_string_pretty(answers.as_map()).unwrap()
    );

    // Reference: the upstream `option_marker.pt` network (gliclass/von-sdk
    // weights) in PyTorch fp32 on the CPU, with the same tokenizer and rows.
    let noul = answers.noul("billing").expect("noul");
    assert!((noul.noul - 0.751_221_488_846_809_4).abs() < 1e-5);

    let choice = answers.choice("tone").expect("choice");
    assert_eq!(choice.choice, "calm");
    assert!((choice.probabilities["angry"] - 0.418_791_487_783_842_73).abs() < 1e-5);
    assert!((choice.probabilities["calm"] - 0.581_208_512_216_157_2).abs() < 1e-5);

    let score = answers.score("urgency").expect("score");
    assert!((score.score - 1.599_888_771_511_035_5).abs() < 1e-5);
    assert!((score.probabilities["1"] - 0.257_988_912_293_183_77).abs() < 1e-5);
    assert!((score.probabilities["2"] - 0.670_949_929_608_925_9).abs() < 1e-5);
}
