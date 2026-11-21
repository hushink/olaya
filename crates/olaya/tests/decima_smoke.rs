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
#[ignore = "requires ~/.olaya/models/decima-small"]
fn decima_small_answers_three_primitives() {
    let response = engine()
        .ask("decima-small", &state(), &questions())
        .unwrap_or_else(|error| panic!("decima answers: {error}"));
    println!("decima: usage {:?}", response.usage);
    assert_eq!(response.usage.input_tokens, 197);
    let answers = Answers::new(response.answers);
    println!(
        "decima: {}",
        serde_json::to_string_pretty(answers.as_map()).unwrap()
    );

    // Reference: upstream decima 2.3.0 (`Decima.decide`) on the same PyTorch
    // checkpoint, tokenizer and rows.
    let noul = answers.noul("billing").expect("noul");
    assert!((noul.noul - 0.762_467_861_175_537_1).abs() < 1e-5);

    let choice = answers.choice("tone").expect("choice");
    assert_eq!(choice.choice, "angry");
    assert!((choice.probabilities["angry"] - 0.592_934_131_622_314_5).abs() < 1e-5);
    assert!((choice.probabilities["calm"] - 0.407_065_838_575_363_16).abs() < 1e-5);

    let score = answers.score("urgency").expect("score");
    assert!((score.score - 1.229_308_784_008_026_1).abs() < 1e-5);
    assert!((score.probabilities["0"] - 0.132_349_610_328_674_32).abs() < 1e-5);
    assert!((score.probabilities["1"] - 0.505_991_935_729_980_5).abs() < 1e-5);
    assert!((score.probabilities["2"] - 0.361_658_424_139_022_8).abs() < 1e-5);
}

fn urgency_choice(ordinal: bool) -> Questions {
    let mut value = serde_json::json!({
        "type": "choice",
        "instructions": "How urgent is this message?",
        "criteria": {
            "low": "Can wait",
            "mid": "Needs attention this week",
            "high": "Needs attention today"
        }
    });
    if ordinal {
        value["ordinal"] = serde_json::Value::Bool(true);
    }
    let question: olaya::Question = serde_json::from_value(value).expect("question");
    Questions::new().insert("urgency", question)
}

/// Compare an ordinal choice with upstream decima 2.3.0's PyTorch fp32
/// probabilities for the same request.
fn assert_ordinal_reference(model: &str, expected: [f64; 3]) {
    let response = engine()
        .ask(model, &state(), &urgency_choice(true))
        .unwrap_or_else(|error| panic!("{model} ordinal answers: {error}"));
    let choice = Answers::new(response.answers)
        .choice("urgency")
        .expect("choice")
        .clone();
    println!("{model} ordinal: {:?}", choice.probabilities);
    for (label, want) in ["low", "mid", "high"].into_iter().zip(expected) {
        assert!(
            (choice.probabilities[label] - want).abs() < 1e-5,
            "{model} {label}: {} != {want}",
            choice.probabilities[label]
        );
    }
}

#[test]
#[ignore = "requires ~/.olaya/models/decima-small"]
fn decima_small_reads_ordinal_choices_with_the_ordinal_head() {
    let engine = engine();
    let ordinal = engine
        .ask("decima-small", &state(), &urgency_choice(true))
        .expect("ordinal choice");
    let plain = engine
        .ask("decima-small", &state(), &urgency_choice(false))
        .expect("plain choice");

    let ordinal = Answers::new(ordinal.answers)
        .choice("urgency")
        .expect("choice")
        .clone();
    let plain = Answers::new(plain.answers)
        .choice("urgency")
        .expect("choice")
        .clone();
    println!("decima-small ordinal: {:?}", ordinal.probabilities);
    println!("decima-small plain:   {:?}", plain.probabilities);

    for (label, want) in [
        ("low", 0.146_130_666_136_741_64),
        ("mid", 0.523_808_836_936_950_7),
        ("high", 0.330_060_422_420_501_7),
    ] {
        assert!(
            (ordinal.probabilities[label] - want).abs() < 1e-5,
            "decima-small {label}: {} != {want}",
            ordinal.probabilities[label]
        );
    }

    assert_eq!(ordinal.probabilities.len(), 3);
    let total: f64 = ordinal.probabilities.values().sum();
    assert!((total - 1.0).abs() < 1e-9);
    let top = ordinal
        .probabilities
        .iter()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("top")
        .0;
    assert_eq!(&ordinal.choice, top);
    assert!(
        (ordinal.confidence - olaya::typesafe_confidence(&ordinal.probabilities)).abs() < 1e-12
    );

    // The ordinal head reads the same rows differently from the plain softmax.
    let difference = ordinal
        .probabilities
        .iter()
        .map(|(label, probability)| (probability - plain.probabilities[label]).abs())
        .fold(0.0f64, f64::max);
    assert!(
        difference > 1e-3,
        "ordinal and plain probabilities are identical"
    );
}

#[test]
#[ignore = "requires ~/.olaya/models/decima-small"]
fn decima_small_reports_a_truncated_state() {
    let long = State::text("refund ".repeat(5000));
    let questions = Questions::new().noul("billing", "Is this message about billing?");
    let error = engine()
        .ask("decima-small", &long, &questions)
        .expect_err("the state exceeds decima-small's 512-token window");
    match error {
        olaya::Error::Model {
            model,
            source: olaya::RuntimeError::StateTruncated { model: truncated },
        } => {
            assert_eq!(model, "decima-small");
            assert_eq!(truncated, "decima-small");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
#[ignore = "requires ~/.olaya/models/decima-base"]
fn decima_base_answers_three_primitives() {
    let response = engine()
        .ask("decima-base", &state(), &questions())
        .unwrap_or_else(|error| panic!("decima-base answers: {error}"));
    println!("decima-base: usage {:?}", response.usage);
    let answers = Answers::new(response.answers);
    println!(
        "decima-base: {}",
        serde_json::to_string_pretty(answers.as_map()).unwrap()
    );

    assert_eq!(response.usage.input_tokens, 162);
    let noul = answers.noul("billing").expect("noul");
    assert!((noul.noul - 0.964_643_239_974_975_6).abs() < 1e-5);

    let choice = answers.choice("tone").expect("choice");
    assert_eq!(choice.choice, "angry");
    assert!((choice.probabilities["angry"] - 0.725_826_978_683_471_7).abs() < 1e-5);
    assert!((choice.probabilities["calm"] - 0.274_173_021_316_528_3).abs() < 1e-5);

    let score = answers.score("urgency").expect("score");
    assert!((score.score - 1.267_062_664_031_982_4).abs() < 1e-5);
    assert!((score.probabilities["0"] - 0.069_595_985_114_574_43).abs() < 1e-5);
    assert!((score.probabilities["1"] - 0.593_745_350_837_707_5).abs() < 1e-5);
    assert!((score.probabilities["2"] - 0.336_658_656_597_137_45).abs() < 1e-5);

    assert_ordinal_reference(
        "decima-base",
        [
            0.053_077_772_259_712_22,
            0.505_119_323_730_468_8,
            0.441_802_918_910_980_2,
        ],
    );
}

#[test]
#[ignore = "requires ~/.olaya/models/decima-guard-small"]
fn decima_guard_small_answers_three_primitives() {
    let response = engine()
        .ask("decima-guard-small", &state(), &questions())
        .unwrap_or_else(|error| panic!("decima-guard-small answers: {error}"));
    println!("decima-guard-small: usage {:?}", response.usage);
    let answers = Answers::new(response.answers);
    println!(
        "decima-guard-small: {}",
        serde_json::to_string_pretty(answers.as_map()).unwrap()
    );

    assert_eq!(response.usage.input_tokens, 162);
    let noul = answers.noul("billing").expect("noul");
    assert!((noul.noul - 0.934_252_262_115_478_5).abs() < 1e-5);

    let choice = answers.choice("tone").expect("choice");
    assert_eq!(choice.choice, "calm");
    assert!((choice.probabilities["angry"] - 0.202_638_387_680_053_7).abs() < 1e-5);
    assert!((choice.probabilities["calm"] - 0.797_361_612_319_946_3).abs() < 1e-5);

    let score = answers.score("urgency").expect("score");
    assert!((score.score - 0.581_901_073_455_810_5).abs() < 1e-5);
    assert!((score.probabilities["0"] - 0.535_275_101_661_682_1).abs() < 1e-5);
    assert!((score.probabilities["1"] - 0.347_548_753_023_147_6).abs() < 1e-5);
    assert!((score.probabilities["2"] - 0.117_176_160_216_331_48).abs() < 1e-5);

    assert_ordinal_reference(
        "decima-guard-small",
        [
            0.593_711_018_562_316_9,
            0.314_358_860_254_287_7,
            0.091_930_113_732_814_79,
        ],
    );
}

#[test]
#[ignore = "requires ~/.olaya/models/decima-guard"]
fn decima_guard_answers_three_primitives() {
    let response = engine()
        .ask("decima-guard", &state(), &questions())
        .unwrap_or_else(|error| panic!("decima-guard answers: {error}"));
    println!("decima-guard: usage {:?}", response.usage);
    let answers = Answers::new(response.answers);
    println!(
        "decima-guard: {}",
        serde_json::to_string_pretty(answers.as_map()).unwrap()
    );

    assert_eq!(response.usage.input_tokens, 162);
    let noul = answers.noul("billing").expect("noul");
    assert!((noul.noul - 0.976_327_657_699_585).abs() < 1e-5);

    let choice = answers.choice("tone").expect("choice");
    assert_eq!(choice.choice, "calm");
    assert!((choice.probabilities["angry"] - 0.151_424_035_429_954_53).abs() < 1e-5);
    assert!((choice.probabilities["calm"] - 0.848_575_949_668_884_3).abs() < 1e-5);

    let score = answers.score("urgency").expect("score");
    assert!((score.probabilities["0"] - 0.315_286_755_561_828_6).abs() < 1e-5);
    assert!((score.probabilities["1"] - 0.447_556_853_294_372_56).abs() < 1e-5);
    assert!((score.probabilities["2"] - 0.237_156_450_748_443_6).abs() < 1e-5);

    assert_ordinal_reference(
        "decima-guard",
        [
            0.379_689_693_450_927_73,
            0.421_016_544_103_622_44,
            0.199_293_777_346_611_02,
        ],
    );
}

#[test]
#[ignore = "requires ~/.olaya/models/decima-agent"]
fn decima_agent_answers_three_primitives() {
    let response = engine()
        .ask("decima-agent", &state(), &questions())
        .unwrap_or_else(|error| panic!("decima-agent answers: {error}"));
    println!("decima-agent: usage {:?}", response.usage);
    let answers = Answers::new(response.answers);
    println!(
        "decima-agent: {}",
        serde_json::to_string_pretty(answers.as_map()).unwrap()
    );

    assert_eq!(response.usage.input_tokens, 162);
    let noul = answers.noul("billing").expect("noul");
    assert!((noul.noul - 0.968_333_244_323_730_5).abs() < 1e-5);

    let choice = answers.choice("tone").expect("choice");
    assert_eq!(choice.choice, "angry");
    assert!((choice.probabilities["angry"] - 0.712_745_726_108_551).abs() < 1e-5);
    assert!((choice.probabilities["calm"] - 0.287_254_273_891_449).abs() < 1e-5);

    let score = answers.score("urgency").expect("score");
    assert!((score.probabilities["0"] - 0.119_372_896_850_109_1).abs() < 1e-5);
    assert!((score.probabilities["1"] - 0.786_098_361_015_319_8).abs() < 1e-5);
    assert!((score.probabilities["2"] - 0.094_528_794_288_635_25).abs() < 1e-5);

    assert_ordinal_reference(
        "decima-agent",
        [
            0.159_509_271_383_285_52,
            0.780_840_218_067_169_2,
            0.059_650_421_142_578_125,
        ],
    );
}

#[test]
#[ignore = "requires ~/.olaya/models/decima-base-clean"]
fn decima_base_clean_answers_three_primitives() {
    let response = engine()
        .ask("decima-base-clean", &state(), &questions())
        .unwrap_or_else(|error| panic!("decima-base-clean answers: {error}"));
    println!("decima-base-clean: usage {:?}", response.usage);
    let answers = Answers::new(response.answers);
    println!(
        "decima-base-clean: {}",
        serde_json::to_string_pretty(answers.as_map()).unwrap()
    );

    assert_eq!(response.usage.input_tokens, 162);
    let noul = answers.noul("billing").expect("noul");
    assert!((noul.noul - 0.975_634_515_285_491_9).abs() < 1e-5);

    let choice = answers.choice("tone").expect("choice");
    assert_eq!(choice.choice, "calm");
    assert!((choice.probabilities["angry"] - 0.297_314_703_464_508_06).abs() < 1e-5);
    assert!((choice.probabilities["calm"] - 0.702_685_296_535_491_9).abs() < 1e-5);

    let score = answers.score("urgency").expect("score");
    assert!((score.probabilities["0"] - 0.189_079_940_319_061_28).abs() < 1e-5);
    assert!((score.probabilities["1"] - 0.544_243_454_933_166_5).abs() < 1e-5);
    assert!((score.probabilities["2"] - 0.266_676_634_550_094_6).abs() < 1e-5);

    assert_ordinal_reference(
        "decima-base-clean",
        [
            0.206_639_230_251_312_26,
            0.540_393_471_717_834_5,
            0.252_967_298_030_853_27,
        ],
    );
}
