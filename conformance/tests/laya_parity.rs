use std::path::PathBuf;

use olaya::{Answers, BuiltinFamilies, Engine, Questions, State};

fn registry_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../registry")
}

fn build_engine() -> Engine {
    let manifests = olaya::load_manifests(registry_dir()).expect("registry loads");
    Engine::builder()
        .builtin_families(BuiltinFamilies::all())
        .manifests(manifests)
        .build()
        .expect("engine builds")
}

#[test]
#[ignore = "requires ~/.olaya/models/laya and a running ollaya reference for regeneration"]
fn laya_matches_the_ollaya_ort_reference() {
    let engine = build_engine();

    let state = State::text(
        "I was charged twice for my subscription this month. Please refund the extra charge.",
    );
    let questions = Questions::new()
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
        );

    let response = engine
        .ask("laya", &state, &questions)
        .expect("laya answers");
    assert_eq!(response.model, "laya");
    assert_eq!(response.usage.input_tokens, 147);

    let answers = Answers::new(response.answers);
    let billing = answers.noul("billing").expect("noul");
    assert!((billing.noul - 0.9416).abs() < 1e-3, "{billing:?}");

    let tone = answers.choice("tone").expect("choice");
    assert_eq!(tone.choice, "angry");
    assert!((tone.probabilities["angry"] - 0.7229).abs() < 1e-3);
    assert!((tone.probabilities["calm"] - 0.2771).abs() < 1e-3);

    let urgency = answers.score("urgency").expect("score");
    assert!((urgency.score - 1.8162).abs() < 1e-3, "{urgency:?}");
    assert!((urgency.probabilities["2"] - 0.8379).abs() < 1e-3);
}

#[test]
#[ignore = "requires ~/.olaya/models/laya-multilingual and a running ollaya reference"]
fn laya_multilingual_matches_the_ollaya_ort_reference() {
    let engine = build_engine();

    let english_state = State::text(
        "I was charged twice for my subscription this month. Please refund the extra charge.",
    );
    let english_questions = Questions::new()
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
        );
    let response = engine
        .ask("laya-multilingual", &english_state, &english_questions)
        .expect("multilingual answers");
    assert_eq!(response.usage.input_tokens, 148);
    let answers = Answers::new(response.answers);
    assert!((answers.noul("billing").expect("noul").noul - 0.901).abs() < 1e-3);
    let tone = answers.choice("tone").expect("choice");
    assert_eq!(tone.choice, "angry");
    assert!((tone.probabilities["angry"] - 0.669).abs() < 1e-3);
    let urgency = answers.score("urgency").expect("score");
    assert!((urgency.score - 1.6519).abs() < 1e-3, "{urgency:?}");

    let chinese_state = State::text("我这个月被重复扣款了，请退还多扣的费用。");
    let chinese_questions = Questions::new()
        .noul("billing", "这条消息是关于账单的吗？")
        .choice_with_instructions(
            "tone",
            "这条消息的语气是什么？",
            [("angry", "愤怒或敌对的消息"), ("calm", "中性或礼貌的消息")],
        )
        .score_with_instructions(
            "urgency",
            "这条消息有多紧急？",
            ["可以等待", "本周需要处理", "今天需要处理"],
        );
    let response = engine
        .ask("laya-multilingual", &chinese_state, &chinese_questions)
        .expect("multilingual Chinese answers");
    assert_eq!(response.usage.input_tokens, 156);
    let answers = Answers::new(response.answers);
    assert!((answers.noul("billing").expect("noul").noul - 0.6051).abs() < 1e-3);
    let tone = answers.choice("tone").expect("choice");
    assert_eq!(tone.choice, "calm");
    assert!((tone.probabilities["calm"] - 0.6699).abs() < 1e-3);
    let urgency = answers.score("urgency").expect("score");
    assert!((urgency.score - 1.4134).abs() < 1e-3, "{urgency:?}");
}

#[test]
#[ignore = "requires ~/.olaya/models/laya and laya-multilingual"]
fn laya_latest_routes_by_script() {
    let engine = build_engine();

    let english = engine
        .ask(
            "laya-latest",
            &State::text(
                "I was charged twice for my subscription this month. Please refund the extra charge.",
            ),
            &Questions::new().noul("billing", "Is this message about billing?"),
        )
        .expect("English routed");
    assert_eq!(english.model, "laya");
    let english_answers = Answers::new(english.answers);
    assert!((english_answers.noul("billing").expect("noul").noul - 0.9416).abs() < 1e-3);

    let chinese = engine
        .ask(
            "laya-latest",
            &State::text("我这个月被重复扣款了，请退还多扣的费用。"),
            &Questions::new().noul("billing", "这条消息是关于账单的吗？"),
        )
        .expect("Chinese routed");
    assert_eq!(chinese.model, "laya-multilingual");
    let chinese_answers = Answers::new(chinese.answers);
    assert!((chinese_answers.noul("billing").expect("noul").noul - 0.6051).abs() < 1e-3);
}
