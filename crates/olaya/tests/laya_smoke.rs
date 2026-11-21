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

fn ask(model: &str) -> Answers {
    let response = engine()
        .ask(model, &state(), &questions())
        .unwrap_or_else(|error| panic!("{model} answers: {error}"));
    println!("{model}: usage {:?}", response.usage);
    let answers = Answers::new(response.answers);
    println!(
        "{model}: {}",
        serde_json::to_string_pretty(answers.as_map()).unwrap()
    );
    assert!(answers.noul("billing").is_ok());
    assert!(answers.choice("tone").is_ok());
    assert!(answers.score("urgency").is_ok());
    answers
}

#[test]
#[ignore = "requires ~/.olaya/models/laya"]
fn laya_answers_three_primitives() {
    ask("laya");
}

#[test]
#[ignore = "requires ~/.olaya/models/laya"]
fn laya_reports_a_truncated_state() {
    let long = State::text("refund ".repeat(20_000));
    let error = engine()
        .ask("laya", &long, &questions())
        .expect_err("the state exceeds laya's context window");
    match error {
        olaya::Error::Model {
            model,
            source: olaya::RuntimeError::StateTruncated { model: truncated },
        } => {
            assert_eq!(model, "laya");
            assert_eq!(truncated, "laya");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
#[ignore = "requires ~/.olaya/models/laya-multilingual"]
fn laya_multilingual_answers_three_primitives() {
    ask("laya-multilingual");
}

#[test]
#[ignore = "requires ~/.olaya/models/laya-typed-decisions"]
fn laya_typed_decisions_answers_three_primitives() {
    ask("laya-typed-decisions");
}
