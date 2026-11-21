//! Timing harness for local model tuning.
//!
//! ```console
//! OLAYA_BENCH_MODEL=kev-0.8b OLAYA_BENCH_DEVICE=cpu `
//!   cargo test --release -p olaya --test bench -- --ignored --nocapture
//! ```
//!
//! Optional: `OLAYA_BENCH_QUESTIONS` (default 8), `OLAYA_BENCH_REPEATS`
//! (default 2), `OLAYA_BENCH_STATE` (defaults to the refund message),
//! `OLAYA_BENCH_DTYPE` (`f32`/`f16`/`bf16`, default `f32`), and
//! `OLAYA_BENCH_TF32=1` to allow CUDA tensor-core F32 GEMMs.

use std::path::PathBuf;
use std::time::Instant;

use olaya::{BuiltinFamilies, Engine, Questions, State};

fn registry_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../registry")
}

fn guard_questions() -> Questions {
    let question = |value: serde_json::Value| {
        serde_json::from_value::<olaya::Question>(value).expect("guard question")
    };
    Questions::new()
        .insert(
            "safety",
            question(serde_json::json!({
                "type": "choice",
                "instructions": "Qwen3Guard safety level of the user text",
                "criteria": {"safe": null, "controversial": null, "unsafe": null}
            })),
        )
        .insert(
            "unsafe",
            question(serde_json::json!({
                "type": "noul",
                "instructions": "Qwen3Guard: the user text is unsafe"
            })),
        )
        .insert(
            "unsafe_strict",
            question(serde_json::json!({
                "type": "noul",
                "instructions": "Qwen3Guard: the user text is unsafe or controversial"
            })),
        )
        .insert(
            "category",
            question(serde_json::json!({
                "type": "choice",
                "instructions": "Qwen3Guard: first unsafe category, given unsafe",
                "criteria": {
                    "none": null, "violent": null, "non_violent_illegal": null,
                    "sexual": null, "pii": null, "suicide_self_harm": null,
                    "unethical": null, "politically_sensitive": null,
                    "copyright": null, "jailbreak": null
                }
            })),
        )
}

fn questions(model: &str, count: usize) -> Questions {
    if model.starts_with("qwen3guard") {
        return guard_questions();
    }
    let mut questions = Questions::new();
    for index in 0..count {
        questions = match index % 3 {
            0 => questions.noul(format!("q{index:02}"), "Is this about billing?"),
            1 => questions.choice_with_instructions(
                format!("q{index:02}"),
                "Which team?",
                [
                    ("billing", "charges and refunds"),
                    ("technical", "bugs and crashes"),
                    ("account", "login and profile"),
                ],
            ),
            _ => questions.score_with_instructions(
                format!("q{index:02}"),
                "How urgent?",
                ["low", "medium", "high"],
            ),
        };
    }
    questions
}

#[test]
#[ignore = "timing harness: requires a model under ~/.olaya/models"]
fn bench_one_model() {
    let model = std::env::var("OLAYA_BENCH_MODEL").unwrap_or_else(|_| "kev-0.8b".to_owned());
    let device = std::env::var("OLAYA_BENCH_DEVICE").unwrap_or_else(|_| "cpu".to_owned());
    let question_count = std::env::var("OLAYA_BENCH_QUESTIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);
    let repeats = std::env::var("OLAYA_BENCH_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2);
    let state = State::text(std::env::var("OLAYA_BENCH_STATE").unwrap_or_else(|_| {
        "I was charged twice for my subscription this month. Please refund the extra charge."
            .to_owned()
    }));
    let dtype = match std::env::var("OLAYA_BENCH_DTYPE").as_deref() {
        Ok("f16") => olaya::candle_core::DType::F16,
        Ok("bf16") => olaya::candle_core::DType::BF16,
        _ => olaya::candle_core::DType::F32,
    };
    let tf32 = std::env::var("OLAYA_BENCH_TF32").is_ok_and(|value| value == "1");
    let device = olaya::olaya_candle::device::resolve(&device).expect("device");
    let manifests = olaya::load_manifests(registry_dir()).expect("registry");
    let engine = Engine::builder()
        .builtin_families(BuiltinFamilies::all())
        .manifests(manifests)
        .device(device)
        .dtype(dtype)
        .tf32(tf32)
        .build()
        .expect("engine");
    let questions = questions(&model, question_count);

    let load = Instant::now();
    engine.load(&model).expect("load");
    let load_seconds = load.elapsed().as_secs_f64();

    let mut runs = Vec::new();
    for repeat in 0..repeats {
        let start = Instant::now();
        let response = engine.ask(&model, &state, &questions).expect("ask");
        let seconds = start.elapsed().as_secs_f64();
        println!(
            "{model} / {}: run {repeat}: {seconds:.3}s, {} tokens",
            response.usage.input_tokens,
            response.answers.len()
        );
        runs.push(seconds);
    }
    runs.sort_by(|left, right| left.total_cmp(right));
    println!(
        "{model} ({dtype:?}, tf32={tf32}): load {load_seconds:.3}s; best {:.3}s, median {:.3}s ({} questions)",
        runs.first().copied().unwrap_or(0.0),
        runs[runs.len() / 2],
        question_count
    );
}
