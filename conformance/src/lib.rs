//! Conformance harness.
//!
//! Goldens are produced by the upstream references (ollaya's ORT runtime for
//! Laya; upstream decima 2.3.0 in PyTorch fp32 and transformers' NLI models for
//! the other families) and checked into `conformance/fixtures`. The candle
//! runtime is never asked to load ONNX; ORT stays outside the published crates.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use olaya::{Answers, Engine, Error, Questions, RuntimeError, parse_request_value};
use serde::Deserialize;
use serde_json::{Map, Value};

/// The fixture root for a family and checkpoint.
pub fn fixture_dir(family: &str, checkpoint: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(family)
        .join(checkpoint)
}

/// The per-model golden file: one JSON request/answer record per line.
pub fn fixture_path(family: &str, model: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(family)
        .join(format!("{model}.jsonl"))
}

/// The logits tolerance used by parity tests.
pub const LOGITS_TOLERANCE: f32 = 1e-4;
/// The probability tolerance used by decima goldens. Long option rows
/// accumulate a little more fp32 noise than the short smoke requests
/// (measured ≤ 1.1e-6, so one order above that).
pub const PROBABILITY_TOLERANCE: f64 = 1e-5;
/// The probability tolerance used by NLI goldens (two independent fp32 stacks).
pub const NLI_PROBABILITY_TOLERANCE: f64 = 1e-5;
/// The probability tolerance used by kev goldens: the merged LoRA delta and the
/// pointer head amplify fp32 noise a little more than a single classification
/// head (measured ≤ `3e-8` on the smoke requests).
pub const KEV_PROBABILITY_TOLERANCE: f64 = 1e-4;
/// The probability tolerance used by Qwen3Guard goldens (transformers fp32).
pub const QWEN3GUARD_PROBABILITY_TOLERANCE: f64 = 1e-5;
/// The probability tolerance used by Decision-1.0-Eos goldens (author PyTorch
/// fp32; measured ≤ `1e-7`).
pub const DECISION_PROBABILITY_TOLERANCE: f64 = 1e-4;
/// The probability tolerance used by Clef goldens. The upstream release is
/// served in bf16 while the candle trunk runs fp32; the 9B backbone's bf16
/// rounding moves low-confidence options by up to ~1e-2.
pub const CLEF_PROBABILITY_TOLERANCE: f64 = 2e-2;

/// One replayed request and its upstream result.
#[derive(Clone, Debug, Deserialize)]
pub struct Golden {
    pub id: String,
    pub model: String,
    pub state: Value,
    pub questions: Map<String, Value>,
    /// `None` for a request upstream answers.
    #[serde(default)]
    pub error: Option<GoldenError>,
    /// Answers keyed by question id; `None` for a rejected request.
    #[serde(default)]
    pub answers: Option<Map<String, Value>>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct GoldenError {
    pub code: GoldenErrorCode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoldenErrorCode {
    InvalidRequest,
    StateTruncated,
    InputTooLong,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum GoldenAnswer {
    Noul {
        noul: f64,
    },
    Choice {
        choice: String,
        probabilities: HashMap<String, f64>,
    },
    Score {
        score: f64,
        legend: Map<String, Value>,
        probabilities: HashMap<String, f64>,
    },
}

/// Read a JSONL golden file.
pub fn load_goldens(path: &Path) -> Result<Vec<Golden>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            serde_json::from_str(line)
                .map_err(|error| format!("{}:{}: {error}", path.display(), index + 1))
        })
        .collect()
}

/// Replay one golden request against `engine` and check its answers.
pub fn run_golden(engine: &Engine, golden: &Golden, tolerance: f64) -> Result<(), String> {
    let body = Value::Object(
        [
            ("model".to_owned(), Value::String(golden.model.clone())),
            ("state".to_owned(), golden.state.clone()),
            (
                "questions".to_owned(),
                Value::Object(golden.questions.clone()),
            ),
        ]
        .into_iter()
        .collect(),
    );
    let request = match parse_request_value(body) {
        Ok(request) => request,
        Err(report) => {
            return match golden.error {
                Some(GoldenError {
                    code: GoldenErrorCode::InvalidRequest,
                }) => Ok(()),
                _ => Err(format!("request rejected by the parser: {report}")),
            };
        }
    };
    let state = request.state;
    let questions = Questions::from(request.questions);
    match golden.error {
        Some(GoldenError { code }) => {
            let result = engine.ask(&golden.model, &state, &questions);
            let matched = matches!(
                (&code, &result),
                (GoldenErrorCode::InvalidRequest, Err(Error::Validation(_)))
                    | (
                        GoldenErrorCode::InvalidRequest,
                        Err(Error::Model {
                            source: RuntimeError::InvalidRequest { .. },
                            ..
                        }),
                    )
                    | (
                        GoldenErrorCode::StateTruncated,
                        Err(Error::Model {
                            source: RuntimeError::StateTruncated { .. },
                            ..
                        }),
                    )
                    | (
                        GoldenErrorCode::InputTooLong,
                        Err(Error::Model {
                            source: RuntimeError::InputTooLong { .. },
                            ..
                        }),
                    )
            );
            if matched {
                Ok(())
            } else {
                Err(format!(
                    "expected {:?}, got {}",
                    code,
                    describe_result(&result)
                ))
            }
        }
        None => {
            let response = engine
                .ask(&golden.model, &state, &questions)
                .map_err(|error| format!("unexpected error: {error}"))?;
            let answers = Answers::new(response.answers);
            check_answers(
                &answers,
                golden.answers.as_ref().ok_or("missing answers")?,
                tolerance,
            )
        }
    }
}

fn describe_result(result: &Result<olaya::SystemOneResponse, Error>) -> String {
    match result {
        Ok(_) => "an answer".to_owned(),
        Err(error) => error.to_string(),
    }
}

fn check_answers(
    answers: &Answers,
    expected: &Map<String, Value>,
    tolerance: f64,
) -> Result<(), String> {
    if answers.len() != expected.len() {
        return Err(format!(
            "{} answers, expected {}",
            answers.len(),
            expected.len()
        ));
    }
    for (qid, value) in expected {
        let golden: GoldenAnswer =
            serde_json::from_value(value.clone()).map_err(|error| format!("{qid}: {error}"))?;
        match golden {
            GoldenAnswer::Noul { noul } => {
                let got = answers
                    .noul(qid)
                    .map_err(|error| format!("{qid}: {error}"))?;
                close(qid, "noul", got.noul, noul, tolerance)?;
            }
            GoldenAnswer::Choice {
                choice,
                probabilities,
            } => {
                let got = answers
                    .choice(qid)
                    .map_err(|error| format!("{qid}: {error}"))?;
                if got.choice != choice {
                    return Err(format!("{qid}: choice {:?} != {choice:?}", got.choice));
                }
                check_probabilities(qid, &got.probabilities, &probabilities, tolerance)?;
            }
            GoldenAnswer::Score {
                score,
                legend,
                probabilities,
            } => {
                let got = answers
                    .score(qid)
                    .map_err(|error| format!("{qid}: {error}"))?;
                close(qid, "score", got.score, score, tolerance)?;
                check_probabilities(qid, &got.probabilities, &probabilities, tolerance)?;
                if got.legend.len() != legend.len() {
                    return Err(format!(
                        "{qid}: {} legend entries, expected {}",
                        got.legend.len(),
                        legend.len()
                    ));
                }
                for (key, want) in legend {
                    let value = got
                        .legend
                        .get(key.as_str())
                        .ok_or_else(|| format!("{qid}: missing legend entry {key}"))?;
                    let value =
                        serde_json::to_value(value).map_err(|error| format!("{qid}: {error}"))?;
                    if value != want {
                        return Err(format!("{qid}: legend[{key}] {value} != {want}"));
                    }
                }
            }
        }
    }
    Ok(())
}

fn check_probabilities(
    qid: &str,
    got: &indexmap::IndexMap<String, f64>,
    want: &HashMap<String, f64>,
    tolerance: f64,
) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!(
            "{qid}: {} probabilities, expected {}",
            got.len(),
            want.len()
        ));
    }
    for (key, expected) in want {
        let value = got
            .get(key)
            .ok_or_else(|| format!("{qid}: missing probability {key}"))?;
        close(qid, key, *value, *expected, tolerance)?;
    }
    Ok(())
}

fn close(qid: &str, field: &str, got: f64, want: f64, tolerance: f64) -> Result<(), String> {
    if (got - want).abs() > tolerance {
        Err(format!("{qid}: {field} {got} != {want}"))
    } else {
        Ok(())
    }
}
