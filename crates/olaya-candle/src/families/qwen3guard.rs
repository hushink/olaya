//! Qwen3Guard-Gen (`qwen3guard-gen-v1`): a fixed-preset safety guard.
//!
//! Qwen3Guard generates `Safety: Safe|Controversial|Unsafe` and a category
//! line. A decision reads two causal rows of its chat template instead of
//! generating:
//!
//! ```text
//! row 0  <prompt>Safety:                          -> Safe / Controversial / Unsafe
//! row 1  <prompt>Safety: Unsafe\nCategories:      -> the first unsafe category
//! ```
//!
//! Only the model's own four preset questions exist; requests must send a
//! subset of them unchanged. Option logits are the first-token logits of the
//! fixed candidate tokens (a tied LM head gathered at those ids), and
//! `unsafe` / `unsafe_strict` combine them with log-sum-exp.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::encoders::qwen3::{Qwen3Config, Qwen3Encoder};
use crate::{Family, FamilyFactory, LoadContext, ModelError};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use indexmap::IndexMap;
use olaya_core::{
    Calibration, CalibrationFile, Content, DecisionError, FileRole, Question, State,
    SystemOneResponse, TokenEncoder, Usage, answer_from_logits, dumps_content,
};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

const LAYOUT: &str = "qwen3guard-gen-v1";
const FAMILY: &str = "qwen3guard";

#[derive(Clone, Debug, Deserialize)]
struct GuardTemplates {
    prompt_head: String,
    prompt_tail: String,
    #[serde(default = "default_row_safety")]
    row_safety: String,
    #[serde(default = "default_row_category")]
    row_category: String,
}

fn default_row_safety() -> String {
    "{prompt_head}{state}{prompt_tail}Safety:".to_owned()
}

fn default_row_category() -> String {
    "{prompt_head}{state}{prompt_tail}Safety: Unsafe\nCategories:".to_owned()
}

#[derive(Clone, Debug, Deserialize)]
struct CandidateSet {
    labels: Vec<String>,
    ids: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize)]
struct GuardCandidates {
    safety: CandidateSet,
    category: CandidateSet,
}

#[derive(Clone, Debug, Deserialize)]
struct GuardDecision {
    #[serde(default)]
    preset_only: bool,
    questions: IndexMap<String, Question>,
    templates: GuardTemplates,
    candidates: GuardCandidates,
    max_len: usize,
}

struct Qwen3GuardFamily {
    tokenizer: Arc<dyn TokenEncoder>,
    encoder: Qwen3Encoder,
    decision: GuardDecision,
    calibration: Calibration,
    device: Device,
}

impl Qwen3GuardFamily {
    fn validate_decision(&self) -> Result<(), ModelError> {
        if !self.decision.preset_only {
            return Err(ModelError::config(
                "qwen3guard decision must be preset_only",
            ));
        }
        if self.decision.max_len == 0 {
            return Err(ModelError::config("qwen3guard max_len is zero"));
        }
        let safety = &self.decision.candidates.safety;
        if safety.ids.len() != 3 || safety.labels != ["safe", "controversial", "unsafe"] {
            return Err(ModelError::config(
                "qwen3guard safety candidates must be safe/controversial/unsafe",
            ));
        }
        let category = &self.decision.candidates.category;
        if category.ids.is_empty() || category.ids.len() != category.labels.len() {
            return Err(ModelError::config(
                "qwen3guard category candidates need one id per label",
            ));
        }
        for question in ["safety", "unsafe", "unsafe_strict", "category"] {
            if !self.decision.questions.contains_key(question) {
                return Err(ModelError::config(format!(
                    "qwen3guard decision has no {question:?} preset question"
                )));
            }
        }
        self.check_choice("safety", &safety.labels)?;
        self.check_choice("category", &category.labels)?;
        Ok(())
    }

    fn check_choice(&self, id: &str, labels: &[String]) -> Result<(), ModelError> {
        let Some(Question::Choice(choice)) = self.decision.questions.get(id) else {
            return Err(ModelError::config(format!(
                "qwen3guard preset {id:?} must be a choice"
            )));
        };
        let keys = choice.criteria.keys().cloned().collect::<Vec<_>>();
        if keys != labels {
            return Err(ModelError::config(format!(
                "qwen3guard preset {id:?} criteria do not match its candidate labels"
            )));
        }
        Ok(())
    }

    /// Run the (at most two) preset rows in one causal batch. Right padding is
    /// inert because the rows are right-padded after their last real token.
    fn run_rows(&self, rows: &[(&str, &[u32])]) -> Result<Vec<(Vec<f32>, usize)>, ModelError> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let mut encoded = Vec::with_capacity(rows.len());
        for (row, candidates) in rows {
            let ids = self.tokenizer.encode(row)?;
            if ids.len() > self.decision.max_len {
                return Err(DecisionError::InputTooLong {
                    tokens: ids.len(),
                    limit: self.decision.max_len,
                }
                .into());
            }
            if ids.is_empty() {
                return Err(DecisionError::invalid("qwen3guard tokenized an empty row").into());
            }
            encoded.push((ids, *candidates));
        }
        let width = encoded.iter().map(|(ids, _)| ids.len()).max().unwrap_or(1);
        let mut batch = vec![0u32; encoded.len() * width];
        for (index, (ids, _)) in encoded.iter().enumerate() {
            batch[index * width..index * width + ids.len()].copy_from_slice(ids);
        }
        let input = Tensor::from_vec(batch, (encoded.len(), width), &self.device)?;
        let hidden = self.encoder.forward(&input)?;
        let mut out = Vec::with_capacity(encoded.len());
        for (index, (ids, candidates)) in encoded.iter().enumerate() {
            let last = hidden.i((index, ids.len() - 1))?.unsqueeze(0)?;
            let logits = self
                .encoder
                .candidate_logits(&last, candidates)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            out.push((logits, ids.len()));
        }
        Ok(out)
    }
}

fn render_row(template: &str, templates: &GuardTemplates, state: &str) -> String {
    template
        .replace("{prompt_head}", &templates.prompt_head)
        .replace("{prompt_tail}", &templates.prompt_tail)
        .replace("{state}", state)
}

fn logsumexp2(a: f32, b: f32) -> f32 {
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

impl Family for Qwen3GuardFamily {
    fn id(&self) -> &'static str {
        FAMILY
    }

    fn capabilities(&self) -> olaya_core::Capabilities {
        olaya_core::Capabilities {
            noul: true,
            choice: true,
            score: false,
            ..olaya_core::Capabilities::default()
        }
    }

    fn tokenizer(&self) -> &dyn TokenEncoder {
        self.tokenizer.as_ref()
    }

    fn calibration(&self) -> &Calibration {
        &self.calibration
    }

    fn predict(
        &self,
        state: &State,
        questions: &olaya_core::Questions,
    ) -> Result<SystemOneResponse, ModelError> {
        let plan = self.plan(state, questions)?;
        let mut specs = Vec::with_capacity(2);
        if let Some(row) = &plan.safety_row {
            specs.push((row.as_str(), self.decision.candidates.safety.ids.as_slice()));
        }
        if let Some(row) = &plan.category_row {
            specs.push((
                row.as_str(),
                self.decision.candidates.category.ids.as_slice(),
            ));
        }
        let mut results = self.run_rows(&specs)?.into_iter();
        let safety = plan.safety_row.as_ref().and_then(|_| results.next());
        let category = plan.category_row.as_ref().and_then(|_| results.next());
        self.answer_plan(plan, safety, category)
    }

    fn predict_batch(
        &self,
        requests: &[(&State, &olaya_core::Questions)],
    ) -> Vec<Result<SystemOneResponse, ModelError>> {
        let plans = requests
            .iter()
            .map(|(state, questions)| Some(self.plan(state, questions)))
            .collect::<Vec<_>>();
        let mut specs = Vec::new();
        let mut marks = Vec::new();
        for (index, plan) in plans.iter().enumerate() {
            let Some(Ok(plan)) = plan else { continue };
            if let Some(row) = &plan.safety_row {
                specs.push((row.as_str(), self.decision.candidates.safety.ids.as_slice()));
                marks.push((index, false));
            }
            if let Some(row) = &plan.category_row {
                specs.push((
                    row.as_str(),
                    self.decision.candidates.category.ids.as_slice(),
                ));
                marks.push((index, true));
            }
        }
        let outputs = match self.run_rows(&specs) {
            Ok(outputs) => outputs,
            Err(_) => {
                return requests
                    .iter()
                    .map(|(state, questions)| self.predict(state, questions))
                    .collect();
            }
        };
        let mut safety = vec![None; requests.len()];
        let mut category = vec![None; requests.len()];
        for ((index, is_category), output) in marks.into_iter().zip(outputs) {
            if is_category {
                category[index] = Some(output);
            } else {
                safety[index] = Some(output);
            }
        }
        plans
            .into_iter()
            .enumerate()
            .map(|(index, plan)| match plan.expect("plan") {
                Err(error) => Err(error),
                Ok(plan) => self.answer_plan(plan, safety[index].take(), category[index].take()),
            })
            .collect()
    }
}

struct GuardPlan {
    questions: Vec<(String, Question)>,
    safety_row: Option<String>,
    category_row: Option<String>,
}

impl Qwen3GuardFamily {
    fn plan(
        &self,
        state: &State,
        questions: &olaya_core::Questions,
    ) -> Result<GuardPlan, ModelError> {
        let state_text = match state.as_content() {
            Content::String(text) => text.clone(),
            Content::Null => {
                return Err(
                    DecisionError::invalid("state must be a string, object or array").into(),
                );
            }
            content => dumps_content(content),
        };
        let mut planned = Vec::with_capacity(questions.len());
        for (qid, question) in questions.iter() {
            let preset = self.decision.questions.get(qid.as_str()).ok_or_else(|| {
                DecisionError::invalid(format!(
                    "qwen3guard is preset-only; unknown question {qid:?}"
                ))
            })?;
            if preset != question {
                return Err(DecisionError::invalid(format!(
                    "question {qid:?} does not match the qwen3guard preset"
                ))
                .into());
            }
            planned.push((qid.clone(), question.clone()));
        }
        let templates = &self.decision.templates;
        let needs_safety = planned
            .iter()
            .any(|(id, _)| matches!(id.as_str(), "safety" | "unsafe" | "unsafe_strict"));
        let needs_category = planned.iter().any(|(id, _)| id.as_str() == "category");
        Ok(GuardPlan {
            safety_row: needs_safety
                .then(|| render_row(&templates.row_safety, templates, &state_text)),
            category_row: needs_category
                .then(|| render_row(&templates.row_category, templates, &state_text)),
            questions: planned,
        })
    }

    fn answer_plan(
        &self,
        plan: GuardPlan,
        safety: Option<(Vec<f32>, usize)>,
        category: Option<(Vec<f32>, usize)>,
    ) -> Result<SystemOneResponse, ModelError> {
        let mut answers = IndexMap::with_capacity(plan.questions.len());
        let mut input_tokens = 0u64;
        if let Some((_, tokens)) = &safety {
            input_tokens += *tokens as u64;
        }
        if let Some((_, tokens)) = &category {
            input_tokens += *tokens as u64;
        }
        for (qid, question) in &plan.questions {
            let (option_logits, labels, tokens) = match qid.as_str() {
                "safety" => {
                    let (logits, tokens) = safety.as_ref().expect("requested");
                    (
                        logits.clone(),
                        self.decision.candidates.safety.labels.clone(),
                        *tokens,
                    )
                }
                "category" => {
                    let (logits, tokens) = category.as_ref().expect("requested");
                    (
                        logits.clone(),
                        self.decision.candidates.category.labels.clone(),
                        *tokens,
                    )
                }
                "unsafe" => {
                    let (logits, tokens) = safety.as_ref().expect("requested");
                    (
                        vec![logsumexp2(logits[0], logits[1]), logits[2]],
                        vec!["false".to_owned(), "true".to_owned()],
                        *tokens,
                    )
                }
                "unsafe_strict" => {
                    let (logits, tokens) = safety.as_ref().expect("requested");
                    (
                        vec![logits[0], logsumexp2(logits[1], logits[2])],
                        vec!["false".to_owned(), "true".to_owned()],
                        *tokens,
                    )
                }
                other => unreachable!("validated preset question {other}"),
            };
            let answer = answer_from_logits(
                question,
                &labels,
                &[],
                &option_logits,
                &self.calibration,
                tokens,
            )?;
            answers.insert(qid.clone(), answer);
        }
        Ok(SystemOneResponse::new(
            FAMILY,
            answers,
            Usage::new(input_tokens, 0),
        ))
    }
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, ModelError> {
    let bytes = std::fs::read(path)?;
    serde_json::from_slice(&bytes)
        .map_err(|error| ModelError::config(format!("{}: {error}", path.display())))
}

fn read_calibration(ctx: &LoadContext<'_>) -> Result<Calibration, ModelError> {
    let file = match ctx.manifest.file(FileRole::Calibration) {
        Some(file) => read_json(&ctx.dir.join(&file.path))?,
        None => CalibrationFile {
            temperature: vec![Value::from(1.0); 3],
            ..CalibrationFile::default()
        },
    };
    Ok(Calibration::from_file(&file, ctx.manifest.calibration_mode))
}

/// Loads a Qwen3Guard-Gen checkpoint plus its embedded preset.
pub struct Factory;

impl FamilyFactory for Factory {
    fn family(&self) -> &'static str {
        FAMILY
    }

    fn load(&self, ctx: &LoadContext<'_>) -> Result<Arc<dyn Family>, ModelError> {
        if ctx.manifest.layout != LAYOUT {
            return Err(ModelError::config(format!(
                "qwen3guard serves layout {LAYOUT}, got {:?}",
                ctx.manifest.layout
            )));
        }
        if ctx.manifest.arch != "qwen3" {
            return Err(ModelError::config(format!(
                "qwen3guard serves the qwen3 architecture, got {:?}",
                ctx.manifest.arch
            )));
        }
        let config_file = ctx
            .manifest
            .file(FileRole::Config)
            .ok_or_else(|| ModelError::config("qwen3guard manifest has no config file"))?;
        let config: Qwen3Config = read_json(&ctx.dir.join(&config_file.path))?;
        config.validate()?;
        let decision_file = ctx
            .manifest
            .file(FileRole::Decision)
            .ok_or_else(|| ModelError::config("qwen3guard manifest has no decision file"))?;
        let decision: GuardDecision = read_json(&ctx.dir.join(&decision_file.path))?;

        let weights = ctx
            .manifest
            .file(FileRole::Weights)
            .ok_or_else(|| ModelError::config("qwen3guard manifest has no weights file"))?;
        let weights_path = ctx.dir.join(&weights.path);
        let tensors: HashMap<String, Tensor> =
            candle_core::safetensors::load(&weights_path, ctx.device)?;
        let vb = VarBuilder::from_tensors(tensors.clone(), ctx.compute_dtype(), ctx.device);
        drop(tensors);
        let encoder = Qwen3Encoder::load(&config, vb)?;

        let family = Qwen3GuardFamily {
            tokenizer: ctx.tokenizer.clone(),
            encoder,
            decision,
            calibration: read_calibration(ctx)?,
            device: ctx.device.clone(),
        };
        family.validate_decision()?;
        if family
            .decision
            .candidates
            .safety
            .ids
            .iter()
            .chain(&family.decision.candidates.category.ids)
            .any(|id| *id as usize >= config.vocab_size)
        {
            return Err(ModelError::config(
                "qwen3guard candidate ids exceed the vocabulary",
            ));
        }
        Ok(Arc::new(family))
    }
}
