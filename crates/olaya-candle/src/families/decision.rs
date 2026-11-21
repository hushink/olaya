//! Decision 1.0 (`decision-endpoint-v1`): candidate endpoints over a Qwen3.5
//! text backbone.
//!
//! One causal row per question:
//!
//! ```text
//! Context:\n<state>\n\nTask type: <type>\nQuestion:\n<instructions>\nOptions:
//! <option segment>...
//! \n\nSelect the single option best supported by the context and instructions.\nDecision:
//! ```
//!
//! Every option segment ends at a candidate endpoint; the row ends at a global
//! query. A small FP32 head scores each contextual endpoint against that query
//! with a shared bilinear + MLP term. Probabilities are `softmax(logits / T)`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::encoders::qwen35::{Qwen35Config, Qwen35Encoder};
use crate::{Family, FamilyFactory, LoadContext, ModelError};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder, layer_norm, linear, linear_no_bias};
use indexmap::IndexMap;
use olaya_core::{
    Calibration, CalibrationFile, Content, DecisionError, FileRole, Question, State,
    SystemOneResponse, TokenEncoder, Usage, answer_from_logits, dumps_content_canonical,
    dumps_value_canonical,
};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

const LAYOUT: &str = "decision-endpoint-v1";
const PROMPT_VERSION: &str = "structured-segmented-candidate-endpoints-global-query-v2";
const MAX_ROW_TOKENS: usize = 16_384;
/// Padded tokens (`batch * width`) per backbone pass.
const BATCH_TOKEN_BUDGET: usize = 32_768;
const MAX_CHOICES: usize = 255;
const MIN_LEVELS: usize = 2;
const MAX_LEVELS: usize = 10;
const NOUL_FALSE: &str = "The answer to the question is no.";
const NOUL_TRUE: &str = "The answer to the question is yes.";
const HEAD_LAYER_NORM_EPS: f64 = 1e-5;
const SUFFIX: &str =
    "\n\nSelect the single option best supported by the context and instructions.\nDecision:";

fn default_head_dim() -> usize {
    256
}

#[derive(Clone, Debug, Default, Deserialize)]
struct DecisionConfig {
    #[serde(default)]
    prompt_version: Option<String>,
    #[serde(default = "default_head_dim")]
    head_dim: usize,
}

struct CandidateHead {
    candidate_norm: LayerNorm,
    query_norm: LayerNorm,
    key: Linear,
    query: Linear,
    candidate_mlp: Linear,
    query_mlp: Linear,
    scalar: Linear,
    head_dim: usize,
}

impl CandidateHead {
    fn load(
        tensors: &HashMap<String, Tensor>,
        hidden: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self, ModelError> {
        let vb = VarBuilder::from_tensors(tensors.clone(), DType::F32, device);
        Ok(Self {
            candidate_norm: layer_norm(hidden, HEAD_LAYER_NORM_EPS, vb.pp("candidate_norm"))?,
            query_norm: layer_norm(hidden, HEAD_LAYER_NORM_EPS, vb.pp("query_norm"))?,
            key: linear_no_bias(hidden, head_dim, vb.pp("key"))?,
            query: linear_no_bias(hidden, head_dim, vb.pp("query"))?,
            candidate_mlp: linear(hidden, head_dim, vb.pp("candidate_mlp"))?,
            query_mlp: linear_no_bias(hidden, head_dim, vb.pp("query_mlp"))?,
            scalar: linear_no_bias(head_dim, 1, vb.pp("scalar"))?,
            head_dim,
        })
    }

    fn forward(&self, candidates: &Tensor, query: &Tensor) -> candle_core::Result<Tensor> {
        let candidates = self.candidate_norm.forward(candidates)?;
        let query = self.query_norm.forward(query)?;
        let keyed = self.key.forward(&candidates)?;
        let queried = self.query.forward(&query)?;
        let bilinear = keyed
            .broadcast_mul(&queried)?
            .sum(1)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let interaction = self
            .candidate_mlp
            .forward(&candidates)?
            .broadcast_add(&self.query_mlp.forward(&query)?)?
            .gelu_erf()?;
        let interaction = self.scalar.forward(&interaction)?.squeeze(1)?;
        bilinear + interaction
    }
}

struct DecisionRow {
    ids: Vec<u32>,
    candidate_positions: Vec<usize>,
    query_position: usize,
}

struct DecisionFamily {
    tokenizer: Arc<dyn TokenEncoder>,
    encoder: Qwen35Encoder,
    head: CandidateHead,
    calibration: Calibration,
    device: Device,
}

fn payload(content: &Content) -> String {
    match content {
        Content::String(text) => text.clone(),
        other => dumps_content_canonical(other),
    }
}

fn content_value(content: &Content) -> Value {
    serde_json::to_value(content).expect("Content is always serializable")
}

fn option_segment(key: &str, description: Value) -> String {
    let mut object = Map::new();
    object.insert("key".to_owned(), Value::String(key.to_owned()));
    object.insert("description".to_owned(), description);
    format!(
        "\n<option>\n{}\n</option>",
        dumps_value_canonical(&Value::Object(object))
    )
}

fn noul_description(criteria: Option<&olaya_core::NoulCriteria>, is_true: bool) -> Value {
    let (value, default) = match criteria {
        Some(criteria) if is_true => (criteria.is_true.as_ref(), NOUL_TRUE),
        Some(criteria) => (criteria.is_false.as_ref(), NOUL_FALSE),
        None if is_true => (None, NOUL_TRUE),
        None => (None, NOUL_FALSE),
    };
    match value {
        None => Value::String(default.to_owned()),
        Some(content) => content_value(content),
    }
}

impl DecisionFamily {
    fn build_row(
        &self,
        qid: &str,
        question: &Question,
        state: &Content,
    ) -> Result<DecisionRow, ModelError> {
        let instructions = question.instructions().ok_or_else(|| {
            DecisionError::invalid(format!("question {qid:?}: instructions is required"))
        })?;
        let kind = question.question_type().name();
        let (options, keys) = match question {
            Question::Choice(choice) => {
                if !(MIN_LEVELS..=MAX_CHOICES).contains(&choice.criteria.len()) {
                    return Err(DecisionError::invalid(format!(
                        "question {qid:?}: choice needs {MIN_LEVELS}-{MAX_CHOICES} criteria"
                    ))
                    .into());
                }
                let mut options = Vec::with_capacity(choice.criteria.len());
                let mut keys = Vec::with_capacity(choice.criteria.len());
                for (key, description) in &choice.criteria {
                    keys.push(key.clone());
                    options.push(option_segment(key, content_value(description)));
                }
                (options, keys)
            }
            Question::Score(score) => {
                if !(MIN_LEVELS..=MAX_LEVELS).contains(&score.criteria.len()) {
                    return Err(DecisionError::invalid(format!(
                        "question {qid:?}: score needs {MIN_LEVELS}-{MAX_LEVELS} levels"
                    ))
                    .into());
                }
                let mut options = Vec::with_capacity(score.criteria.len());
                let mut keys = Vec::with_capacity(score.criteria.len());
                for (index, level) in score.criteria.iter().enumerate() {
                    let key = index.to_string();
                    keys.push(key.clone());
                    options.push(option_segment(&key, content_value(level)));
                }
                (options, keys)
            }
            Question::Noul(noul) => {
                let criteria = noul.criteria.as_ref();
                let mut options = Vec::with_capacity(2);
                let mut keys = Vec::with_capacity(2);
                for (key, is_true) in [("false", false), ("true", true)] {
                    keys.push(key.to_owned());
                    options.push(option_segment(key, noul_description(criteria, is_true)));
                }
                (options, keys)
            }
            Question::Box(_) => {
                return Err(
                    DecisionError::unsupported("decision does not answer box questions").into(),
                );
            }
        };
        let _ = keys;

        let prefix = format!(
            "Context:\n{}\n\nTask type: {kind}\nQuestion:\n{}\nOptions:",
            payload(state),
            payload(instructions),
        );
        let mut ids = self.tokenizer.encode(&prefix)?;
        let mut candidate_positions = Vec::with_capacity(options.len());
        for option in &options {
            let part = self.tokenizer.encode(option)?;
            if part.is_empty() {
                return Err(DecisionError::invalid(format!(
                    "question {qid:?}: an option tokenized to no tokens"
                ))
                .into());
            }
            ids.extend_from_slice(&part);
            candidate_positions.push(ids.len() - 1);
        }
        ids.extend_from_slice(&self.tokenizer.encode(SUFFIX)?);
        if ids.len() > MAX_ROW_TOKENS {
            return Err(DecisionError::InputTooLong {
                tokens: ids.len(),
                limit: MAX_ROW_TOKENS,
            }
            .into());
        }
        Ok(DecisionRow {
            query_position: ids.len() - 1,
            ids,
            candidate_positions,
        })
    }

    /// Score rows in budgeted batches. Right padding cannot change a causal
    /// row's hidden states, so batching is numerically equivalent to one
    /// forward per row.
    fn score_rows(&self, rows: &[&DecisionRow]) -> Result<Vec<Vec<f32>>, ModelError> {
        let mut scores = Vec::with_capacity(rows.len());
        let lengths = rows.iter().map(|row| row.ids.len()).collect::<Vec<_>>();
        for group in super::batch::groups(&lengths, BATCH_TOKEN_BUDGET) {
            let batch = &rows[group];
            let width = batch.iter().map(|row| row.ids.len()).max().unwrap_or(1);
            let mut ids = vec![0u32; batch.len() * width];
            for (index, row) in batch.iter().enumerate() {
                ids[index * width..index * width + row.ids.len()].copy_from_slice(&row.ids);
            }
            let input = Tensor::from_vec(ids, (batch.len(), width), &self.device)?;
            let hidden = self.encoder.forward(&input)?;
            for (index, row) in batch.iter().enumerate() {
                let hidden = hidden.i(index)?;
                let positions = Tensor::from_vec(
                    row.candidate_positions
                        .iter()
                        .map(|position| *position as u32)
                        .collect::<Vec<_>>(),
                    row.candidate_positions.len(),
                    &self.device,
                )?;
                let candidates = hidden.index_select(&positions, 0)?;
                let query = hidden.i(row.query_position)?.unsqueeze(0)?;
                scores.push(
                    self.head
                        .forward(&candidates, &query)?
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()?,
                );
            }
        }
        Ok(scores)
    }
}

impl Family for DecisionFamily {
    fn id(&self) -> &'static str {
        "decision"
    }

    fn capabilities(&self) -> olaya_core::Capabilities {
        olaya_core::Capabilities::classification()
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
        let (rows, input_tokens) = self.build_question_rows(state, questions)?;
        let refs = rows.iter().map(|(_, _, row)| row).collect::<Vec<_>>();
        let logits = self.score_rows(&refs)?;
        let mut answers = IndexMap::with_capacity(rows.len());
        for ((qid, question, row), logits) in rows.iter().zip(logits) {
            answers.insert(
                qid.clone(),
                self.answer_question(question, &logits, row.ids.len())?,
            );
        }
        Ok(SystemOneResponse::new(
            self.id(),
            answers,
            Usage::new(input_tokens, 0),
        ))
    }

    fn predict_batch(
        &self,
        requests: &[(&State, &olaya_core::Questions)],
    ) -> Vec<Result<SystemOneResponse, ModelError>> {
        let mut built = requests
            .iter()
            .map(|(state, questions)| Some(self.build_question_rows(state, questions)))
            .collect::<Vec<_>>();
        let rows = built
            .iter()
            .filter_map(|entry| entry.as_ref())
            .filter_map(|entry| entry.as_ref().ok())
            .flat_map(|(rows, _)| rows.iter().map(|(_, _, row)| row))
            .collect::<Vec<_>>();
        let scores = match self.score_rows(&rows) {
            Ok(scores) => scores,
            Err(_) => {
                return requests
                    .iter()
                    .map(|(state, questions)| self.predict(state, questions))
                    .collect();
            }
        };
        let mut cursor = 0;
        let mut results = Vec::with_capacity(requests.len());
        for entry in built.iter_mut() {
            match entry.take().expect("entry") {
                Err(error) => results.push(Err(error)),
                Ok((rows, input_tokens)) => {
                    let start = cursor;
                    let mut answers = IndexMap::with_capacity(rows.len());
                    let mut failed = None;
                    for (offset, (qid, question, row)) in rows.iter().enumerate() {
                        let logits = &scores[start + offset];
                        match self.answer_question(question, logits, row.ids.len()) {
                            Ok(answer) => {
                                answers.insert(qid.clone(), answer);
                            }
                            Err(error) => {
                                failed = Some(error);
                                break;
                            }
                        }
                    }
                    cursor = start + rows.len();
                    results.push(match failed {
                        Some(error) => Err(error),
                        None => Ok(SystemOneResponse::new(
                            self.id(),
                            answers,
                            Usage::new(input_tokens, 0),
                        )),
                    });
                }
            }
        }
        results
    }
}

/// The rendered rows of one request plus the token count they consumed.
type QuestionRows = (Vec<(String, Question, DecisionRow)>, u64);

impl DecisionFamily {
    fn build_question_rows(
        &self,
        state: &State,
        questions: &olaya_core::Questions,
    ) -> Result<QuestionRows, ModelError> {
        let state = match state.as_content() {
            Content::String(_) | Content::Object(_) | Content::Array(_) => state.as_content(),
            Content::Null => {
                return Err(
                    DecisionError::invalid("state must be a string, object or array").into(),
                );
            }
        };
        let mut rows = Vec::with_capacity(questions.len());
        let mut input_tokens = 0u64;
        for (qid, question) in questions.iter() {
            if matches!(question, Question::Box(_)) {
                return Err(
                    DecisionError::unsupported("decision does not answer box questions").into(),
                );
            }
            let row = self.build_row(qid, question, state)?;
            input_tokens += row.ids.len() as u64;
            rows.push((qid.clone(), question.clone(), row));
        }
        Ok((rows, input_tokens))
    }

    fn answer_question(
        &self,
        question: &Question,
        logits: &[f32],
        tokens: usize,
    ) -> Result<olaya_core::Answer, ModelError> {
        Ok(match question {
            Question::Choice(choice) => {
                let labels = choice.criteria.keys().cloned().collect::<Vec<_>>();
                answer_from_logits(question, &labels, &[], logits, &self.calibration, tokens)?
            }
            Question::Score(score) => answer_from_logits(
                question,
                &[],
                &score.criteria,
                logits,
                &self.calibration,
                tokens,
            )?,
            Question::Noul(_) => answer_from_logits(
                question,
                &["false".to_owned(), "true".to_owned()],
                &[],
                logits,
                &self.calibration,
                tokens,
            )?,
            Question::Box(_) => {
                return Err(
                    DecisionError::unsupported("decision does not answer box questions").into(),
                );
            }
        })
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

fn weights_paths<'a>(
    ctx: &'a LoadContext<'a>,
) -> Result<(&'a olaya_core::ManifestFile, &'a olaya_core::ManifestFile), ModelError> {
    let weights = ctx
        .manifest
        .files
        .iter()
        .filter(|file| file.role == FileRole::Weights)
        .collect::<Vec<_>>();
    let backbone = weights
        .iter()
        .find(|file| file.path.contains("backbone"))
        .or_else(|| {
            weights
                .iter()
                .find(|file| !file.path.ends_with("decision_head.safetensors"))
        })
        .ok_or_else(|| ModelError::config("decision manifest has no backbone weights"))?;
    let head = weights
        .iter()
        .find(|file| file.path.ends_with("decision_head.safetensors"))
        .ok_or_else(|| ModelError::config("decision manifest has no decision_head.safetensors"))?;
    Ok((backbone, head))
}

/// Loads a Decision 1.0 checkpoint (`decision-head` + Qwen3.5 backbone).
pub struct Factory;

impl FamilyFactory for Factory {
    fn family(&self) -> &'static str {
        "decision"
    }

    fn load(&self, ctx: &LoadContext<'_>) -> Result<Arc<dyn Family>, ModelError> {
        if ctx.manifest.layout != LAYOUT {
            return Err(ModelError::config(format!(
                "decision serves layout {LAYOUT}, got {:?}",
                ctx.manifest.layout
            )));
        }
        if ctx.manifest.arch != "qwen35" {
            return Err(ModelError::config(format!(
                "decision serves the qwen35 architecture, got {:?}",
                ctx.manifest.arch
            )));
        }
        let config_file = ctx
            .manifest
            .file(FileRole::Config)
            .ok_or_else(|| ModelError::config("decision manifest has no backbone config"))?;
        let config = Qwen35Config::from_file(&ctx.dir.join(&config_file.path))?;
        let decision_file = ctx
            .manifest
            .file(FileRole::Decision)
            .ok_or_else(|| ModelError::config("decision manifest has no decision config"))?;
        let decision: DecisionConfig = read_json(&ctx.dir.join(&decision_file.path))?;
        if let Some(version) = &decision.prompt_version
            && version != PROMPT_VERSION
        {
            return Err(ModelError::config(format!(
                "decision checkpoint prompt_version {version:?} is not {PROMPT_VERSION}"
            )));
        }
        if decision.head_dim == 0 {
            return Err(ModelError::config("decision head_dim is zero"));
        }

        let (backbone_file, head_file) = weights_paths(ctx)?;
        let mut backbone_tensors: HashMap<String, Tensor> =
            candle_core::safetensors::load(ctx.dir.join(&backbone_file.path), ctx.device)?;
        let mut head_tensors: HashMap<String, Tensor> =
            candle_core::safetensors::load(ctx.dir.join(&head_file.path), ctx.device)?;
        for tensor in backbone_tensors.values_mut() {
            if tensor.dtype() != DType::F32 {
                *tensor = tensor.to_dtype(DType::F32)?;
            }
        }
        for tensor in head_tensors.values_mut() {
            if tensor.dtype() != DType::F32 {
                *tensor = tensor.to_dtype(DType::F32)?;
            }
        }
        let encoder = Qwen35Encoder::load(&config, backbone_tensors, ctx.device)?;
        let head = CandidateHead::load(
            &head_tensors,
            config.hidden_size,
            decision.head_dim,
            ctx.device,
        )?;
        Ok(Arc::new(DecisionFamily {
            tokenizer: ctx.tokenizer.clone(),
            encoder,
            head,
            calibration: read_calibration(ctx)?,
            device: ctx.device.clone(),
        }))
    }
}
