//! Kev (`kev-pointer-v1`): a LoRA-adapted Qwen3.5 backbone with a pointer head.
//!
//! One causal row per question:
//!
//! ```text
//! [<|fim_prefix|>] user(render(state))[:max_state-1]
//! [<|fim_middle|>] user(render(instructions))
//! ([<|box_start|>] user(option) [<|box_end|>])*
//! [<|fim_suffix|>]
//! ```
//!
//! The option score is `k(h_option)·q(h_decide)/16`; probabilities are
//! `softmax(scores / T)`. User text escapes every `<|name|>` delimiter so it
//! cannot forge the layout's special tokens.
//!
//! The PEFT LoRA adapter (r=16, alpha=32) is merged into the fp32 backbone at
//! load (`W + 2·B·A`), matching upstream's `merge=True` reference.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::encoders::qwen35::{Qwen35Config, Qwen35Encoder};
use crate::torch;
use crate::{Family, FamilyFactory, LoadContext, ModelError};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module};
use indexmap::IndexMap;
use olaya_core::{
    Calibration, CalibrationFile, Content, DecisionError, FileRole, Question, State,
    SystemOneResponse, TokenEncoder, Usage, answer_from_logits, float_repr,
};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

const LAYOUT: &str = "kev-pointer-v1";
const MAX_OPTIONS: usize = 255;
const POINTER_SCALE: f64 = 1.0 / 16.0;
/// Padded tokens (`batch * width`) per backbone pass.
const BATCH_TOKEN_BUDGET: usize = 32_768;

#[derive(Clone, Debug, Deserialize)]
struct KevSpecials {
    state: u32,
    question: u32,
    option_open: u32,
    option_close: u32,
    decide: u32,
    #[serde(default)]
    texts: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct KevLora {
    #[serde(default)]
    scaling: Option<f64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct KevUpstream {
    #[serde(default)]
    lora: Option<KevLora>,
}

#[derive(Clone, Debug, Deserialize)]
struct KevDecision {
    max_state_tokens: usize,
    max_row_tokens: usize,
    special_tokens: KevSpecials,
    #[serde(default)]
    upstream: Option<KevUpstream>,
}

struct PointerHead {
    query: Linear,
    key: Linear,
}

impl PointerHead {
    fn load(
        tensors: &HashMap<String, Tensor>,
        hidden: usize,
        device: &Device,
    ) -> Result<Self, ModelError> {
        let find = |suffix: &str| -> Result<Tensor, ModelError> {
            tensors
                .iter()
                .find(|(name, _)| name.ends_with(suffix))
                .map(|(_, tensor)| tensor.to_dtype(DType::F32))
                .transpose()?
                .ok_or_else(|| ModelError::config(format!("kev head.pt has no {suffix}")))
        };
        let query_weight = find("q.weight")?;
        let query_bias = find("q.bias")?;
        let key_weight = find("k.weight")?;
        let key_bias = find("k.bias")?;
        if query_weight.dims().len() != 2
            || query_weight.dim(1)? != hidden
            || key_weight.dims().len() != 2
            || key_weight.dim(1)? != hidden
        {
            return Err(ModelError::config(format!(
                "kev pointer head expects hidden size {hidden}, got {:?} and {:?}",
                query_weight.dims(),
                key_weight.dims()
            )));
        }
        let _ = device;
        Ok(Self {
            query: Linear::new(query_weight, Some(query_bias)),
            key: Linear::new(key_weight, Some(key_bias)),
        })
    }

    fn forward(&self, candidates: &Tensor, decide: &Tensor) -> candle_core::Result<Tensor> {
        let query = self.query.forward(decide)?;
        let key = self.key.forward(candidates)?;
        key.broadcast_mul(&query)?
            .sum(1)?
            .affine(POINTER_SCALE, 0.0)
    }
}

struct KevFamily {
    tokenizer: Arc<dyn TokenEncoder>,
    encoder: Qwen35Encoder,
    head: PointerHead,
    decision: KevDecision,
    calibration: Calibration,
    device: Device,
}

struct KevRow {
    ids: Vec<u32>,
    option_positions: Vec<usize>,
    decide_position: usize,
}

fn python_number(number: &serde_json::Number) -> String {
    if let Some(value) = number.as_i64() {
        value.to_string()
    } else if let Some(value) = number.as_u64() {
        value.to_string()
    } else {
        float_repr(number.as_f64().unwrap_or(f64::NAN))
    }
}

/// `kev.api.render`: None -> "", scalars -> Python `str`, lists/dicts -> lines.
fn render(value: &Value, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match value {
        Value::Null => String::new(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(number) => python_number(number),
        Value::String(text) => text.clone(),
        Value::Array(items) => items
            .iter()
            .map(|item| format!("{pad}- {}", render(item, indent + 1).trim_start()))
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(map) => map
            .iter()
            .map(|(key, value)| match value {
                Value::Object(_) | Value::Array(_) => {
                    format!("{pad}{key}:\n{}", render(value, indent + 1))
                }
                other => format!("{pad}{key}: {}", render(other, indent)),
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn content_value(content: &Content) -> Value {
    serde_json::to_value(content).expect("Content is always serializable")
}

fn option_text(name: &str, description: &Value) -> String {
    match description {
        Value::Null => name.to_owned(),
        Value::String(text) if text.is_empty() => name.to_owned(),
        other => format!("{name}: {}", render(other, 0)),
    }
}

/// Rewrite `<|name|>` to `<¦name¦>` so user text cannot produce the layout's
/// special tokens.
fn escape_user(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find("<|") {
        out.push_str(&rest[..open]);
        let after = &rest[open + 2..];
        let name_len = after
            .bytes()
            .take_while(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
            .count();
        if name_len > 0 && after[name_len..].starts_with("|>") {
            out.push_str("<¦");
            out.push_str(&after[..name_len]);
            out.push_str("¦>");
            rest = &after[name_len + 2..];
        } else {
            out.push_str("<|");
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

/// Strip a base-checkpoint prefix from a tensor map (`model.language_model.`).
fn base_prefix(tensors: &HashMap<String, Tensor>) -> &'static str {
    for prefix in ["model.language_model", "model", ""] {
        let key = if prefix.is_empty() {
            "embed_tokens.weight".to_owned()
        } else {
            format!("{prefix}.embed_tokens.weight")
        };
        if tensors.contains_key(&key) {
            return prefix;
        }
    }
    ""
}

fn joined(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_owned()
    } else {
        format!("{prefix}.{name}")
    }
}

/// Merge a PEFT LoRA adapter (`W + scale·B·A`) into the fp32 backbone map.
fn merge_lora(
    tensors: &mut HashMap<String, Tensor>,
    adapter: HashMap<String, Tensor>,
    scale: f64,
    device: &Device,
) -> Result<(), ModelError> {
    let prefix = base_prefix(tensors);
    let mut pairs: HashMap<String, (Option<Tensor>, Option<Tensor>)> = HashMap::new();
    for (name, tensor) in adapter {
        let name = name
            .strip_prefix("base_model.model.")
            .unwrap_or(name.as_str());
        let Some((module, suffix)) = name.split_once(".lora_") else {
            continue;
        };
        let entry = pairs.entry(module.to_owned()).or_default();
        match suffix {
            "A.weight" => entry.0 = Some(tensor.to_dtype(DType::F32)?),
            "B.weight" => entry.1 = Some(tensor.to_dtype(DType::F32)?),
            _ => {}
        }
    }
    for (module, (a, b)) in pairs {
        let (a, b) = match (a, b) {
            (Some(a), Some(b)) => (a, b),
            _ => {
                return Err(ModelError::config(format!(
                    "kev adapter module {module:?} lacks lora_A or lora_B"
                )));
            }
        };
        let base_key = format!("{}.weight", joined(prefix, &module));
        let delta = b.matmul(&a)?.affine(scale, 0.0)?;
        let base = tensors.get_mut(&base_key).ok_or_else(|| {
            ModelError::config(format!("kev adapter targets missing weight {base_key}"))
        })?;
        *base = (base.to_dtype(DType::F32)? + delta)?;
    }
    let _ = device;
    Ok(())
}

impl KevFamily {
    fn build_row(
        &self,
        state: &Content,
        qid: &str,
        question: &Question,
    ) -> Result<KevRow, ModelError> {
        let state_text = match state {
            Content::String(text) => text.clone(),
            Content::Object(_) | Content::Array(_) => render(&content_value(state), 0),
            Content::Null => {
                return Err(
                    DecisionError::invalid("state must be a string, object or array").into(),
                );
            }
        };
        let instructions = match question.instructions() {
            None => Value::Null,
            Some(content) => content_value(content),
        };
        let options = match question {
            Question::Choice(choice) => {
                if !(1..=MAX_OPTIONS).contains(&choice.criteria.len()) {
                    return Err(DecisionError::invalid(format!(
                        "question {qid:?}: choice needs 1-{MAX_OPTIONS} criteria"
                    ))
                    .into());
                }
                choice
                    .criteria
                    .iter()
                    .map(|(key, description)| option_text(key, &content_value(description)))
                    .collect::<Vec<_>>()
            }
            Question::Score(score) => {
                if !(1..=MAX_OPTIONS).contains(&score.criteria.len()) {
                    return Err(DecisionError::invalid(format!(
                        "question {qid:?}: score needs 1-{MAX_OPTIONS} levels"
                    ))
                    .into());
                }
                score
                    .criteria
                    .iter()
                    .map(|level| render(&content_value(level), 0))
                    .collect::<Vec<_>>()
            }
            Question::Noul(noul) => {
                let criteria = noul.criteria.as_ref();
                let description = |value: Option<&Content>| match value {
                    None => Value::Null,
                    Some(content) => content_value(content),
                };
                vec![
                    option_text(
                        "no",
                        &description(criteria.and_then(|c| c.is_false.as_ref())),
                    ),
                    option_text(
                        "yes",
                        &description(criteria.and_then(|c| c.is_true.as_ref())),
                    ),
                ]
            }
            Question::Box(_) => {
                return Err(DecisionError::unsupported("kev does not answer box questions").into());
            }
        };
        let specials = &self.decision.special_tokens;
        let state_ids = self.tokenizer.encode(&escape_user(&state_text))?;
        if state_ids.len() + 1 > self.decision.max_state_tokens {
            // The state does not fit the model's state budget; the engine
            // reports `state_truncated` rather than reading a partial state.
            return Err(DecisionError::StateTruncated.into());
        }
        let mut ids = Vec::with_capacity(state_ids.len() + 1);
        ids.push(specials.state);
        ids.extend_from_slice(&state_ids);
        ids.push(specials.question);
        ids.extend(
            self.tokenizer
                .encode(&escape_user(&render(&instructions, 0)))?,
        );
        let mut option_positions = Vec::with_capacity(options.len());
        for option in &options {
            ids.push(specials.option_open);
            ids.extend(self.tokenizer.encode(&escape_user(option))?);
            ids.push(specials.option_close);
            option_positions.push(ids.len() - 1);
        }
        ids.push(specials.decide);
        if ids.len() > self.decision.max_row_tokens {
            return Err(DecisionError::InputTooLong {
                tokens: ids.len(),
                limit: self.decision.max_row_tokens,
            }
            .into());
        }
        Ok(KevRow {
            decide_position: ids.len() - 1,
            ids,
            option_positions,
        })
    }

    /// Score rows in budgeted batches. Right padding cannot change a causal
    /// row's hidden states, so batching is numerically equivalent to one
    /// forward per row.
    fn score_rows(&self, rows: &[&KevRow]) -> Result<Vec<Vec<f32>>, ModelError> {
        let lengths = rows.iter().map(|row| row.ids.len()).collect::<Vec<_>>();
        let mut scores = Vec::with_capacity(rows.len());
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
                    row.option_positions
                        .iter()
                        .map(|position| *position as u32)
                        .collect::<Vec<_>>(),
                    row.option_positions.len(),
                    &self.device,
                )?;
                let candidates = hidden.index_select(&positions, 0)?;
                let decide = hidden.i(row.decide_position)?.unsqueeze(0)?;
                scores.push(
                    self.head
                        .forward(&candidates, &decide)?
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()?,
                );
            }
        }
        Ok(scores)
    }
}

impl Family for KevFamily {
    fn id(&self) -> &'static str {
        "kev"
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
                // A rare forward failure should report per-request errors, not
                // lose the whole batch.
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
type QuestionRows = (Vec<(String, Question, KevRow)>, u64);

impl KevFamily {
    fn build_question_rows(
        &self,
        state: &State,
        questions: &olaya_core::Questions,
    ) -> Result<QuestionRows, ModelError> {
        let mut rows = Vec::with_capacity(questions.len());
        let mut input_tokens = 0u64;
        for (qid, question) in questions.iter() {
            if matches!(question, Question::Box(_)) {
                return Err(DecisionError::unsupported("kev does not answer box questions").into());
            }
            let row = self.build_row(state.as_content(), qid, question)?;
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
            Question::Score(score) => {
                // Kev's legend is the rendered text the model reads, not the
                // raw level value (kev.api.to_answers).
                let legend = score
                    .criteria
                    .iter()
                    .map(|level| Content::String(render(&content_value(level), 0)))
                    .collect::<Vec<_>>();
                answer_from_logits(question, &[], &legend, logits, &self.calibration, tokens)?
            }
            Question::Noul(_) => answer_from_logits(
                question,
                &["false".to_owned(), "true".to_owned()],
                &[],
                logits,
                &self.calibration,
                tokens,
            )?,
            Question::Box(_) => {
                return Err(DecisionError::unsupported("kev does not answer box questions").into());
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

/// Loads a Kev checkpoint: Qwen3.5 base + PEFT adapter + `head.pt`.
pub struct Factory;

impl FamilyFactory for Factory {
    fn family(&self) -> &'static str {
        "kev"
    }

    fn load(&self, ctx: &LoadContext<'_>) -> Result<Arc<dyn Family>, ModelError> {
        if ctx.manifest.layout != LAYOUT {
            return Err(ModelError::config(format!(
                "kev serves layout {LAYOUT}, got {:?}",
                ctx.manifest.layout
            )));
        }
        if ctx.manifest.arch != "qwen35" {
            return Err(ModelError::config(format!(
                "kev serves the qwen35 architecture, got {:?}",
                ctx.manifest.arch
            )));
        }
        let config_file = ctx
            .manifest
            .file(FileRole::Config)
            .ok_or_else(|| ModelError::config("kev manifest has no base config"))?;
        let config = Qwen35Config::from_file(&ctx.dir.join(&config_file.path))?;
        let decision_file = ctx
            .manifest
            .file(FileRole::Decision)
            .ok_or_else(|| ModelError::config("kev manifest has no decision file"))?;
        let decision: KevDecision = read_json(&ctx.dir.join(&decision_file.path))?;
        if decision.max_state_tokens < 2
            || decision.max_row_tokens < 2
            || decision.max_state_tokens > decision.max_row_tokens
        {
            return Err(ModelError::config(
                "kev decision has inconsistent max_state_tokens / max_row_tokens",
            ));
        }
        if let Some(text) = decision.special_tokens.texts.first()
            && ctx.tokenizer.token_id(text) != Some(decision.special_tokens.state)
        {
            return Err(ModelError::config(format!(
                "kev tokenizer {text:?} does not match the state token id"
            )));
        }

        let weights = ctx
            .manifest
            .files
            .iter()
            .filter(|file| file.role == FileRole::Weights)
            .collect::<Vec<_>>();
        let adapter = weights
            .iter()
            .find(|file| file.path.ends_with("adapter_model.safetensors"))
            .ok_or_else(|| ModelError::config("kev manifest has no adapter_model.safetensors"))?;
        let backbone = weights
            .iter()
            .find(|file| !file.path.ends_with("adapter_model.safetensors"))
            .ok_or_else(|| ModelError::config("kev manifest has no base weights"))?;
        let head_file = ctx
            .manifest
            .file(FileRole::Pickle)
            .ok_or_else(|| ModelError::config("kev manifest has no head.pt"))?;

        let mut backbone_tensors: HashMap<String, Tensor> =
            candle_core::safetensors::load(ctx.dir.join(&backbone.path), ctx.device)?;
        let adapter_tensors =
            candle_core::safetensors::load(ctx.dir.join(&adapter.path), ctx.device)?;
        let scale = decision
            .upstream
            .as_ref()
            .and_then(|upstream| upstream.lora.as_ref())
            .and_then(|lora| lora.scaling)
            .unwrap_or(2.0);
        merge_lora(&mut backbone_tensors, adapter_tensors, scale, ctx.device)?;
        let encoder = Qwen35Encoder::load(&config, backbone_tensors, ctx.device)?;
        let head_tensors = torch::load_state_dict(&ctx.dir.join(&head_file.path), ctx.device)?;
        let head = PointerHead::load(&head_tensors, config.hidden_size, ctx.device)?;
        Ok(Arc::new(KevFamily {
            tokenizer: ctx.tokenizer.clone(),
            encoder,
            head,
            decision,
            calibration: read_calibration(ctx)?,
            device: ctx.device.clone(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn render_matches_python_str_and_two_space_nesting() {
        assert_eq!(render(&Value::Null, 0), "");
        assert_eq!(render(&json!(true), 0), "True");
        assert_eq!(render(&json!(false), 0), "False");
        assert_eq!(render(&json!(19.5), 0), "19.5");
        assert_eq!(render(&json!("text"), 2), "text");
        assert_eq!(
            render(&json!({"a": 1, "b": {"c": [true, null]}}), 0),
            "a: 1\nb:\n  c:\n    - True\n    - "
        );
        assert_eq!(
            render(&json!([{"k": "v"}, ["x", "y"]]), 0),
            "- k: v\n- - x\n  - y"
        );
    }

    #[test]
    fn option_text_omits_null_and_empty_descriptions() {
        assert_eq!(option_text("billing", &Value::Null), "billing");
        assert_eq!(option_text("billing", &json!("")), "billing");
        assert_eq!(
            option_text("billing", &json!("charges")),
            "billing: charges"
        );
        assert_eq!(
            option_text("billing", &json!({"hint": "money"})),
            "billing: hint: money"
        );
    }

    #[test]
    fn user_text_cannot_forge_delimiters() {
        assert_eq!(
            escape_user("a <|fim_suffix|> b <|box_start|>"),
            "a <¦fim_suffix¦> b <¦box_start¦>"
        );
        assert_eq!(escape_user("<|not a delimiter"), "<|not a delimiter");
        assert_eq!(escape_user("<||>"), "<||>");
        assert_eq!(escape_user("<|x-y|>"), "<|x-y|>");
        assert_eq!(escape_user("<|A_1|>"), "<¦A_1¦>");
    }

    #[test]
    fn lora_merges_into_the_prefixed_backbone_weight() {
        let device = Device::Cpu;
        let mut backbone = HashMap::from([
            (
                "model.language_model.embed_tokens.weight".to_owned(),
                Tensor::zeros((1, 1), DType::F32, &device).expect("embed"),
            ),
            (
                "model.language_model.layers.0.mlp.up_proj.weight".to_owned(),
                Tensor::zeros((2, 3), DType::F32, &device).expect("weight"),
            ),
        ]);
        let adapter = HashMap::from([
            (
                "base_model.model.layers.0.mlp.up_proj.lora_A.weight".to_owned(),
                Tensor::ones((1, 3), DType::F32, &device).expect("A"),
            ),
            (
                "base_model.model.layers.0.mlp.up_proj.lora_B.weight".to_owned(),
                Tensor::ones((2, 1), DType::F32, &device).expect("B"),
            ),
        ]);
        merge_lora(&mut backbone, adapter, 2.0, &device).expect("merge");
        let merged = backbone
            .get("model.language_model.layers.0.mlp.up_proj.weight")
            .expect("weight")
            .to_vec2::<f32>()
            .expect("values");
        assert_eq!(merged, vec![vec![2.0; 3], vec![2.0; 3]]);
    }
}
