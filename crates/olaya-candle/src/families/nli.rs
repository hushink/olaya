//! Zero-shot NLI cross-encoder (`nli-pairs-v1`), ModernBERT variant.
//!
//! Every option becomes a hypothesis; one encoder row scores one
//! `(state, hypothesis)` pair:
//!
//! ```text
//! [CLS] <state> [SEP] <hypothesis> [SEP]
//! ```
//!
//! The state is cut from the right (`only_first`). Option logits are the rows'
//! entailment logits; a noul without both descriptions is one row read as
//! `[not_entailment, entailment]`.

use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use crate::encoders::deberta::{DebertaConfig, DebertaEncoder};
use crate::encoders::modernbert::{EncoderConfig, ModernBertEncoder};
use crate::{Family, FamilyFactory, LoadContext, ModelError};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{
    LayerNorm, Linear, Module, VarBuilder, layer_norm, layer_norm_no_bias, linear, linear_no_bias,
};
use indexmap::IndexMap;
use olaya_core::{
    Calibration, CalibrationFile, Content, DecisionError, FileRole, Question, State,
    SystemOneResponse, TokenEncoder, Usage, answer_from_logits, dumps_content,
};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

/// Padded tokens per forward pass. A single longer row still runs alone.
const TOKEN_BUDGET: usize = 8_192;
/// Rows per forward pass.
const MAX_ROWS: usize = 64;
/// `[CLS]`, `[SEP]`, `[SEP]`.
const PAIR_SPECIALS: usize = 3;

fn default_max_len() -> usize {
    512
}

fn one() -> usize {
    1
}

fn default_choice_with_description() -> String {
    "{description}".to_owned()
}

fn default_choice_without_description() -> String {
    "This text is about {label}.".to_owned()
}

fn default_score() -> String {
    "{description}".to_owned()
}

fn default_noul_pair() -> String {
    "{description}".to_owned()
}

fn default_noul_single() -> String {
    "{instructions}".to_owned()
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct NliSpecials {
    cls: u32,
    sep: u32,
    pad: u32,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct NliClasses {
    #[serde(default)]
    entailment: usize,
    #[serde(default = "one")]
    not_entailment: usize,
}

impl Default for NliClasses {
    fn default() -> Self {
        Self {
            entailment: 0,
            not_entailment: 1,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct ChoiceTemplates {
    #[serde(default = "default_choice_with_description")]
    with_description: String,
    #[serde(default = "default_choice_without_description")]
    without_description: String,
}

impl Default for ChoiceTemplates {
    fn default() -> Self {
        Self {
            with_description: default_choice_with_description(),
            without_description: default_choice_without_description(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct NliTemplates {
    #[serde(default)]
    choice: ChoiceTemplates,
    #[serde(default = "default_score")]
    score: String,
    #[serde(default = "default_noul_pair")]
    noul_pair: String,
    #[serde(default = "default_noul_single")]
    noul_single: String,
}

impl NliTemplates {
    fn deberta() -> Self {
        let option = "The answer to \"{instructions}\" is: {option}".to_owned();
        Self {
            choice: ChoiceTemplates {
                with_description: option.clone(),
                without_description: option,
            },
            score: "The answer to \"{instructions}\" is: {description}".to_owned(),
            noul_pair: "{description}".to_owned(),
            noul_single: "{instructions}".to_owned(),
        }
    }
}

impl Default for NliTemplates {
    fn default() -> Self {
        Self {
            choice: ChoiceTemplates::default(),
            score: default_score(),
            noul_pair: default_noul_pair(),
            noul_single: default_noul_single(),
        }
    }
}

/// The optional `decision.json` layer. Missing fields fall back to the
/// ModernBERT zeroshot-v2.0 template set above.
#[derive(Clone, Debug, Deserialize)]
struct NliDecision {
    #[serde(default = "default_max_len")]
    max_len: usize,
    #[serde(default)]
    special_tokens: Option<NliSpecials>,
    #[serde(default)]
    classes: NliClasses,
    #[serde(default)]
    templates: NliTemplates,
    #[serde(default)]
    classifier_pooling: Option<String>,
}

impl NliDecision {
    fn defaults_for(arch: &str) -> Self {
        let templates = match arch {
            "deberta-v3" | "deberta-v2" => NliTemplates::deberta(),
            _ => NliTemplates::default(),
        };
        Self {
            max_len: default_max_len(),
            special_tokens: None,
            classes: NliClasses::default(),
            templates,
            classifier_pooling: None,
        }
    }
}

impl Default for NliDecision {
    fn default() -> Self {
        Self::defaults_for("modernbert")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// One row per option; read its entailment logit.
    Entail,
    /// One row; read `[not_entailment, entailment]` as `[false, true]`.
    Single,
}

#[derive(Clone, Debug)]
struct QuestionRows {
    rows: Vec<Vec<u32>>,
    mode: Mode,
    truncated: bool,
}

struct NliLayout {
    max_len: usize,
    special: NliSpecials,
    classes: NliClasses,
    templates: NliTemplates,
}

impl NliLayout {
    fn validate(&self, num_labels: usize) -> Result<(), ModelError> {
        if self.max_len < PAIR_SPECIALS {
            return Err(ModelError::config(format!(
                "nli max_len {} is shorter than the three pair specials",
                self.max_len
            )));
        }
        if self.classes.entailment == self.classes.not_entailment {
            return Err(ModelError::config(
                "nli entailment and not_entailment share a class column",
            ));
        }
        if self.classes.entailment >= num_labels || self.classes.not_entailment >= num_labels {
            return Err(ModelError::config(format!(
                "nli classes {:?} exceed the classifier's {num_labels} labels",
                (self.classes.entailment, self.classes.not_entailment)
            )));
        }
        for (template, names) in self.templates_with_fields() {
            let fields: Vec<(&str, &str)> = names.iter().map(|&name| (name, "")).collect();
            fill(template, &fields).map_err(ModelError::Decision)?;
        }
        Ok(())
    }

    fn templates_with_fields(&self) -> [(&str, &'static [&'static str]); 5] {
        [
            (
                &self.templates.choice.with_description,
                &["instructions", "label", "description", "option"],
            ),
            (
                &self.templates.choice.without_description,
                &["instructions", "label", "description", "option"],
            ),
            (
                &self.templates.score,
                &["instructions", "level", "description"],
            ),
            (&self.templates.noul_pair, &["instructions", "description"]),
            (&self.templates.noul_single, &["instructions"]),
        ]
    }

    /// Hypotheses for one question, in option order.
    fn hypotheses(
        &self,
        qid: &str,
        question: &Question,
    ) -> Result<(Vec<String>, Mode), DecisionError> {
        let instructions = question.instructions_or_id(qid);
        let templates = &self.templates;
        match question {
            Question::Choice(choice) => {
                let mut hypotheses = Vec::with_capacity(choice.criteria.len());
                for (label, value) in &choice.criteria {
                    let description = if matches!(value, Content::Null) {
                        String::new()
                    } else {
                        render_content(value)
                    };
                    let option = if description.is_empty() {
                        label.clone()
                    } else {
                        format!("{label}: {description}")
                    };
                    let template = if description.is_empty() {
                        &templates.choice.without_description
                    } else {
                        &templates.choice.with_description
                    };
                    hypotheses.push(fill(
                        template,
                        &[
                            ("instructions", &instructions),
                            ("label", label),
                            ("description", &description),
                            ("option", &option),
                        ],
                    )?);
                }
                Ok((hypotheses, Mode::Entail))
            }
            Question::Score(score) => {
                let mut hypotheses = Vec::with_capacity(score.criteria.len());
                for (index, level) in score.criteria.iter().enumerate() {
                    let level_text = index.to_string();
                    let description = render_content(level);
                    hypotheses.push(fill(
                        &templates.score,
                        &[
                            ("instructions", &instructions),
                            ("level", &level_text),
                            ("description", &description),
                        ],
                    )?);
                }
                Ok((hypotheses, Mode::Entail))
            }
            Question::Noul(noul) => {
                let (is_false, is_true) = match &noul.criteria {
                    Some(criteria) => (described(&criteria.is_false), described(&criteria.is_true)),
                    None => (None, None),
                };
                match (is_false, is_true) {
                    (Some(is_false), Some(is_true)) => Ok((
                        vec![
                            fill(
                                &templates.noul_pair,
                                &[("instructions", &instructions), ("description", &is_false)],
                            )?,
                            fill(
                                &templates.noul_pair,
                                &[("instructions", &instructions), ("description", &is_true)],
                            )?,
                        ],
                        Mode::Entail,
                    )),
                    _ => Ok((
                        vec![fill(
                            &templates.noul_single,
                            &[("instructions", &instructions)],
                        )?],
                        Mode::Single,
                    )),
                }
            }
            Question::Box(_) => Err(DecisionError::unsupported(
                "nli does not answer box questions",
            )),
        }
    }

    /// Encode one question's rows against the tokenized state.
    fn encode(
        &self,
        premise: &[u32],
        qid: &str,
        question: &Question,
        tokenizer: &dyn TokenEncoder,
    ) -> Result<QuestionRows, DecisionError> {
        let (hypotheses, mode) = self.hypotheses(qid, question)?;
        let mut rows = Vec::with_capacity(hypotheses.len());
        let mut truncated = false;
        for hypothesis in &hypotheses {
            let ids = tokenizer.encode(hypothesis)?;
            let room = self
                .max_len
                .checked_sub(ids.len() + PAIR_SPECIALS)
                .ok_or_else(|| {
                    DecisionError::invalid(format!(
                        "question {qid:?}: its hypothesis is {} tokens and leaves no room for the state \
                         (the model reads {} tokens per state and hypothesis); shorten the instructions or options",
                        ids.len(),
                        self.max_len
                    ))
                })?;
            let keep = if premise.len() <= room {
                premise.len()
            } else if room > 0 {
                truncated = true;
                room
            } else {
                return Err(DecisionError::invalid(format!(
                    "question {qid:?}: its hypothesis is {} tokens and leaves no room for the state \
                     (the model reads {} tokens per state and hypothesis); shorten the instructions or options",
                    ids.len(),
                    self.max_len
                )));
            };
            let mut row = Vec::with_capacity(keep + ids.len() + PAIR_SPECIALS);
            row.push(self.special.cls);
            row.extend_from_slice(&premise[..keep]);
            row.push(self.special.sep);
            row.extend_from_slice(&ids);
            row.push(self.special.sep);
            rows.push(row);
        }
        Ok(QuestionRows {
            rows,
            mode,
            truncated,
        })
    }

    /// Option logits from a question's class logits.
    fn option_logits(&self, mode: Mode, rows: &[Vec<f32>]) -> Result<Vec<f32>, DecisionError> {
        let (entailment, not_entailment) = (self.classes.entailment, self.classes.not_entailment);
        let width = entailment.max(not_entailment) + 1;
        if rows.iter().any(|row| row.len() < width) {
            return Err(DecisionError::invalid(format!(
                "classifier output is narrower than the configured class columns ({width})"
            )));
        }
        Ok(match mode {
            Mode::Entail => rows.iter().map(|row| row[entailment]).collect(),
            Mode::Single => rows
                .first()
                .map(|row| vec![row[not_entailment], row[entailment]])
                .unwrap_or_default(),
        })
    }
}

/// A description as the reference tests it: absent for missing, null or `""`.
fn described(value: &Option<Content>) -> Option<String> {
    match value {
        None | Some(Content::Null) => None,
        Some(Content::String(text)) if text.is_empty() => None,
        Some(value) => Some(render_content(value)),
    }
}

fn render_content(content: &Content) -> String {
    match content {
        Content::String(text) => text.clone(),
        other => dumps_content(other),
    }
}

/// Substitute `{field}` placeholders in one left-to-right pass.
fn fill(template: &str, fields: &[(&str, &str)]) -> Result<String, DecisionError> {
    let mut out = String::with_capacity(template.len() + 64);
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let end = after
            .find(|character: char| !(character.is_alphanumeric() || character == '_'))
            .unwrap_or(after.len());
        if end > 0 && after[end..].starts_with('}') {
            let name = &after[..end];
            let value = fields
                .iter()
                .find_map(|&(key, value)| (key == name).then_some(value))
                .ok_or_else(|| {
                    DecisionError::invalid(format!(
                        "template {template:?} uses unknown field {{{name}}}"
                    ))
                })?;
            out.push_str(value);
            rest = &after[end + 1..];
        } else {
            out.push('{');
            rest = after;
        }
    }
    out.push_str(rest);
    Ok(out)
}

#[derive(Clone, Copy, Debug)]
enum Pooling {
    Mean,
    Cls,
}

struct NliHead {
    dense: Linear,
    norm: LayerNorm,
    classifier: Linear,
    pooling: Pooling,
}

impl NliHead {
    fn load(
        tensors: &HashMap<String, Tensor>,
        encoder: &EncoderConfig,
        pooling: Pooling,
        num_labels: usize,
        device: &Device,
    ) -> Result<Self, ModelError> {
        let vb = VarBuilder::from_tensors(tensors.clone(), DType::F32, device);
        let hidden = encoder.hidden_size;
        let dense = if tensors.contains_key("head.dense.bias") {
            linear(hidden, hidden, vb.pp("head.dense"))?
        } else {
            linear_no_bias(hidden, hidden, vb.pp("head.dense"))?
        };
        let norm = if tensors.contains_key("head.norm.bias") {
            layer_norm(hidden, encoder.layer_norm_eps, vb.pp("head.norm"))?
        } else {
            layer_norm_no_bias(hidden, encoder.layer_norm_eps, vb.pp("head.norm"))?
        };
        let classifier = if tensors.contains_key("classifier.bias") {
            linear(hidden, num_labels, vb.pp("classifier"))?
        } else {
            linear_no_bias(hidden, num_labels, vb.pp("classifier"))?
        };
        Ok(Self {
            dense,
            norm,
            classifier,
            pooling,
        })
    }

    fn forward(&self, hidden: &Tensor, mask: &Tensor) -> Result<Tensor, ModelError> {
        let pooled = match self.pooling {
            Pooling::Cls => hidden.i((.., 0, ..))?.contiguous()?,
            Pooling::Mean => {
                let mask = mask.unsqueeze(2)?;
                let sum = hidden.broadcast_mul(&mask)?.sum(1)?;
                let count = mask.sum(1)?;
                sum.broadcast_div(&count)?
            }
        };
        let pooled = self.dense.forward(&pooled)?.gelu_erf()?;
        let pooled = self.norm.forward(&pooled)?;
        Ok(self.classifier.forward(&pooled)?)
    }
}

/// `DebertaV2ForSequenceClassification`'s `ContextPooler` plus classifier.
struct DebertaHead {
    dense: Linear,
    classifier: Linear,
}

impl DebertaHead {
    fn load(
        tensors: &HashMap<String, Tensor>,
        config: &DebertaConfig,
        num_labels: usize,
        device: &Device,
    ) -> Result<Self, ModelError> {
        let vb = VarBuilder::from_tensors(tensors.clone(), DType::F32, device);
        let hidden = config.hidden_size;
        let dense = linear(hidden, hidden, vb.pp("pooler.dense"))?;
        let classifier = if tensors.contains_key("classifier.bias") {
            linear(hidden, num_labels, vb.pp("classifier"))?
        } else {
            linear_no_bias(hidden, num_labels, vb.pp("classifier"))?
        };
        Ok(Self { dense, classifier })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor, ModelError> {
        let pooled = hidden.i((.., 0, ..))?.contiguous()?;
        let pooled = self.dense.forward(&pooled)?.gelu_erf()?;
        Ok(self.classifier.forward(&pooled)?)
    }
}

enum EncoderKind {
    Modern(ModernBertEncoder),
    Deberta(Box<DebertaEncoder>),
}

impl EncoderKind {
    fn forward(&self, ids: &Tensor, mask: &Tensor) -> Result<Tensor, ModelError> {
        match self {
            Self::Modern(encoder) => encoder.forward(ids, mask),
            Self::Deberta(encoder) => encoder.forward(ids, mask),
        }
    }
}

enum Head {
    Modern(NliHead),
    Deberta(DebertaHead),
}

impl Head {
    fn forward(&self, hidden: &Tensor, mask: &Tensor) -> Result<Tensor, ModelError> {
        match self {
            Self::Modern(head) => head.forward(hidden, mask),
            Self::Deberta(head) => head.forward(hidden),
        }
    }
}

struct NliFamily {
    tokenizer: Arc<dyn TokenEncoder>,
    encoder: EncoderKind,
    head: Head,
    layout: NliLayout,
    calibration: Calibration,
    device: Device,
}

impl NliFamily {
    fn run_rows(&self, rows: &[Vec<u32>]) -> Result<Vec<Vec<f32>>, ModelError> {
        let mut out = vec![Vec::new(); rows.len()];
        if rows.is_empty() {
            return Ok(out);
        }
        let lengths: Vec<usize> = rows.iter().map(Vec::len).collect();
        for range in batches(&lengths, TOKEN_BUDGET, MAX_ROWS) {
            let batch = &rows[range.clone()];
            let sequence = batch.iter().map(Vec::len).max().unwrap_or(1).max(1);
            let count = batch.len();
            let pad = self.layout.special.pad;
            let mut ids = vec![pad; count * sequence];
            let mut mask = vec![0f32; count * sequence];
            for (row_index, row) in batch.iter().enumerate() {
                let start = row_index * sequence;
                ids[start..start + row.len()].copy_from_slice(row);
                mask[start..start + row.len()].fill(1.0);
            }
            let ids = Tensor::from_vec(ids, (count, sequence), &self.device)?;
            let mask = Tensor::from_vec(mask, (count, sequence), &self.device)?;
            let hidden = self.encoder.forward(&ids, &mask)?.to_dtype(DType::F32)?;
            let logits = self.head.forward(&hidden, &mask)?;
            let values = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            if values.len() != count {
                return Err(ModelError::config(format!(
                    "classifier returned {} rows for a batch of {count}",
                    values.len()
                )));
            }
            for (offset, value) in values.into_iter().enumerate() {
                out[range.start + offset] = value;
            }
        }
        Ok(out)
    }
}

impl Family for NliFamily {
    fn id(&self) -> &'static str {
        "nli"
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
        if questions.iter().any(|(_, question)| question.is_ordinal()) {
            return Err(DecisionError::Unsupported {
                message: "nli does not support ordinal choices".to_owned(),
            }
            .into());
        }
        let state_text = match state.as_content() {
            Content::String(text) => text.clone(),
            content => dumps_content(content),
        };
        let premise = self.tokenizer.encode(&state_text)?;
        let mut encodings = Vec::with_capacity(questions.len());
        let mut rows = Vec::new();
        for (qid, question) in questions.iter() {
            let encoding = self
                .layout
                .encode(&premise, qid, question, self.tokenizer.as_ref())?;
            if encoding.truncated {
                return Err(DecisionError::StateTruncated.into());
            }
            rows.extend(encoding.rows.iter().cloned());
            encodings.push(encoding);
        }
        let row_logits = self.run_rows(&rows)?;

        let mut answers = IndexMap::with_capacity(questions.len());
        let mut cursor = 0;
        for ((qid, question), encoding) in questions.iter().zip(&encodings) {
            let count = encoding.rows.len();
            let logits = self
                .layout
                .option_logits(encoding.mode, &row_logits[cursor..cursor + count])?;
            cursor += count;
            let answer = match question {
                Question::Choice(choice) => {
                    let labels: Vec<String> = choice.criteria.keys().cloned().collect();
                    answer_from_logits(
                        question,
                        &labels,
                        &[],
                        &logits,
                        &self.calibration,
                        premise.len(),
                    )?
                }
                Question::Score(score) => answer_from_logits(
                    question,
                    &[],
                    &score.criteria,
                    &logits,
                    &self.calibration,
                    premise.len(),
                )?,
                Question::Noul(_) => answer_from_logits(
                    question,
                    &["false".to_owned(), "true".to_owned()],
                    &[],
                    &logits,
                    &self.calibration,
                    premise.len(),
                )?,
                Question::Box(_) => {
                    return Err(ModelError::unsupported("nli does not answer box questions"));
                }
            };
            answers.insert(qid.clone(), answer);
        }
        let input_tokens = encodings
            .iter()
            .flat_map(|encoding| &encoding.rows)
            .map(|row| row.len() as u64)
            .sum();
        Ok(SystemOneResponse::new(
            self.id(),
            answers,
            Usage::new(input_tokens, 0),
        ))
    }
}

/// Group row ranges so padded tokens stay within `budget`; one oversized row
/// still runs alone.
fn batches(lengths: &[usize], budget: usize, max_rows: usize) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut longest = 0;
    for (index, &length) in lengths.iter().enumerate() {
        let count = index - start + 1;
        let next_longest = longest.max(length);
        if index > start && (count > max_rows || count * next_longest > budget) {
            out.push(start..index);
            start = index;
            longest = length;
        } else {
            longest = next_longest;
        }
    }
    if start < lengths.len() {
        out.push(start..lengths.len());
    }
    out
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, ModelError> {
    let bytes = std::fs::read(path)?;
    serde_json::from_slice(&bytes)
        .map_err(|error| ModelError::config(format!("{}: {error}", path.display())))
}

fn read_decision(ctx: &LoadContext<'_>, arch: &str) -> Result<NliDecision, ModelError> {
    if let Some(file) = ctx.manifest.file(FileRole::Decision) {
        return read_json(&ctx.dir.join(&file.path));
    }
    let path = ctx.dir.join("decision.json");
    if path.is_file() {
        read_json(&path)
    } else {
        Ok(NliDecision::defaults_for(arch))
    }
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

fn resolve_specials(
    config: &EncoderConfig,
    tokenizer: &dyn TokenEncoder,
) -> Result<NliSpecials, ModelError> {
    let cls = config
        .cls_token_id
        .or_else(|| tokenizer.token_id("[CLS]"))
        .or_else(|| tokenizer.token_id("<bos>"))
        .ok_or_else(|| ModelError::config("nli tokenizer has no [CLS]/<bos> token"))?;
    let sep = config
        .sep_token_id
        .or_else(|| tokenizer.token_id("[SEP]"))
        .or_else(|| tokenizer.token_id("<eos>"))
        .ok_or_else(|| ModelError::config("nli tokenizer has no [SEP]/<eos> token"))?;
    Ok(NliSpecials {
        cls,
        sep,
        pad: config.pad_token_id,
    })
}

fn resolve_deberta_specials(
    config: &DebertaConfig,
    tokenizer: &dyn TokenEncoder,
) -> Result<NliSpecials, ModelError> {
    let cls = tokenizer
        .token_id("[CLS]")
        .or_else(|| tokenizer.token_id("<s>"))
        .ok_or_else(|| ModelError::config("nli DeBERTa tokenizer has no [CLS]/<s> token"))?;
    let sep = tokenizer
        .token_id("[SEP]")
        .or_else(|| tokenizer.token_id("</s>"))
        .ok_or_else(|| ModelError::config("nli DeBERTa tokenizer has no [SEP]/</s> token"))?;
    let pad = config
        .pad_token_id
        .map(|value| value as u32)
        .or_else(|| tokenizer.token_id("[PAD]"))
        .unwrap_or(0);
    Ok(NliSpecials { cls, sep, pad })
}

fn pooling(value: &str) -> Result<Pooling, ModelError> {
    match value {
        "mean" => Ok(Pooling::Mean),
        "cls" => Ok(Pooling::Cls),
        other => Err(ModelError::config(format!(
            "unsupported nli classifier_pooling {other:?}; expected mean or cls"
        ))),
    }
}

/// Loads an NLI checkpoint: ModernBERT or DeBERTa-v2/v3.
pub struct Factory;

impl FamilyFactory for Factory {
    fn family(&self) -> &'static str {
        "nli"
    }

    fn load(&self, ctx: &LoadContext<'_>) -> Result<Arc<dyn Family>, ModelError> {
        let arch = ctx.manifest.arch.as_str();
        if !matches!(arch, "modernbert" | "deberta-v3" | "deberta-v2") {
            return Err(ModelError::config(format!(
                "nli serves modernbert/deberta-v3/deberta-v2, got {arch:?}"
            )));
        }
        if ctx.manifest.layout != "nli-pairs-v1" {
            return Err(ModelError::config(format!(
                "nli serves layout nli-pairs-v1, got {:?}",
                ctx.manifest.layout
            )));
        }
        let config_file = ctx
            .manifest
            .file(FileRole::Config)
            .ok_or_else(|| ModelError::config("nli manifest has no config file"))?;
        let config_path = ctx.dir.join(&config_file.path);
        let config_value: Value = read_json(&config_path)?;

        let decision = read_decision(ctx, arch)?;
        let num_labels = config_value
            .get("num_labels")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .or_else(|| {
                config_value
                    .get("id2label")
                    .and_then(Value::as_object)
                    .map(|labels| labels.len())
            })
            .unwrap_or(2);

        let modern_config = if arch == "modernbert" {
            Some(
                serde_json::from_value::<EncoderConfig>(config_value.clone()).map_err(|error| {
                    ModelError::config(format!("{}: {error}", config_path.display()))
                })?,
            )
        } else {
            None
        };
        let deberta_config = if arch == "modernbert" {
            None
        } else {
            Some(
                serde_json::from_value::<DebertaConfig>(config_value.clone()).map_err(|error| {
                    ModelError::config(format!("{}: {error}", config_path.display()))
                })?,
            )
        };

        let special = match decision.special_tokens {
            Some(special) => special,
            None => match (&modern_config, &deberta_config) {
                (Some(config), _) => resolve_specials(config, ctx.tokenizer.as_ref())?,
                (_, Some(config)) => resolve_deberta_specials(config, ctx.tokenizer.as_ref())?,
                _ => unreachable!("one config is parsed for each supported arch"),
            },
        };

        let layout = NliLayout {
            max_len: decision.max_len,
            special,
            classes: decision.classes,
            templates: decision.templates,
        };
        layout.validate(num_labels)?;

        let weights = ctx
            .manifest
            .file(FileRole::Weights)
            .ok_or_else(|| ModelError::config("nli manifest has no weights file"))?;
        let weights_path = ctx.dir.join(&weights.path);
        let mut tensors: HashMap<String, Tensor> =
            candle_core::safetensors::load(&weights_path, ctx.device)?;

        let (encoder, head) = if let Some(encoder_config) = &modern_config {
            let activation = config_value
                .get("classifier_activation")
                .and_then(Value::as_str)
                .unwrap_or("gelu");
            if activation != "gelu" {
                return Err(ModelError::config(format!(
                    "unsupported nli classifier_activation {activation:?}; expected gelu"
                )));
            }
            let pooling = pooling(
                decision
                    .classifier_pooling
                    .as_deref()
                    .or_else(|| {
                        config_value
                            .get("classifier_pooling")
                            .and_then(Value::as_str)
                    })
                    .unwrap_or("mean"),
            )?;
            let encoder = ModernBertEncoder::load(
                VarBuilder::from_tensors(tensors.clone(), ctx.compute_dtype(), ctx.device),
                encoder_config,
            )?;
            let head = NliHead::load(&tensors, encoder_config, pooling, num_labels, ctx.device)?;
            (EncoderKind::Modern(encoder), Head::Modern(head))
        } else {
            for tensor in tensors.values_mut() {
                if tensor.dtype() != DType::F32 {
                    *tensor = tensor.to_dtype(DType::F32)?;
                }
            }
            let deberta_config = deberta_config.expect("parsed for every non-ModernBERT arch");
            let activation = deberta_config
                .pooler_hidden_act
                .as_deref()
                .unwrap_or("gelu");
            if activation != "gelu" {
                return Err(ModelError::config(format!(
                    "unsupported nli pooler_hidden_act {activation:?}; expected gelu"
                )));
            }
            let encoder = DebertaEncoder::load(&tensors, &deberta_config, ctx.device)?;
            let head = DebertaHead::load(&tensors, &deberta_config, num_labels, ctx.device)?;
            (EncoderKind::Deberta(Box::new(encoder)), Head::Deberta(head))
        };

        Ok(Arc::new(NliFamily {
            tokenizer: ctx.tokenizer.clone(),
            encoder,
            head,
            layout,
            calibration: read_calibration(ctx)?,
            device: ctx.device.clone(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Words;

    impl TokenEncoder for Words {
        fn encode(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
            Ok(text
                .split_whitespace()
                .map(|word| word.len() as u32)
                .collect())
        }
    }

    fn layout(max_len: usize) -> NliLayout {
        NliLayout {
            max_len,
            special: NliSpecials {
                cls: 101,
                sep: 102,
                pad: 0,
            },
            classes: NliClasses::default(),
            templates: NliTemplates::default(),
        }
    }

    fn question(value: Value) -> Question {
        serde_json::from_value(value).expect("question")
    }

    #[test]
    fn deberta_defaults_use_its_templates() {
        let decision = NliDecision::defaults_for("deberta-v3");
        assert_eq!(
            decision.templates.choice.with_description,
            "The answer to \"{instructions}\" is: {option}"
        );
        assert_eq!(
            decision.templates.score,
            "The answer to \"{instructions}\" is: {description}"
        );
        let modern = NliDecision::defaults_for("modernbert");
        assert_eq!(modern.templates.choice.with_description, "{description}");
    }

    #[test]
    fn fills_in_one_pass() {
        let filled = fill(
            "{instructions} / {option}",
            &[("instructions", "{option}"), ("option", "x")],
        )
        .expect("fill");
        assert_eq!(filled, "{option} / x");
        assert!(fill("{label}", &[("instructions", "")]).is_err());
    }

    #[test]
    fn hypotheses_follow_the_templates() {
        let layout = layout(512);
        let choice = question(json!({
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "payments", "other": "", "tier": {"sla_h": 24}}
        }));
        let (hypotheses, mode) = layout.hypotheses("q", &choice).expect("hypotheses");
        assert_eq!(mode, Mode::Entail);
        assert_eq!(
            hypotheses,
            ["payments", "This text is about other.", "{\"sla_h\": 24}",]
        );

        let score = question(json!({
            "type": "score",
            "instructions": "How bad?",
            "criteria": ["fine", "3"]
        }));
        let (hypotheses, _) = layout.hypotheses("q", &score).expect("score hypotheses");
        assert_eq!(hypotheses, ["fine", "3"]);

        let pair = question(json!({
            "type": "noul",
            "instructions": "Angry?",
            "criteria": {
                "TRUE": "The customer is angry.",
                "false": "The customer is calm."
            }
        }));
        assert_eq!(
            layout.hypotheses("q", &pair).expect("pair"),
            (
                vec![
                    "The customer is calm.".to_owned(),
                    "The customer is angry.".to_owned()
                ],
                Mode::Entail
            )
        );

        let single = question(json!({
            "type": "noul",
            "instructions": "It is spam.",
            "criteria": {"true": "yes", "false": ""}
        }));
        assert_eq!(
            layout.hypotheses("q", &single).expect("single"),
            (vec!["It is spam.".to_owned()], Mode::Single)
        );
    }

    #[test]
    fn truncates_only_the_state() {
        let layout = layout(10);
        let question = question(json!({"type": "noul", "instructions": "h h h"}));
        let pairs = layout
            .encode(&[7, 8], "q", &question, &Words)
            .expect("encode");
        assert_eq!(pairs.rows, [vec![101, 7, 8, 102, 1, 1, 1, 102]]);
        assert!(!pairs.truncated);

        let pairs = layout
            .encode(&[1, 2, 3, 4, 5, 6], "q", &question, &Words)
            .expect("encode");
        assert_eq!(pairs.rows, [vec![101, 1, 2, 3, 4, 102, 1, 1, 1, 102]]);
        assert!(pairs.truncated);
    }

    #[test]
    fn rejects_a_hypothesis_that_fills_the_window() {
        let layout = layout(10);
        let question = |words: usize| {
            question(json!({
                "type": "noul",
                "instructions": vec!["h"; words].join(" ")
            }))
        };
        assert_eq!(
            layout
                .encode(&[5; 9], "q", &question(6), &Words)
                .expect("fits")
                .rows[0]
                .len(),
            10
        );
        assert_eq!(
            layout
                .encode(&[], "q", &question(7), &Words)
                .expect("fits")
                .rows[0]
                .len(),
            10
        );
        assert!(layout.encode(&[5], "q", &question(7), &Words).is_err());
        assert!(layout.encode(&[], "q", &question(8), &Words).is_err());
    }

    #[test]
    fn reads_option_logits() {
        let layout = layout(512);
        let rows = [vec![1.0, -1.0], vec![0.5, 2.0]];
        assert_eq!(
            layout.option_logits(Mode::Entail, &rows).expect("entail"),
            [1.0, 0.5]
        );
        assert_eq!(
            layout.option_logits(Mode::Single, &rows).expect("single"),
            [-1.0, 1.0]
        );
    }

    #[test]
    fn batches_respect_the_budget() {
        let lengths = [100, 100, 100, 10];
        assert_eq!(batches(&lengths, 250, 8), vec![0..2, 2..4]);
        assert_eq!(batches(&lengths, 1_000, 2), vec![0..2, 2..4]);
        assert_eq!(batches(&[512], 100, 8), vec![0..1]);
        assert!(batches(&[], 100, 8).is_empty());
    }
}
