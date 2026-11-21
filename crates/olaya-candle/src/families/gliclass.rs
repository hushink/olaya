//! GLiClass uni-encoder (`gliclass-uni-v1`): v4-edge and instruct-large v1.0.
//!
//! One row per question:
//!
//! ```text
//! [CLS] <<LABEL>>label<<LABEL>>label<<SEP>><prompt><state> [SEP]
//! ```
//!
//! Labels are pooled at their `<<LABEL>>` markers (first token, or the mean of
//! each label's span), the text at `[CLS]`, both projected and scored by an MLP.
//! v4's optional calibrator rescales every label logit from the projected
//! text/label and the row's logit statistics. v1.0 adds learned segment
//! embeddings; they are folded into an expanded word-embedding table so the
//! shared DeBERTa backbone can serve both checkpoints unchanged.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::encoders::deberta::{DebertaConfig, DebertaEncoder};
use crate::encoders::modernbert::{EncoderConfig, ModernBertEncoder};
use crate::families::batch::groups;
use crate::{Family, FamilyFactory, LoadContext, ModelError};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::ops::sigmoid as tensor_sigmoid;
use candle_nn::var_builder::{Rename, Renamer};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder, layer_norm, linear};
use indexmap::IndexMap;
use olaya_core::{
    Calibration, CalibrationFile, Content, DecisionError, FileRole, Question, State,
    SystemOneResponse, TokenEncoder, Usage, answer_from_logits, dumps_content,
};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

const MAX_LEN: usize = 1024;
const MARKER_TEXT: &str = "<<LABEL>>";
const TEXT_SEP_TEXT: &str = "<<SEP>>";
const EXAMPLE_TEXT: &str = "<<EXAMPLE>>";
const CALIBRATOR_LAYER_NORM_EPS: f64 = 1e-5;

fn default_beta_max() -> f64 {
    3.0
}

fn default_encoder_layer() -> i64 {
    -1
}

#[derive(Clone, Debug, Deserialize)]
struct GliclassConfig {
    #[serde(default)]
    architecture_type: String,
    #[serde(default)]
    class_token_pooling: String,
    #[serde(default)]
    embed_class_token: bool,
    #[serde(default)]
    pooling_strategy: String,
    #[serde(default)]
    projector_hidden_act: String,
    #[serde(default)]
    scorer_type: String,
    #[serde(default)]
    use_segment_embeddings: bool,
    #[serde(default)]
    use_calibrator: bool,
    #[serde(default = "default_beta_max")]
    calibrator_beta_max: f64,
    #[serde(default)]
    calibrator_use_bias: bool,
    #[serde(default)]
    normalize_features: bool,
    #[serde(default)]
    use_lstm: bool,
    #[serde(default)]
    squeeze_layers: bool,
    #[serde(default)]
    layer_wise: bool,
    #[serde(default = "default_encoder_layer")]
    encoder_layer_id: i64,
    #[serde(default)]
    class_token_index: u32,
    #[serde(default)]
    text_token_index: u32,
    #[serde(default)]
    example_token_index: u32,
    #[serde(default)]
    hidden_size: usize,
    /// Raw `encoder_config` object; the `model_type` field selects DeBERTa or
    /// ModernBERT.
    encoder_config: Value,
}

enum GliclassEncoderConfig {
    Modern(Box<EncoderConfig>),
    Deberta(Box<DebertaConfig>),
}

impl GliclassEncoderConfig {
    fn name(&self) -> &'static str {
        match self {
            Self::Modern(_) => "modernbert",
            Self::Deberta(_) => "deberta",
        }
    }
}

impl GliclassConfig {
    fn validate(&self) -> Result<(), ModelError> {
        let checks = [
            (self.architecture_type == "uni-encoder", "architecture_type"),
            (
                matches!(self.class_token_pooling.as_str(), "first" | "average"),
                "class_token_pooling=first|average",
            ),
            (self.embed_class_token, "embed_class_token=true"),
            (self.pooling_strategy == "first", "pooling_strategy=first"),
            (
                self.projector_hidden_act == "gelu",
                "projector_hidden_act=gelu",
            ),
            (self.scorer_type == "mlp", "scorer_type=mlp"),
            (!self.normalize_features, "normalize_features=false"),
            (!self.use_lstm, "use_lstm=false"),
            (!self.squeeze_layers, "squeeze_layers=false"),
            (!self.layer_wise, "layer_wise=false"),
            (self.encoder_layer_id == -1, "encoder_layer_id=-1"),
        ];
        for (valid, what) in checks {
            if !valid {
                return Err(ModelError::config(format!(
                    "GLiClass uni-encoder requires {what}"
                )));
            }
        }
        for (name, id) in [
            ("class_token_index", self.class_token_index),
            ("text_token_index", self.text_token_index),
            ("example_token_index", self.example_token_index),
        ] {
            if id == 0 {
                return Err(ModelError::config(format!("GLiClass {name} is missing")));
            }
        }
        Ok(())
    }

    fn encoder_kind(&self) -> Result<GliclassEncoderConfig, ModelError> {
        let model_type = self
            .encoder_config
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match model_type {
            "modernbert" => serde_json::from_value(self.encoder_config.clone())
                .map(|config| GliclassEncoderConfig::Modern(Box::new(config)))
                .map_err(|error| {
                    ModelError::config(format!("GLiClass ModernBERT encoder config: {error}"))
                }),
            "deberta-v2" | "deberta-v3" => serde_json::from_value(self.encoder_config.clone())
                .map(|config| GliclassEncoderConfig::Deberta(Box::new(config)))
                .map_err(|error| {
                    ModelError::config(format!("GLiClass DeBERTa encoder config: {error}"))
                }),
            other => Err(ModelError::config(format!(
                "unsupported GLiClass encoder model_type {other:?}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct GliclassSpecials {
    cls: u32,
    sep: u32,
    label: u32,
    text_sep: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LabelPooling {
    First,
    Average,
}

impl LabelPooling {
    fn parse(value: &str) -> Result<Self, ModelError> {
        match value {
            "first" => Ok(Self::First),
            "average" => Ok(Self::Average),
            other => Err(ModelError::config(format!(
                "unsupported GLiClass class_token_pooling {other:?}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Multi,
    Single,
}

#[derive(Clone)]
struct GliclassRow {
    ids: Vec<u32>,
    markers: Vec<usize>,
    mode: Mode,
    truncated: bool,
}

struct GliclassLayout {
    specials: GliclassSpecials,
    /// Original vocabulary size when segment embeddings are folded into an
    /// expanded word-embedding table.
    segment_vocab: Option<usize>,
}

impl GliclassLayout {
    fn labels_and_prompt(
        &self,
        qid: &str,
        question: &Question,
    ) -> Result<(Vec<String>, String, Mode), DecisionError> {
        let instructions = question.instructions_or_id(qid);
        match question {
            Question::Choice(choice) => {
                let mut labels = Vec::with_capacity(choice.criteria.len());
                for (label, value) in &choice.criteria {
                    let description = if matches!(value, Content::Null) {
                        String::new()
                    } else {
                        render_content(value)
                    };
                    labels.push(if description.is_empty() {
                        label.clone()
                    } else {
                        format!("{label}: {description}")
                    });
                }
                Ok((labels, instructions, Mode::Multi))
            }
            Question::Score(score) => {
                let labels = score
                    .criteria
                    .iter()
                    .enumerate()
                    .map(|(index, level)| format!("level {index}: {}", render_content(level)))
                    .collect();
                Ok((labels, instructions, Mode::Multi))
            }
            Question::Noul(noul) => {
                let (is_false, is_true) = match &noul.criteria {
                    Some(criteria) => (described(&criteria.is_false), described(&criteria.is_true)),
                    None => (None, None),
                };
                match (is_false, is_true) {
                    (Some(is_false), Some(is_true)) => {
                        Ok((vec![is_false, is_true], instructions, Mode::Multi))
                    }
                    _ => Ok((vec![instructions], String::new(), Mode::Single)),
                }
            }
            Question::Box(_) => Err(DecisionError::unsupported(
                "gliclass does not answer box questions",
            )),
        }
    }

    fn encode(
        &self,
        state_text: &str,
        qid: &str,
        question: &Question,
        tokenizer: &dyn TokenEncoder,
    ) -> Result<GliclassRow, DecisionError> {
        let (labels, prompt, mode) = self.labels_and_prompt(qid, question)?;
        let labels = labels
            .iter()
            .map(|label| sanitize(label))
            .collect::<Vec<_>>();
        let prompt = sanitize(&prompt);
        let state = sanitize(state_text);
        let mut text = String::with_capacity(state.len() + prompt.len() + 64);
        for label in &labels {
            text.push_str(MARKER_TEXT);
            text.push_str(label);
        }
        text.push_str(TEXT_SEP_TEXT);
        text.push_str(&prompt);
        text.push_str(&state);

        let raw = tokenizer.encode(&text)?;
        let keep = raw.len().min(MAX_LEN.saturating_sub(2));
        let truncated = raw.len() > keep;
        let mut ids = Vec::with_capacity(keep + 2);
        ids.push(self.specials.cls);
        ids.extend_from_slice(&raw[..keep]);
        ids.push(self.specials.sep);

        let markers = ids
            .iter()
            .enumerate()
            .filter_map(|(position, id)| (*id == self.specials.label).then_some(position))
            .collect::<Vec<_>>();
        if markers.len() != labels.len() {
            return Err(DecisionError::invalid(format!(
                "question {qid:?}: {} labels but {} `{MARKER_TEXT}` tokens fit in {MAX_LEN} tokens; \
                 shorten the instructions or labels",
                labels.len(),
                markers.len()
            )));
        }
        if let Some(vocab) = self.segment_vocab {
            // Upstream's argmax semantics: segment 1 starts at the first
            // `<<SEP>>`; without one, every position is segment 1. Segment 2
            // only exists for `<<EXAMPLE>>` sections, which this layout does
            // not emit.
            let boundary = ids
                .iter()
                .position(|id| *id == self.specials.text_sep)
                .unwrap_or(0);
            for id in &mut ids[boundary..] {
                *id += vocab as u32;
            }
        }
        Ok(GliclassRow {
            ids,
            markers,
            mode,
            truncated,
        })
    }
}

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

fn sanitize(text: &str) -> String {
    text.replace(MARKER_TEXT, " ")
        .replace(TEXT_SEP_TEXT, " ")
        .replace(EXAMPLE_TEXT, " ")
}

struct Projector {
    linear1: Linear,
    linear2: Linear,
}

impl Projector {
    fn load(
        vb: VarBuilder,
        input: usize,
        hidden: usize,
        output: usize,
    ) -> Result<Self, ModelError> {
        Ok(Self {
            linear1: linear(input, hidden, vb.pp("linear_1"))?,
            linear2: linear(hidden, output, vb.pp("linear_2"))?,
        })
    }

    fn forward(&self, features: &Tensor) -> Result<Tensor, ModelError> {
        let features = self.linear1.forward(features)?.gelu_erf()?;
        Ok(self.linear2.forward(&features)?)
    }
}

struct Scorer {
    linear0: Linear,
    linear2: Linear,
    linear4: Linear,
}

impl Scorer {
    fn load(
        vb: VarBuilder,
        input: usize,
        tensors: &HashMap<String, Tensor>,
    ) -> Result<Self, ModelError> {
        let hidden0 = dim(tensors, "model.scorer.mlp.0.weight")?;
        let hidden1 = dim(tensors, "model.scorer.mlp.2.weight")?;
        let output = dim(tensors, "model.scorer.mlp.4.weight")?;
        Ok(Self {
            linear0: linear(input, hidden0, vb.pp("mlp.0"))?,
            linear2: linear(hidden0, hidden1, vb.pp("mlp.2"))?,
            linear4: linear(hidden1, output, vb.pp("mlp.4"))?,
        })
    }

    fn forward(&self, text: &Tensor, labels: &Tensor) -> Result<Tensor, ModelError> {
        let dim = text.dim(1)?;
        let text = text.broadcast_as((labels.dim(0)?, dim))?;
        let combined = Tensor::cat(&[&text, labels], 1)?;
        let scores = self.linear0.forward(&combined)?.relu()?;
        let scores = self.linear2.forward(&scores)?.relu()?;
        Ok(self.linear4.forward(&scores)?.squeeze(1)?)
    }
}

struct Calibrator {
    text_norm: LayerNorm,
    label_norm: LayerNorm,
    linear0: Linear,
    linear3: Linear,
    linear6: Linear,
    beta_max: f32,
    use_bias: bool,
}

impl Calibrator {
    fn load(
        vb: VarBuilder,
        representation: usize,
        tensors: &HashMap<String, Tensor>,
        beta_max: f64,
        use_bias: bool,
    ) -> Result<Self, ModelError> {
        let hidden0 = dim(tensors, "calibrator.mlp.0.weight")?;
        let hidden1 = dim(tensors, "calibrator.mlp.3.weight")?;
        let output = dim(tensors, "calibrator.mlp.6.weight")?;
        let features = representation * 4 + 6;
        Ok(Self {
            text_norm: layer_norm(
                representation,
                CALIBRATOR_LAYER_NORM_EPS,
                vb.pp("text_norm"),
            )?,
            label_norm: layer_norm(
                representation,
                CALIBRATOR_LAYER_NORM_EPS,
                vb.pp("label_norm"),
            )?,
            linear0: linear(features, hidden0, vb.pp("mlp.0"))?,
            linear3: linear(hidden0, hidden1, vb.pp("mlp.3"))?,
            linear6: linear(hidden1, output, vb.pp("mlp.6"))?,
            beta_max: beta_max as f32,
            use_bias,
        })
    }

    fn forward(
        &self,
        text: &Tensor,
        labels: &Tensor,
        logits: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let count = labels.dim(0)?;
        let text = self
            .text_norm
            .forward(text)?
            .broadcast_as((count, labels.dim(1)?))?;
        let labels = self.label_norm.forward(labels)?;
        let product = (&text * &labels)?;
        let difference = text.sub(&labels)?.abs()?;
        let stats = self.stats(logits, count)?;
        let features = Tensor::cat(&[&text, &labels, &product, &difference, &stats], 1)?;
        let out = self.linear0.forward(&features)?.gelu_erf()?;
        let out = self.linear3.forward(&out)?.gelu_erf()?;
        let out = self.linear6.forward(&out)?;
        let beta = tensor_sigmoid(&out.i((.., 0))?)?.affine(self.beta_max as f64, 0.0)?;
        let bias = if self.use_bias {
            out.i((.., 1))?
        } else {
            Tensor::zeros_like(&beta)?
        };
        Ok((logits * &beta + &bias)?)
    }

    fn stats(&self, logits: &Tensor, count: usize) -> Result<Tensor, ModelError> {
        let values = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let entropy = values
            .iter()
            .map(|value| {
                let probability = sigmoid(*value);
                -(probability * log_sigmoid(*value) + (1.0 - probability) * log_sigmoid(-*value))
            })
            .sum::<f32>()
            / values.len() as f32;
        let log_labels = (values.len() as f32).ln();
        let mut stats = Vec::with_capacity(values.len() * 6);
        for value in values {
            stats.extend_from_slice(&[
                value,
                value.abs(),
                value - max,
                value - mean,
                entropy,
                log_labels,
            ]);
        }
        Ok(Tensor::from_vec(stats, (count, 6), logits.device())?)
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn softplus(value: f32) -> f32 {
    if value > 0.0 {
        value + (1.0 + (-value).exp()).ln()
    } else {
        (1.0 + value.exp()).ln()
    }
}

fn log_sigmoid(value: f32) -> f32 {
    -softplus(-value)
}

fn dim(tensors: &HashMap<String, Tensor>, name: &str) -> Result<usize, ModelError> {
    tensors
        .get(name)
        .map(|tensor| tensor.dims()[0])
        .ok_or_else(|| ModelError::config(format!("missing tensor {name}")))
}

enum GliclassEncoder {
    Deberta(Box<DebertaEncoder>),
    Modern(Box<ModernBertEncoder>),
}

impl GliclassEncoder {
    fn forward(&self, ids: &Tensor, mask: &Tensor) -> Result<Tensor, ModelError> {
        match self {
            Self::Deberta(encoder) => encoder.forward(ids, mask),
            Self::Modern(encoder) => encoder.forward(ids, mask),
        }
    }
}

struct GliclassFamily {
    tokenizer: Arc<dyn TokenEncoder>,
    encoder: GliclassEncoder,
    text_projector: Projector,
    classes_projector: Projector,
    scorer: Scorer,
    calibrator: Option<Calibrator>,
    layout: GliclassLayout,
    label_pooling: LabelPooling,
    calibration: Calibration,
    device: Device,
}

fn pool_labels(
    hidden: &Tensor,
    markers: &[usize],
    pooling: LabelPooling,
    length: usize,
) -> Result<Tensor, ModelError> {
    match pooling {
        LabelPooling::First => {
            let positions = Tensor::from_vec(
                markers
                    .iter()
                    .map(|marker| *marker as u32)
                    .collect::<Vec<u32>>(),
                markers.len(),
                hidden.device(),
            )?;
            Ok(hidden.index_select(&positions, 0)?)
        }
        LabelPooling::Average => {
            let width = hidden.dim(1)?;
            let mut pooled = Vec::with_capacity(markers.len());
            for (index, start) in markers.iter().enumerate() {
                // Upstream averages from marker k up to marker k+1; the last
                // label's span runs to the end of the *row*, not to the padded
                // width of a batched encoder call.
                let end = markers.get(index + 1).copied().unwrap_or(length);
                let start = (*start).min(end);
                if end == start {
                    pooled.push(Tensor::zeros((width,), hidden.dtype(), hidden.device())?);
                } else {
                    pooled.push(hidden.narrow(0, start, end - start)?.mean(0)?);
                }
            }
            if pooled.is_empty() {
                Ok(Tensor::zeros((0, width), hidden.dtype(), hidden.device())?)
            } else {
                Ok(Tensor::stack(&pooled, 0)?)
            }
        }
    }
}

const BATCH_TOKEN_BUDGET: usize = 32_768;

impl GliclassFamily {
    fn predict_rows(&self, rows: &[GliclassRow]) -> Result<Vec<(Vec<f32>, usize)>, ModelError> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let lengths = rows.iter().map(|row| row.ids.len()).collect::<Vec<_>>();
        let mut out = vec![(Vec::new(), 0usize); rows.len()];
        for range in groups(&lengths, BATCH_TOKEN_BUDGET) {
            let batch = &rows[range.clone()];
            let width = batch
                .iter()
                .map(|row| row.ids.len())
                .max()
                .unwrap_or(1)
                .max(1);
            let mut ids = vec![0u32; batch.len() * width];
            let mut mask = vec![0f32; batch.len() * width];
            for (index, row) in batch.iter().enumerate() {
                let start = index * width;
                ids[start..start + row.ids.len()].copy_from_slice(&row.ids);
                mask[start..start + row.ids.len()].fill(1.0);
            }
            let ids = Tensor::from_vec(ids, (batch.len(), width), &self.device)?;
            let mask = Tensor::from_vec(mask, (batch.len(), width), &self.device)?;
            let hidden = self.encoder.forward(&ids, &mask)?.to_dtype(DType::F32)?;
            for (offset, row) in batch.iter().enumerate() {
                let length = row.ids.len();
                let row_hidden = hidden.i(offset)?;
                let text = self
                    .text_projector
                    .forward(&row_hidden.i(0)?.unsqueeze(0)?)?;
                let label_hidden =
                    pool_labels(&row_hidden, &row.markers, self.label_pooling, length)?;
                let labels = self.classes_projector.forward(&label_hidden)?;
                let logits = self.scorer.forward(&text, &labels)?;
                let logits = match &self.calibrator {
                    Some(calibrator) => calibrator.forward(&text, &labels, &logits)?,
                    None => logits,
                };
                out[range.start + offset] =
                    (logits.to_dtype(DType::F32)?.to_vec1::<f32>()?, length);
            }
        }
        Ok(out)
    }
}

impl Family for GliclassFamily {
    fn id(&self) -> &'static str {
        "gliclass"
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
            return Err(
                DecisionError::unsupported("gliclass does not support ordinal choices").into(),
            );
        }
        let state_text = match state.as_content() {
            Content::String(text) => text.clone(),
            content => dumps_content(content),
        };
        let mut answers = IndexMap::with_capacity(questions.len());
        let mut input_tokens = 0u64;
        let mut rows = Vec::with_capacity(questions.len());
        for (qid, question) in questions.iter() {
            let row = self
                .layout
                .encode(&state_text, qid, question, self.tokenizer.as_ref())?;
            if row.truncated {
                return Err(DecisionError::StateTruncated.into());
            }
            input_tokens += row.ids.len() as u64;
            rows.push(row);
        }
        let row_logits = self.predict_rows(&rows)?;
        for ((qid, question), (row, (logits, length))) in
            questions.iter().zip(rows.iter().zip(row_logits))
        {
            let option_logits = match row.mode {
                Mode::Multi => logits,
                Mode::Single => vec![0.0, logits.first().copied().unwrap_or(0.0)],
            };
            let answer = match question {
                Question::Choice(choice) => {
                    let labels: Vec<String> = choice.criteria.keys().cloned().collect();
                    answer_from_logits(
                        question,
                        &labels,
                        &[],
                        &option_logits,
                        &self.calibration,
                        length,
                    )?
                }
                Question::Score(score) => answer_from_logits(
                    question,
                    &[],
                    &score.criteria,
                    &option_logits,
                    &self.calibration,
                    length,
                )?,
                Question::Noul(_) => answer_from_logits(
                    question,
                    &["false".to_owned(), "true".to_owned()],
                    &[],
                    &option_logits,
                    &self.calibration,
                    length,
                )?,
                Question::Box(_) => {
                    return Err(ModelError::unsupported(
                        "gliclass does not answer box questions",
                    ));
                }
            };
            answers.insert(qid.clone(), answer);
        }
        Ok(SystemOneResponse::new(
            self.id(),
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

fn resolve_specials(
    tokenizer: &dyn TokenEncoder,
    config: &GliclassConfig,
) -> Result<GliclassSpecials, ModelError> {
    let cls = tokenizer
        .token_id("[CLS]")
        .or_else(|| tokenizer.token_id("<s>"))
        .ok_or_else(|| ModelError::config("gliclass tokenizer has no [CLS] token"))?;
    let sep = tokenizer
        .token_id("[SEP]")
        .or_else(|| tokenizer.token_id("</s>"))
        .ok_or_else(|| ModelError::config("gliclass tokenizer has no [SEP] token"))?;
    if tokenizer.token_id(MARKER_TEXT) != Some(config.class_token_index) {
        return Err(ModelError::config(format!(
            "tokenizer {MARKER_TEXT} id does not match class_token_index {}",
            config.class_token_index
        )));
    }
    if tokenizer.token_id(TEXT_SEP_TEXT) != Some(config.text_token_index) {
        return Err(ModelError::config(format!(
            "tokenizer {TEXT_SEP_TEXT} id does not match text_token_index {}",
            config.text_token_index
        )));
    }
    Ok(GliclassSpecials {
        cls,
        sep,
        label: config.class_token_index,
        text_sep: config.text_token_index,
    })
}

const DEBERTA_WORD_EMBEDDINGS: &str = "model.encoder_model.embeddings.word_embeddings.weight";
const MODERNBERT_WORD_EMBEDDINGS: &str = "model.encoder_model.embeddings.tok_embeddings.weight";
const SEGMENT_EMBEDDINGS: &str = "model.segment_embeddings.weight";

/// Folds the learned segment embeddings into an expanded word-embedding table:
/// rows `[0, vocab)` are segment 0, rows `[vocab, 2*vocab)` are segment 1.
/// Segment 2 only covers `<<EXAMPLE>>` sections, which this layout never emits,
/// so it is dropped. The candle backbone then needs no segment input.
fn fold_segment_embeddings(
    tensors: &mut HashMap<String, Tensor>,
    word_key: &str,
    vocab_size: &mut usize,
) -> Result<usize, ModelError> {
    let word = tensors
        .remove(word_key)
        .ok_or_else(|| ModelError::config(format!("missing tensor {word_key}")))?;
    let segments = tensors
        .get(SEGMENT_EMBEDDINGS)
        .ok_or_else(|| ModelError::config(format!("missing tensor {SEGMENT_EMBEDDINGS}")))?
        .clone();
    let vocab = word.dim(0)?;
    let hidden = word.dim(1)?;
    if segments.dim(0)? < 2 || segments.dim(1)? != hidden {
        return Err(ModelError::config(format!(
            "{SEGMENT_EMBEDDINGS} has shape {:?}, expected at least [2, {hidden}]",
            segments.dims()
        )));
    }
    let segment_zero = segments.narrow(0, 0, 1)?;
    let segment_one = segments.narrow(0, 1, 1)?;
    let first = word.broadcast_add(&segment_zero)?;
    let second = word.broadcast_add(&segment_one)?;
    tensors.insert(word_key.to_owned(), Tensor::cat(&[first, second], 0)?);
    *vocab_size = vocab * 2;
    Ok(vocab)
}

/// Maps the ModernBERT encoder's `model.*` names to the checkpoint's
/// `model.encoder_model.*` prefix.
struct EncoderModelRenamer;

impl Renamer for EncoderModelRenamer {
    fn rename(&self, value: &str) -> Cow<'_, str> {
        match value.strip_prefix("model.") {
            Some(rest) => Cow::Owned(format!("model.encoder_model.{rest}")),
            None => Cow::Owned(value.to_owned()),
        }
    }
}

/// Loads a GLiClass uni-encoder checkpoint (v4-edge, instruct-large v1.0, or
/// the ModernBERT-based instruct-edge v1.0).
pub struct Factory;

impl FamilyFactory for Factory {
    fn family(&self) -> &'static str {
        "gliclass"
    }

    fn load(&self, ctx: &LoadContext<'_>) -> Result<Arc<dyn Family>, ModelError> {
        if !matches!(
            ctx.manifest.arch.as_str(),
            "deberta-v3" | "deberta-v2" | "modernbert"
        ) {
            return Err(ModelError::config(format!(
                "gliclass serves DeBERTa or ModernBERT, got arch {:?}",
                ctx.manifest.arch
            )));
        }
        if ctx.manifest.layout != "gliclass-uni-v1" {
            return Err(ModelError::config(format!(
                "gliclass serves layout gliclass-uni-v1, got {:?}",
                ctx.manifest.layout
            )));
        }
        let config_file = ctx
            .manifest
            .file(FileRole::Config)
            .ok_or_else(|| ModelError::config("gliclass manifest has no config file"))?;
        let config_path = ctx.dir.join(&config_file.path);
        let config: GliclassConfig = read_json(&config_path)?;
        config.validate()?;
        let specials = resolve_specials(ctx.tokenizer.as_ref(), &config)?;
        let label_pooling = LabelPooling::parse(&config.class_token_pooling)?;

        let weights = ctx
            .manifest
            .file(FileRole::Weights)
            .ok_or_else(|| ModelError::config("gliclass manifest has no weights file"))?;
        let weights_path = ctx.dir.join(&weights.path);
        let mut tensors: HashMap<String, Tensor> =
            candle_core::safetensors::load(&weights_path, ctx.device)?;
        for tensor in tensors.values_mut() {
            if tensor.dtype() != DType::F32 {
                *tensor = tensor.to_dtype(DType::F32)?;
            }
        }

        let encoder_config = config.encoder_kind()?;
        if encoder_config.name() != ctx.manifest.arch
            && !(matches!(&encoder_config, GliclassEncoderConfig::Deberta(_))
                && ctx.manifest.arch.starts_with("deberta"))
        {
            return Err(ModelError::config(format!(
                "GLiClass arch {:?} does not match its {} encoder config",
                ctx.manifest.arch,
                encoder_config.name()
            )));
        }
        let (encoder, encoder_hidden, segment_vocab) = match encoder_config {
            GliclassEncoderConfig::Modern(mut modern) => {
                let segment_vocab = if config.use_segment_embeddings {
                    Some(fold_segment_embeddings(
                        &mut tensors,
                        MODERNBERT_WORD_EMBEDDINGS,
                        &mut modern.vocab_size,
                    )?)
                } else {
                    None
                };
                let hidden = modern.hidden_size;
                let dtype = ctx.compute_dtype();
                let vb = VarBuilder::from_tensors(tensors.clone(), dtype, ctx.device);
                let vb = VarBuilder::from_backend(
                    Box::new(Rename::new(vb, EncoderModelRenamer)),
                    dtype,
                    ctx.device.clone(),
                );
                let encoder = ModernBertEncoder::load(vb, &modern)?;
                (
                    GliclassEncoder::Modern(Box::new(encoder)),
                    hidden,
                    segment_vocab,
                )
            }
            GliclassEncoderConfig::Deberta(mut deberta) => {
                let segment_vocab = if config.use_segment_embeddings {
                    Some(fold_segment_embeddings(
                        &mut tensors,
                        DEBERTA_WORD_EMBEDDINGS,
                        &mut deberta.vocab_size,
                    )?)
                } else {
                    None
                };
                let hidden = deberta.hidden_size;
                let encoder = DebertaEncoder::load_with_prefix(
                    &tensors,
                    &deberta,
                    ctx.device,
                    "model.encoder_model",
                )?;
                (
                    GliclassEncoder::Deberta(Box::new(encoder)),
                    hidden,
                    segment_vocab,
                )
            }
        };

        let project_hidden = config.hidden_size.max(1);
        // The projectors and scorer stay F32; only the encoder can run in half
        // precision.
        let vb = VarBuilder::from_tensors(tensors.clone(), DType::F32, ctx.device);
        let text_projector = Projector::load(
            vb.pp("model.text_projector"),
            encoder_hidden,
            project_hidden,
            encoder_hidden,
        )?;
        let classes_projector = Projector::load(
            vb.pp("model.classes_projector"),
            encoder_hidden,
            project_hidden,
            encoder_hidden,
        )?;
        let scorer = Scorer::load(vb.pp("model.scorer"), encoder_hidden * 2, &tensors)?;
        let calibrator = if config.use_calibrator {
            Some(Calibrator::load(
                vb.pp("calibrator"),
                encoder_hidden,
                &tensors,
                config.calibrator_beta_max,
                config.calibrator_use_bias,
            )?)
        } else {
            None
        };

        Ok(Arc::new(GliclassFamily {
            tokenizer: ctx.tokenizer.clone(),
            encoder,
            text_projector,
            classes_projector,
            scorer,
            calibrator,
            layout: GliclassLayout {
                specials,
                segment_vocab,
            },
            label_pooling,
            calibration: read_calibration(ctx)?,
            device: ctx.device.clone(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn question(value: Value) -> Question {
        serde_json::from_value(value).expect("question")
    }

    struct MarkerTokenizer;

    impl TokenEncoder for MarkerTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
            let mut ids = Vec::new();
            let mut rest = text;
            while !rest.is_empty() {
                if let Some(tail) = rest.strip_prefix(MARKER_TEXT) {
                    ids.push(100);
                    rest = tail;
                } else if let Some(tail) = rest.strip_prefix(TEXT_SEP_TEXT) {
                    ids.push(101);
                    rest = tail;
                } else {
                    ids.push(1);
                    rest = &rest[1..];
                }
            }
            Ok(ids)
        }
    }

    #[test]
    fn segment_ids_are_folded_into_the_vocabulary() {
        let layout = GliclassLayout {
            specials: GliclassSpecials {
                cls: 1,
                sep: 2,
                label: 100,
                text_sep: 101,
            },
            segment_vocab: Some(1000),
        };
        let choice = question(json!({
            "type": "choice",
            "instructions": "b",
            "criteria": {"a": null}
        }));
        let row = layout
            .encode("", "q", &choice, &MarkerTokenizer)
            .expect("row");
        assert_eq!(row.ids, [1, 100, 1, 1101, 1001, 1002]);
        assert_eq!(row.markers, [1]);
    }

    #[test]
    fn builds_labels_and_prompt() {
        let layout = GliclassLayout {
            specials: GliclassSpecials {
                cls: 1,
                sep: 2,
                label: 100,
                text_sep: 101,
            },
            segment_vocab: None,
        };
        let choice = question(json!({
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "payments", "other": ""}
        }));
        let (labels, prompt, mode) = layout.labels_and_prompt("q", &choice).expect("labels");
        assert_eq!(labels, ["billing: payments", "other"]);
        assert_eq!(prompt, "Which team?");
        assert_eq!(mode, Mode::Multi);

        let pair = question(json!({
            "type": "noul",
            "instructions": "Angry?",
            "criteria": {"TRUE": "The customer is angry.", "false": "The customer is calm."}
        }));
        assert_eq!(
            layout.labels_and_prompt("q", &pair).expect("pair"),
            (
                vec![
                    "The customer is calm.".to_owned(),
                    "The customer is angry.".to_owned()
                ],
                "Angry?".to_owned(),
                Mode::Multi
            )
        );

        let single = question(json!({
            "type": "noul",
            "instructions": "It is spam.",
            "criteria": {"true": "yes", "false": ""}
        }));
        assert_eq!(
            layout.labels_and_prompt("q", &single).expect("single"),
            (vec!["It is spam.".to_owned()], String::new(), Mode::Single)
        );
    }

    #[test]
    fn flags_a_truncated_state() {
        let layout = GliclassLayout {
            specials: GliclassSpecials {
                cls: 1,
                sep: 2,
                label: 100,
                text_sep: 101,
            },
            segment_vocab: None,
        };
        let choice = question(json!({
            "type": "choice",
            "instructions": "b",
            "criteria": {"a": null}
        }));
        let row = layout
            .encode(&"x".repeat(2000), "q", &choice, &MarkerTokenizer)
            .expect("row");
        assert!(row.truncated);
        assert_eq!(row.ids.len(), MAX_LEN);
        assert_eq!(row.markers, [1]);

        let row = layout
            .encode("short", "q", &choice, &MarkerTokenizer)
            .expect("row");
        assert!(!row.truncated);
    }

    #[test]
    fn sanitizes_markers() {
        assert_eq!(sanitize("a<<LABEL>>b<<SEP>>c<<EXAMPLE>>d"), "a b c d");
    }

    #[test]
    fn first_pooling_selects_the_marker_rows() {
        let hidden = Tensor::from_vec(
            (0..12).map(|value| value as f32).collect::<Vec<f32>>(),
            (6, 2),
            &Device::Cpu,
        )
        .expect("hidden");
        let pooled = pool_labels(&hidden, &[1, 4], LabelPooling::First, 6)
            .expect("first pooling")
            .to_vec2::<f32>()
            .expect("values");
        assert_eq!(pooled, vec![vec![2.0, 3.0], vec![8.0, 9.0]]);
    }

    #[test]
    fn average_pooling_uses_each_label_span() {
        let hidden = Tensor::from_vec(
            vec![
                0.0f32, 0.0, //
                2.0, 2.0, //
                4.0, 4.0, //
                10.0, 10.0, //
                20.0, 20.0, //
                30.0, 30.0,
            ],
            (6, 2),
            &Device::Cpu,
        )
        .expect("hidden");
        let pooled = pool_labels(&hidden, &[0, 3], LabelPooling::Average, 6)
            .expect("average pooling")
            .to_vec2::<f32>()
            .expect("values");
        assert_eq!(pooled, vec![vec![2.0, 2.0], vec![20.0, 20.0]]);
    }

    #[test]
    fn average_pooling_runs_the_last_label_to_the_row_end() {
        let hidden =
            Tensor::from_vec(vec![2.0f32, 4.0, 6.0], (3, 1), &Device::Cpu).expect("hidden");
        let pooled = pool_labels(&hidden, &[1], LabelPooling::Average, 3)
            .expect("average pooling")
            .to_vec2::<f32>()
            .expect("values");
        assert_eq!(pooled, vec![vec![5.0]]);
    }

    #[test]
    fn average_pooling_ignores_batch_padding() {
        let hidden =
            Tensor::from_vec(vec![2.0f32, 4.0, 100.0], (3, 1), &Device::Cpu).expect("hidden");
        let pooled = pool_labels(&hidden, &[1], LabelPooling::Average, 2)
            .expect("average pooling")
            .to_vec2::<f32>()
            .expect("values");
        assert_eq!(pooled, vec![vec![4.0]]);
    }

    #[test]
    fn folds_segment_embeddings_into_word_embeddings() {
        let mut tensors = HashMap::from([
            (
                DEBERTA_WORD_EMBEDDINGS.to_owned(),
                Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu).expect("word"),
            ),
            (
                SEGMENT_EMBEDDINGS.to_owned(),
                Tensor::from_vec(
                    vec![10.0f32, 20.0, 100.0, 200.0, 0.0, 0.0],
                    (3, 2),
                    &Device::Cpu,
                )
                .expect("segments"),
            ),
        ]);
        let mut vocab_size = 2;
        let vocab = fold_segment_embeddings(&mut tensors, DEBERTA_WORD_EMBEDDINGS, &mut vocab_size)
            .expect("folds");
        assert_eq!(vocab, 2);
        assert_eq!(vocab_size, 4);
        let expanded = tensors[DEBERTA_WORD_EMBEDDINGS]
            .to_vec2::<f32>()
            .expect("expanded");
        assert_eq!(
            expanded,
            vec![
                vec![11.0, 22.0],
                vec![13.0, 24.0],
                vec![101.0, 202.0],
                vec![103.0, 204.0],
            ]
        );
    }
}
