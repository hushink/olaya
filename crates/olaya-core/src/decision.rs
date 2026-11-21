//! Engine-agnostic decision layer: tokenization boundary, sequence layout,
//! temperature calibration, and answer construction.

use std::collections::HashMap;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::answer::{Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer};
use crate::content::Content;
use crate::question::{Question, QuestionType};
use crate::state::State;

/// A fitted temperature below this sharpens logits instead of softening them.
pub const TEMP_MIN: f64 = 0.5;
/// A fitted temperature above this over-softens an already flat distribution.
pub const TEMP_MAX: f64 = 5.0;

/// The primitive a question asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QType {
    Choice,
    Score,
    Noul,
    Box,
}

impl QType {
    pub fn name(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
            Self::Box => "box",
        }
    }

    /// Calibration slot: `choice = 0`, `score = 1`, `noul = 2`, `box = 3`.
    pub fn index(self) -> usize {
        match self {
            Self::Choice => 0,
            Self::Score => 1,
            Self::Noul => 2,
            Self::Box => 3,
        }
    }

    /// Whether the primitive is scored over a finite option set.
    pub fn is_option_based(self) -> bool {
        !matches!(self, Self::Box)
    }

    pub fn of(question: &Question) -> Self {
        question.question_type().into()
    }
}

impl From<QuestionType> for QType {
    fn from(value: QuestionType) -> Self {
        match value {
            QuestionType::Choice => Self::Choice,
            QuestionType::Score => Self::Score,
            QuestionType::Noul => Self::Noul,
            QuestionType::Box => Self::Box,
        }
    }
}

/// Errors raised by a renderer or readout.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum DecisionError {
    #[error("invalid decision input: {message}")]
    Invalid { message: String },
    #[error("too many options for `{question}`: {options} > {limit}")]
    TooManyOptions {
        question: String,
        options: usize,
        limit: usize,
    },
    #[error("unsupported: {message}")]
    Unsupported { message: String },
    #[error("tokenizer error: {message}")]
    Tokenizer { message: String },
    #[error("inference error: {message}")]
    Inference { message: String },
    #[error("part of the state was dropped to fit the model context")]
    StateTruncated,
    #[error("the rendered input needs {tokens} tokens, over the model limit of {limit}")]
    InputTooLong { tokens: usize, limit: usize },
}

impl DecisionError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid {
            message: message.into(),
        }
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported {
            message: message.into(),
        }
    }

    pub fn tokenizer(message: impl Into<String>) -> Self {
        Self::Tokenizer {
            message: message.into(),
        }
    }
}

/// Text-to-token boundary. Families own their prompt template; the encoder
/// only turns text into token ids (`add_special_tokens = false`).
pub trait TokenEncoder: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<u32>, DecisionError>;

    /// The id of a token, when the tokenizer exposes one. Families that place
    /// explicit special tokens (`[CLS]`, `[SEP]`, `[MASK]`) use this.
    fn token_id(&self, token: &str) -> Option<u32> {
        let _ = token;
        None
    }
}

/// A rendered question, ready for a backbone forward pass.
#[derive(Clone, Debug, PartialEq)]
pub struct Encoded {
    /// The full row: `[CLS] ... [MASK] option ... [SEP] state [SEP]`.
    pub ids: Vec<u32>,
    /// Absolute positions of the option markers in `ids`.
    pub markers: Vec<usize>,
    pub qtype: QType,
    /// Number of state tokens (before per-question repetition).
    pub state_tokens: usize,
    /// Whether the state was truncated to fit the model's context.
    pub truncated: bool,
}

/// Builds the per-question sequence for a family's layout.
pub trait Renderer: Send + Sync {
    /// Tokenize the state once; the engine reuses the ids for every question.
    fn encode_state(
        &self,
        state: &State,
        encoder: &dyn TokenEncoder,
    ) -> Result<Vec<u32>, DecisionError>;

    /// Render one question against the already-tokenized state. `question_id`
    /// is the caller-chosen key; TypeSafe reads it in place of missing
    /// instructions.
    fn render(
        &self,
        state_ids: &[u32],
        question_id: &str,
        question: &Question,
        encoder: &dyn TokenEncoder,
    ) -> Result<Encoded, DecisionError>;
}

/// Turns option logits into a typed answer.
pub trait Readout: Send + Sync {
    fn read(
        &self,
        question: &Question,
        logits: &[f32],
        calibration: &Calibration,
        state_tokens: usize,
    ) -> Result<Answer, DecisionError>;
}

/// How a model's fitted temperatures are applied.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CalibrationMode {
    /// Clamp fitted temperatures to `[0.5, 5.0]` or the file's range.
    #[default]
    Clamp,
    /// Use the checkpoint's values as-is.
    Raw,
    /// Prefer the manifest's `onnx_temperature*` overrides, then clamp.
    OnnxBundle,
}

/// The `calibration` file shipped with a model.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CalibrationFile {
    /// One temperature per question type: choice, score, noul.
    #[serde(default)]
    pub temperature: Vec<Value>,
    /// Refinements keyed by `"<type>:<2|3-5|6-10|11+>"`.
    #[serde(default)]
    pub temperature_by_options: HashMap<String, Value>,
    /// An input-conditioned temperature.
    #[serde(default)]
    pub temperature_map: Option<Value>,
    /// `[min, max]` override for the clamp range.
    #[serde(default)]
    pub temperature_range: Option<[f64; 2]>,
    /// The receptron ONNX bundle's temperatures, when they differ.
    #[serde(default)]
    pub onnx_temperature: Option<Vec<Value>>,
    /// The receptron ONNX bundle's per-bucket temperatures.
    #[serde(default)]
    pub onnx_temperature_by_options: Option<HashMap<String, Value>>,
}

impl CalibrationFile {
    /// The clamp range: the file's own valid range, else `[TEMP_MIN, TEMP_MAX]`.
    pub fn range(&self) -> [f64; 2] {
        match self.temperature_range {
            Some([lo, hi]) if lo.is_finite() && hi.is_finite() && 0.0 < lo && lo <= hi => [lo, hi],
            _ => [TEMP_MIN, TEMP_MAX],
        }
    }
}

/// Temperatures as applied: clamped or raw, with neutral fallbacks.
#[derive(Clone, Debug)]
pub struct Calibration {
    by_type: [f64; 3],
    by_options: HashMap<String, f64>,
    map: Option<TemperatureMap>,
    mode: CalibrationMode,
}

impl Default for Calibration {
    fn default() -> Self {
        Self::from_file(&CalibrationFile::default(), CalibrationMode::Clamp)
    }
}

/// A temperature computed per question from its logits and the request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TemperatureMap {
    /// `von-entropy-length-v1`:
    ///
    /// ```text
    /// T = clamp(bias + entropy·H + log_tokens·log10(max(state_tokens,1))/4
    ///           + n_options·K/8, lo, hi)
    /// ```
    EntropyLength {
        bias: f64,
        entropy: f64,
        log_tokens: f64,
        n_options: f64,
        lo: f64,
        hi: f64,
    },
}

impl TemperatureMap {
    pub fn parse(value: &Value) -> Option<Self> {
        let map = value.as_object()?;
        match map.get("kind")?.as_str()? {
            "von-entropy-length-v1" => {
                let get = |key: &str| match map.get(key) {
                    None => Some(None),
                    Some(v) => v.as_f64().filter(|x| x.is_finite()).map(Some),
                };
                let (bias, entropy) = (get("bias")?, get("entropy")?);
                let (log_tokens, n_options) = (get("log_tokens")?, get("n_options")?);
                if [bias, entropy, log_tokens, n_options]
                    .iter()
                    .all(Option::is_none)
                {
                    return None;
                }
                let (lo, hi) = (get("lo")?.unwrap_or(0.5), get("hi")?.unwrap_or(12.0));
                (lo <= hi).then_some(Self::EntropyLength {
                    bias: bias.unwrap_or(0.0),
                    entropy: entropy.unwrap_or(0.0),
                    log_tokens: log_tokens.unwrap_or(0.0),
                    n_options: n_options.unwrap_or(0.0),
                    lo,
                    hi,
                })
            }
            _ => None,
        }
    }

    pub fn temperature(&self, logits: &[f32], state_tokens: usize) -> f64 {
        match *self {
            Self::EntropyLength {
                bias,
                entropy,
                log_tokens,
                n_options,
                lo,
                hi,
            } => {
                let k = logits.len();
                let h = if k > 1 {
                    let p = softmax(logits.iter().map(|&z| f64::from(z)));
                    -p.iter().map(|&p| p * p.max(1e-12).ln()).sum::<f64>() / (k as f64).ln()
                } else {
                    0.0
                };
                let tokens = state_tokens.max(1) as f64;
                let t = bias
                    + entropy * h
                    + log_tokens * (tokens.log10() / 4.0)
                    + n_options * (k as f64 / 8.0);
                t.max(lo).min(hi)
            }
        }
    }
}

/// `laya.common.temp_bucket`: `"<type>:<2|3-5|6-10|11+>"`.
pub fn bucket(qtype: QType, k: usize) -> String {
    let size = match k {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    };
    format!("{}:{size}", qtype.name())
}

pub fn clamp_temperature(t: &Value) -> f64 {
    clamp_temperature_in(t, [TEMP_MIN, TEMP_MAX])
}

pub fn clamp_temperature_in(t: &Value, [lo, hi]: [f64; 2]) -> f64 {
    match t.as_f64() {
        Some(t) if t.is_finite() => t.clamp(lo, hi),
        _ => 1.0,
    }
}

pub fn temperature_range(file: &CalibrationFile) -> [f64; 2] {
    file.range()
}

impl Calibration {
    pub fn from_file(file: &CalibrationFile, mode: CalibrationMode) -> Self {
        let range = file.range();
        let temperatures: &[Value] = match mode {
            CalibrationMode::OnnxBundle => file
                .onnx_temperature
                .as_deref()
                .unwrap_or(file.temperature.as_slice()),
            _ => &file.temperature,
        };
        let by_options_source = match mode {
            CalibrationMode::OnnxBundle => file
                .onnx_temperature_by_options
                .as_ref()
                .unwrap_or(&file.temperature_by_options),
            _ => &file.temperature_by_options,
        };

        let convert = |value: &Value| match value.as_f64() {
            Some(value) if value.is_finite() => match mode {
                CalibrationMode::Raw => value,
                CalibrationMode::Clamp | CalibrationMode::OnnxBundle => {
                    value.clamp(range[0], range[1])
                }
            },
            _ => 1.0,
        };

        let mut by_type = [1.0; 3];
        for (slot, value) in by_type.iter_mut().zip(temperatures) {
            *slot = convert(value);
        }
        let by_options = by_options_source
            .iter()
            .map(|(key, value)| (key.clone(), convert(value)))
            .collect();

        Self {
            by_type,
            by_options,
            map: file
                .temperature_map
                .as_ref()
                .and_then(TemperatureMap::parse),
            mode,
        }
    }

    pub fn mode(&self) -> CalibrationMode {
        self.mode
    }

    pub fn map(&self) -> Option<&TemperatureMap> {
        self.map.as_ref()
    }

    /// The fixed temperature for a question type and option count.
    pub fn temperature(&self, qtype: QType, k: usize) -> f64 {
        if qtype == QType::Box {
            return 1.0;
        }
        self.by_options
            .get(&bucket(qtype, k))
            .copied()
            .unwrap_or(self.by_type[qtype.index()])
    }

    /// The temperature applied to one question's logits.
    pub fn temperature_for(&self, qtype: QType, logits: &[f32], state_tokens: usize) -> f64 {
        match &self.map {
            Some(map) => map.temperature(logits, state_tokens).max(1e-4),
            None => self.temperature(qtype, logits.len()),
        }
    }

    /// Calibrated option probabilities from raw logits.
    pub fn probabilities(&self, qtype: QType, logits: &[f32], state_tokens: usize) -> Vec<f64> {
        let t = self.temperature_for(qtype, logits, state_tokens);
        softmax(logits.iter().map(|&z| f64::from(z) / t))
    }
}

pub fn softmax(z: impl Iterator<Item = f64> + Clone) -> Vec<f64> {
    let max = z.clone().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.map(|v| (v - max).exp()).collect();
    let sum: f64 = e.iter().sum();
    e.into_iter().map(|v| v / sum).collect()
}

/// TypeSafe's normalized top probability `(K * max(p) - 1) / (K - 1)`.
pub fn typesafe_confidence(probabilities: &IndexMap<String, f64>) -> f64 {
    let count = probabilities.len();
    if count < 2 {
        return 1.0;
    }
    let max = probabilities
        .values()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    ((count as f64 * max - 1.0) / (count as f64 - 1.0)).clamp(0.0, 1.0)
}

/// Build a choice answer from option labels and calibrated probabilities.
pub fn choice_answer(labels: &[String], probabilities: &[f64]) -> Answer {
    let best = argmax(probabilities);
    let probabilities = labels
        .iter()
        .zip(probabilities)
        .map(|(label, probability)| (label.clone(), *probability))
        .collect::<IndexMap<_, _>>();
    Answer::Choice(ChoiceAnswer {
        choice: labels[best].clone(),
        confidence: typesafe_confidence(&probabilities),
        probabilities,
    })
}

/// Build a score answer from ordered levels and calibrated probabilities.
pub fn score_answer(legend: &[Content], probabilities: &[f64]) -> Answer {
    let probabilities = (0..legend.len().min(probabilities.len()))
        .map(|index| (index.to_string(), probabilities[index]))
        .collect::<IndexMap<_, _>>();
    let score = probabilities
        .values()
        .enumerate()
        .map(|(index, probability)| index as f64 * probability)
        .sum();
    let legend = legend
        .iter()
        .enumerate()
        .map(|(index, content)| (index.to_string(), content.clone()))
        .collect::<IndexMap<_, _>>();
    Answer::Score(ScoreAnswer {
        score,
        confidence: typesafe_confidence(&probabilities),
        legend,
        probabilities,
    })
}

/// Build a noul answer from `[false, true]` calibrated probabilities.
pub fn noul_answer(probabilities: &[f64]) -> Answer {
    Answer::Noul(NoulAnswer {
        noul: probabilities.get(1).copied().unwrap_or(0.5).clamp(0.0, 1.0),
    })
}

/// The first maximum, as Python's `argmax`.
pub fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

/// Construct a typed answer from raw option logits.
pub fn answer_from_logits(
    question: &Question,
    labels: &[String],
    legend: &[Content],
    logits: &[f32],
    calibration: &Calibration,
    state_tokens: usize,
) -> Result<Answer, DecisionError> {
    let qtype = QType::of(question);
    if !qtype.is_option_based() {
        return Err(DecisionError::unsupported(
            "box questions require a vision readout",
        ));
    }
    if logits.is_empty() {
        return Err(DecisionError::invalid("empty option logits"));
    }
    let probabilities = calibration.probabilities(qtype, logits, state_tokens);
    Ok(match qtype {
        QType::Choice => choice_answer(labels, &probabilities),
        QType::Score => score_answer(legend, &probabilities),
        QType::Noul => noul_answer(&probabilities),
        QType::Box => unreachable!("box is rejected above"),
    })
}

/// What a model family can answer.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    #[serde(default)]
    pub noul: bool,
    #[serde(default)]
    pub choice: bool,
    #[serde(default)]
    pub score: bool,
    #[serde(default, rename = "box")]
    pub boxed: bool,
    #[serde(default)]
    pub vision: bool,
    /// Whether `choice` questions may opt into the ordinal head (Decima).
    #[serde(default)]
    pub ordinal: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_options: Option<usize>,
}

impl Capabilities {
    pub fn classification() -> Self {
        Self {
            noul: true,
            choice: true,
            score: true,
            ..Self::default()
        }
    }

    pub fn all() -> Self {
        Self {
            noul: true,
            choice: true,
            score: true,
            boxed: true,
            vision: true,
            ordinal: true,
            max_options: None,
        }
    }

    /// Mark that ordinal choices are supported.
    pub fn with_ordinal(mut self) -> Self {
        self.ordinal = true;
        self
    }

    pub fn supports(&self, qtype: QType) -> bool {
        match qtype {
            QType::Noul => self.noul,
            QType::Choice => self.choice,
            QType::Score => self.score,
            QType::Box => self.boxed,
        }
    }
}
