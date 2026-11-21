//! Von 1.1 (`von-option-marker-v1`): a ModernBERT-large encoder that scores
//! every option at its own `[MASK]` in one sequence.
//!
//! ```text
//! [CLS] <instructions> <state> [SEP] [MASK] <description 0> [MASK] ... [SEP]
//! ```
//!
//! The state is Python `str()` text, one `key: value` line per top-level key.
//! An option is read from its description alone (the label only when it has
//! none). A noul question without criteria adds a second, empty-state row whose
//! logits estimate the question's own yes/no bias and are subtracted from the
//! first row's.
//!
//! The checkpoint is a PyTorch pickle (`option_marker.pt`) read in place by
//! [`crate::torch`].

use std::cell::OnceCell;
use std::path::Path;
use std::sync::Arc;

use crate::encoders::modernbert::{EncoderConfig, EncoderRenamer, ModernBertEncoder};
use crate::families::batch::groups;
use crate::tokenizer::HfTokenizer;
use crate::{Family, FamilyFactory, LoadContext, ModelError};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::var_builder::Rename;
use candle_nn::{LayerNorm, Linear, Module, VarBuilder, layer_norm, linear};
use indexmap::IndexMap;
use olaya_core::{
    Calibration, CalibrationFile, Content, DecisionError, FileRole, Question, QuestionType, State,
    SystemOneResponse, TokenEncoder, Usage, answer_from_logits, pyrepr,
};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

const MAX_LEN: usize = 8192;
const CLS_TEXT: &str = "[CLS]";
const SEP_TEXT: &str = "[SEP]";
const MASK_TEXT: &str = "[MASK]";
const PAD_TEXT: &str = "[PAD]";
const NOUL_DEFAULT_TRUE: &str = "Yes, condition holds true.";
const NOUL_DEFAULT_FALSE: &str = "No, condition is false.";
const NOUL_PRIOR_A: f64 = 0.7;
const NOUL_PRIOR_B: f64 = 0.0;
const SCORER_LAYER_NORM_EPS: f64 = 1e-5;

/// A tokenizer that can also say where each token ends, for cutting the state
/// at a token boundary.
trait CharOffsetEncoder: TokenEncoder {
    fn char_ends(&self, text: &str) -> Result<Vec<usize>, ModelError>;
}

impl CharOffsetEncoder for HfTokenizer {
    fn char_ends(&self, text: &str) -> Result<Vec<usize>, ModelError> {
        HfTokenizer::char_ends(self, text)
    }
}

#[derive(Clone, Copy, Debug)]
struct VonSpecials {
    cls: u32,
    sep: u32,
    mask: u32,
    #[allow(dead_code)]
    pad: u32,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct ZeroShotPrior {
    #[serde(default = "default_prior_a")]
    a: f64,
    #[serde(default = "default_prior_b")]
    b: f64,
}

impl Default for ZeroShotPrior {
    fn default() -> Self {
        Self {
            a: NOUL_PRIOR_A,
            b: NOUL_PRIOR_B,
        }
    }
}

fn default_prior_a() -> f64 {
    NOUL_PRIOR_A
}

fn default_prior_b() -> f64 {
    NOUL_PRIOR_B
}

fn default_temperature() -> f64 {
    1.0
}

#[derive(Clone, Debug, Deserialize)]
struct MarkerCalibration {
    #[serde(default = "default_temperature")]
    temperature: f64,
    #[serde(default)]
    calibration_map: Option<Map<String, Value>>,
    #[serde(default)]
    noul_zero_shot_prior: Option<ZeroShotPrior>,
}

struct StateText {
    /// Tokens in the rendered state before sanitising: the temperature map's
    /// `state_tokens`.
    tokens: usize,
    /// The rendered state with mask tokens replaced, as rows read it.
    text: String,
    /// Where each token of `text` ends (in characters), once a row has had to
    /// cut the state.
    ends: OnceCell<Vec<usize>>,
}

#[derive(Clone)]
struct Row {
    ids: Vec<u32>,
    markers: Vec<usize>,
}

struct Scoring {
    qtype: QuestionType,
    rows: Vec<Row>,
    zero_shot: bool,
    truncated: bool,
}

struct VonLayout {
    specials: VonSpecials,
    max_len: usize,
    prior: ZeroShotPrior,
}

impl VonLayout {
    fn sanitize(&self, text: &str) -> String {
        text.replace(MASK_TEXT, " ")
    }

    fn state_text(&self, content: &Content) -> String {
        match content {
            Content::String(text) => text.clone(),
            Content::Object(map) => map
                .iter()
                .map(|(key, value)| format!("{key}: {}", pyrepr::str(value)))
                .collect::<Vec<_>>()
                .join("\n"),
            Content::Array(items) => pyrepr::repr(&Value::Array(items.clone())),
            Content::Null => pyrepr::repr(&Value::Null),
        }
    }

    fn encode_state(
        &self,
        enc: &dyn CharOffsetEncoder,
        state: &Content,
    ) -> Result<StateText, ModelError> {
        let raw = self.state_text(state);
        Ok(StateText {
            tokens: enc.encode(&raw)?.len(),
            text: self.sanitize(&raw),
            ends: OnceCell::new(),
        })
    }

    fn instructions(&self, qid: &str, question: &Question) -> String {
        question.instructions_or_id(qid)
    }

    fn truthy(&self, content: &Content) -> bool {
        match content {
            Content::Null => false,
            Content::String(text) => !text.is_empty(),
            Content::Object(map) => !map.is_empty(),
            Content::Array(items) => !items.is_empty(),
        }
    }

    fn join_examples(&self, examples: &Value) -> String {
        match examples {
            Value::String(text) => text
                .chars()
                .map(String::from)
                .collect::<Vec<_>>()
                .join(", "),
            Value::Array(items) => items.iter().map(pyrepr::str).collect::<Vec<_>>().join(", "),
            Value::Object(map) => map.keys().cloned().collect::<Vec<_>>().join(", "),
            other => pyrepr::str(other),
        }
    }

    fn level_description(&self, level: &Content) -> String {
        match level {
            Content::Object(map) => {
                let what = map.get("what").map(pyrepr::str).unwrap_or_default();
                let examples = match map.get("examples") {
                    Some(value) if pyrepr::truthy(value) => {
                        format!(" Examples: {}", self.join_examples(value))
                    }
                    _ => String::new(),
                };
                pyrepr::strip(&format!("{what}{examples}")).to_owned()
            }
            other => pyrepr::strip(&pyrepr::str_content(other)).to_owned(),
        }
    }

    fn descriptions(&self, question: &Question) -> Result<(Vec<String>, bool), ModelError> {
        match question {
            Question::Choice(choice) => Ok((
                choice
                    .criteria
                    .iter()
                    .map(|(label, value)| {
                        if self.truthy(value) {
                            pyrepr::strip(&pyrepr::str_content(value)).to_owned()
                        } else {
                            pyrepr::strip(label).to_owned()
                        }
                    })
                    .collect(),
                false,
            )),
            Question::Score(score) => Ok((
                score
                    .criteria
                    .iter()
                    .map(|level| self.level_description(level))
                    .collect(),
                false,
            )),
            Question::Noul(noul) => {
                let (is_false, is_true) = match &noul.criteria {
                    Some(criteria) => (
                        criteria
                            .is_false
                            .as_ref()
                            .filter(|value| self.truthy(value)),
                        criteria.is_true.as_ref().filter(|value| self.truthy(value)),
                    ),
                    None => (None, None),
                };
                let zero_shot = is_false.is_none() && is_true.is_none();
                let descriptions = vec![
                    is_true.map_or_else(|| NOUL_DEFAULT_TRUE.to_owned(), pyrepr::str_content),
                    is_false.map_or_else(|| NOUL_DEFAULT_FALSE.to_owned(), pyrepr::str_content),
                ];
                Ok((descriptions, zero_shot))
            }
            Question::Box(_) => Err(ModelError::unsupported("von does not answer box questions")),
        }
    }

    fn pack(&self, state: &str, instructions: &str, descriptions: &[String]) -> String {
        let prefix = if instructions.is_empty() {
            pyrepr::strip(state).to_owned()
        } else {
            pyrepr::strip(&format!("{instructions} {state}")).to_owned()
        };
        let mut text = format!("{prefix} {SEP_TEXT} ");
        for (index, description) in descriptions.iter().enumerate() {
            if index > 0 {
                text.push(' ');
            }
            text.push_str(MASK_TEXT);
            text.push(' ');
            text.push_str(pyrepr::strip(description));
        }
        text
    }

    fn tokenize(&self, enc: &dyn CharOffsetEncoder, text: &str) -> Result<Vec<u32>, ModelError> {
        let body = enc.encode(text)?;
        let mut ids = Vec::with_capacity(body.len() + 2);
        ids.push(self.specials.cls);
        ids.extend(body);
        ids.push(self.specials.sep);
        Ok(ids)
    }

    fn fit_row(
        &self,
        enc: &dyn CharOffsetEncoder,
        state: &str,
        ends: &OnceCell<Vec<usize>>,
        instructions: &str,
        descriptions: &[String],
        qid: &str,
    ) -> Result<(Row, bool), ModelError> {
        let mut text = self.pack(state, instructions, descriptions);
        let mut ids = self.tokenize(enc, &text)?;
        let state_chars = state.chars().count();
        let mut kept = state_chars;
        if ids.len() > self.max_len {
            let ends = match ends.get() {
                Some(ends) => ends,
                None => {
                    let computed = enc.char_ends(state)?;
                    ends.get_or_init(|| computed)
                }
            };
            let mut n = ends.len() as isize;
            while ids.len() > self.max_len {
                if n <= 0 {
                    return Err(DecisionError::invalid(format!(
                        "question {qid:?}: its instructions and options alone take {} tokens; \
                         the model reads at most {}",
                        ids.len(),
                        self.max_len
                    ))
                    .into());
                }
                n -= (ids.len() - self.max_len) as isize;
                kept = if n > 0 { ends[n as usize - 1] } else { 0 };
                text = self.pack(char_prefix(state, kept), instructions, descriptions);
                ids = self.tokenize(enc, &text)?;
            }
        }
        let markers = ids
            .iter()
            .enumerate()
            .filter_map(|(position, id)| (*id == self.specials.mask).then_some(position))
            .collect::<Vec<_>>();
        if markers.len() != descriptions.len() {
            return Err(DecisionError::invalid(format!(
                "question {qid:?}: the text tokenizes to {} option markers for {} options",
                markers.len(),
                descriptions.len()
            ))
            .into());
        }
        Ok((Row { ids, markers }, kept < state_chars))
    }

    fn encode_question(
        &self,
        enc: &dyn CharOffsetEncoder,
        state: &StateText,
        qid: &str,
        question: &Question,
    ) -> Result<Scoring, ModelError> {
        let (descriptions, zero_shot) = self.descriptions(question)?;
        let instructions = self.sanitize(&self.instructions(qid, question));
        let descriptions = descriptions
            .iter()
            .map(|description| self.sanitize(description))
            .collect::<Vec<_>>();
        let (row, truncated) = self.fit_row(
            enc,
            &state.text,
            &state.ends,
            &instructions,
            &descriptions,
            qid,
        )?;
        let mut rows = vec![row];
        if zero_shot {
            let empty = OnceCell::new();
            let (row, _) = self.fit_row(enc, "", &empty, &instructions, &descriptions, qid)?;
            rows.push(row);
        }
        Ok(Scoring {
            qtype: question.question_type(),
            rows,
            zero_shot,
            truncated,
        })
    }

    fn option_logits(&self, scoring: &Scoring, rows: &[Vec<f32>]) -> Vec<f32> {
        let Some(first) = rows.first() else {
            return Vec::new();
        };
        if scoring.qtype != QuestionType::Noul {
            return first.clone();
        }
        let (l_true, l_false) = (f64::from(first[0]), f64::from(first[1]));
        let correction = match (scoring.zero_shot, rows.get(1)) {
            (true, Some(null)) => {
                self.prior.a * (f64::from(null[0]) - f64::from(null[1])) + self.prior.b
            }
            _ => 0.0,
        };
        vec![l_false as f32, (l_true - correction) as f32]
    }
}

fn char_prefix(text: &str, chars: usize) -> &str {
    text.char_indices()
        .nth(chars)
        .map_or(text, |(index, _)| &text[..index])
}

struct OptionMarkerScorer {
    input_norm: LayerNorm,
    dense: Linear,
    norm: LayerNorm,
    out_proj: Linear,
}

impl OptionMarkerScorer {
    fn load(vb: VarBuilder, hidden_size: usize) -> Result<Self, ModelError> {
        Ok(Self {
            input_norm: layer_norm(hidden_size, SCORER_LAYER_NORM_EPS, vb.pp("input_norm"))?,
            dense: linear(hidden_size, hidden_size / 2, vb.pp("dense"))?,
            norm: layer_norm(hidden_size / 2, SCORER_LAYER_NORM_EPS, vb.pp("norm"))?,
            out_proj: linear(hidden_size / 2, 1, vb.pp("out_proj"))?,
        })
    }

    fn forward(&self, features: &Tensor) -> Result<Tensor, ModelError> {
        let hidden = self.input_norm.forward(features)?;
        let hidden = self.dense.forward(&hidden)?.gelu_erf()?;
        let hidden = self.norm.forward(&hidden)?;
        Ok(self.out_proj.forward(&hidden)?.squeeze(1)?)
    }
}

struct VonFamily {
    tokenizer: HfTokenizer,
    encoder: ModernBertEncoder,
    scorer: OptionMarkerScorer,
    layout: VonLayout,
    calibration: Calibration,
    device: Device,
}

const BATCH_TOKEN_BUDGET: usize = 32_768;

impl VonFamily {
    fn rows_logits(&self, rows: &[Row]) -> Result<Vec<Vec<f32>>, ModelError> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let lengths = rows.iter().map(|row| row.ids.len()).collect::<Vec<_>>();
        let mut out = vec![Vec::new(); rows.len()];
        for range in groups(&lengths, BATCH_TOKEN_BUDGET) {
            let batch = &rows[range.clone()];
            let width = batch
                .iter()
                .map(|row| row.ids.len())
                .max()
                .unwrap_or(1)
                .max(1);
            let mut ids = vec![self.layout.specials.pad; batch.len() * width];
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
                let row_hidden = hidden.i(offset)?;
                let positions = Tensor::from_vec(
                    row.markers
                        .iter()
                        .map(|position| *position as u32)
                        .collect::<Vec<u32>>(),
                    row.markers.len(),
                    &self.device,
                )?;
                let features = row_hidden.index_select(&positions, 0)?;
                out[range.start + offset] = self
                    .scorer
                    .forward(&features)?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
            }
        }
        Ok(out)
    }
}

impl Family for VonFamily {
    fn id(&self) -> &'static str {
        "von"
    }

    fn capabilities(&self) -> olaya_core::Capabilities {
        olaya_core::Capabilities::classification()
    }

    fn tokenizer(&self) -> &dyn TokenEncoder {
        &self.tokenizer
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
            return Err(DecisionError::unsupported("von does not support ordinal choices").into());
        }
        let state = self
            .layout
            .encode_state(&self.tokenizer, state.as_content())?;
        let mut answers = IndexMap::with_capacity(questions.len());
        let mut input_tokens = 0u64;
        let mut scorings = Vec::with_capacity(questions.len());
        let mut rows = Vec::new();
        for (qid, question) in questions.iter() {
            if matches!(question, Question::Box(_)) {
                return Err(ModelError::unsupported("von does not answer box questions"));
            }
            let scoring = self
                .layout
                .encode_question(&self.tokenizer, &state, qid, question)?;
            if scoring.truncated {
                return Err(DecisionError::StateTruncated.into());
            }
            input_tokens += scoring
                .rows
                .iter()
                .map(|row| row.ids.len() as u64)
                .sum::<u64>();
            rows.extend(scoring.rows.iter().cloned());
            scorings.push(scoring);
        }
        let row_logits = self.rows_logits(&rows)?;
        let mut cursor = 0;
        for ((qid, question), scoring) in questions.iter().zip(&scorings) {
            let row_logits = row_logits[cursor..cursor + scoring.rows.len()].to_vec();
            cursor += scoring.rows.len();
            let option_logits = self.layout.option_logits(scoring, &row_logits);
            let answer = match question {
                Question::Choice(choice) => {
                    let labels = choice.criteria.keys().cloned().collect::<Vec<_>>();
                    answer_from_logits(
                        question,
                        &labels,
                        &[],
                        &option_logits,
                        &self.calibration,
                        state.tokens,
                    )?
                }
                Question::Score(score) => answer_from_logits(
                    question,
                    &[],
                    &score.criteria,
                    &option_logits,
                    &self.calibration,
                    state.tokens,
                )?,
                Question::Noul(_) => answer_from_logits(
                    question,
                    &["false".to_owned(), "true".to_owned()],
                    &[],
                    &option_logits,
                    &self.calibration,
                    state.tokens,
                )?,
                Question::Box(_) => {
                    return Err(ModelError::unsupported("von does not answer box questions"));
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

fn resolve_specials(
    tokenizer: &HfTokenizer,
    config: &EncoderConfig,
) -> Result<VonSpecials, ModelError> {
    let cls = tokenizer
        .token_id(CLS_TEXT)
        .ok_or_else(|| ModelError::config("von tokenizer has no [CLS] token"))?;
    let sep = tokenizer
        .token_id(SEP_TEXT)
        .ok_or_else(|| ModelError::config("von tokenizer has no [SEP] token"))?;
    let mask = tokenizer
        .token_id(MASK_TEXT)
        .ok_or_else(|| ModelError::config("von tokenizer has no [MASK] token"))?;
    let pad = tokenizer
        .token_id(PAD_TEXT)
        .or(Some(config.pad_token_id))
        .ok_or_else(|| ModelError::config("von tokenizer has no [PAD] token"))?;
    if let Some(expected) = config.cls_token_id
        && expected != cls
    {
        return Err(ModelError::config(format!(
            "tokenizer [CLS] id {cls} does not match config {expected}"
        )));
    }
    if let Some(expected) = config.sep_token_id
        && expected != sep
    {
        return Err(ModelError::config(format!(
            "tokenizer [SEP] id {sep} does not match config {expected}"
        )));
    }
    if config.pad_token_id != pad {
        return Err(ModelError::config(format!(
            "tokenizer [PAD] id {pad} does not match config {}",
            config.pad_token_id
        )));
    }
    Ok(VonSpecials {
        cls,
        sep,
        mask,
        pad,
    })
}

fn read_calibration(ctx: &LoadContext<'_>) -> Result<(Calibration, ZeroShotPrior), ModelError> {
    let mut prior = ZeroShotPrior::default();
    let calibration = match ctx.manifest.file(FileRole::Calibration) {
        Some(file) => {
            let raw: MarkerCalibration = read_json(&ctx.dir.join(&file.path))?;
            if let Some(value) = raw.noul_zero_shot_prior {
                prior = value;
            }
            let mut map = raw.calibration_map.unwrap_or_default();
            map.insert(
                "kind".to_owned(),
                Value::String("von-entropy-length-v1".to_owned()),
            );
            Calibration::from_file(
                &CalibrationFile {
                    temperature: vec![Value::from(raw.temperature); 3],
                    temperature_map: Some(Value::Object(map)),
                    ..CalibrationFile::default()
                },
                ctx.manifest.calibration_mode,
            )
        }
        None => Calibration::default(),
    };
    Ok((calibration, prior))
}

/// Loads a Von 1.1 checkpoint.
pub struct Factory;

impl FamilyFactory for Factory {
    fn family(&self) -> &'static str {
        "von"
    }

    fn load(&self, ctx: &LoadContext<'_>) -> Result<Arc<dyn Family>, ModelError> {
        if ctx.manifest.layout != "von-option-marker-v1" {
            return Err(ModelError::config(format!(
                "von serves layout von-option-marker-v1, got {:?}",
                ctx.manifest.layout
            )));
        }
        if ctx.manifest.arch != "modernbert" {
            return Err(ModelError::config(format!(
                "von serves ModernBERT, got arch {:?}",
                ctx.manifest.arch
            )));
        }
        let config_file = ctx
            .manifest
            .file(FileRole::Config)
            .ok_or_else(|| ModelError::config("von manifest has no config file"))?;
        let mut config: EncoderConfig = read_json(&ctx.dir.join(&config_file.path))?;
        if config.max_position_embeddings < MAX_LEN {
            config.max_position_embeddings = MAX_LEN;
        }

        let tokenizer_file = ctx
            .manifest
            .file(FileRole::Tokenizer)
            .ok_or_else(|| ModelError::config("von manifest has no tokenizer file"))?;
        let tokenizer = HfTokenizer::from_file(ctx.dir.join(&tokenizer_file.path))?;
        let specials = resolve_specials(&tokenizer, &config)?;

        let weights = ctx
            .manifest
            .file(FileRole::Pickle)
            .or_else(|| ctx.manifest.file(FileRole::Weights))
            .ok_or_else(|| ModelError::config("von manifest has no pickle weights file"))?;
        let tensors = crate::torch::load_state_dict(&ctx.dir.join(&weights.path), ctx.device)?;

        let encoder_vb = VarBuilder::from_tensors(tensors.clone(), ctx.compute_dtype(), ctx.device);
        let encoder_vb = VarBuilder::from_backend(
            Box::new(Rename::new(encoder_vb, EncoderRenamer)),
            ctx.compute_dtype(),
            ctx.device.clone(),
        );
        let encoder = ModernBertEncoder::load(encoder_vb, &config)?;

        // The marker scorer stays F32; only the encoder runs in half precision.
        let scorer_vb = VarBuilder::from_tensors(tensors, DType::F32, ctx.device);
        let scorer = OptionMarkerScorer::load(scorer_vb.pp("scorer"), config.hidden_size)?;

        let (calibration, prior) = read_calibration(ctx)?;
        Ok(Arc::new(VonFamily {
            tokenizer,
            encoder,
            scorer,
            layout: VonLayout {
                specials,
                max_len: MAX_LEN,
                prior,
            },
            calibration,
            device: ctx.device.clone(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const MASK: u32 = 0x10_0000;
    const SEP: u32 = 0x10_0001;
    const CLS: u32 = 0x10_0002;

    /// One id per char, `[MASK]` / `[SEP]` as single ids, as added tokens are.
    struct Chars;

    impl TokenEncoder for Chars {
        fn encode(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
            let mut ids = Vec::new();
            let mut rest = text;
            while let Some(character) = rest.chars().next() {
                if let Some(tail) = rest.strip_prefix(MASK_TEXT) {
                    if ids.last() == Some(&u32::from(' ')) {
                        ids.pop();
                    }
                    ids.push(MASK);
                    rest = tail;
                } else if let Some(tail) = rest.strip_prefix(SEP_TEXT) {
                    ids.push(SEP);
                    rest = tail;
                } else {
                    ids.push(u32::from(character));
                    rest = &rest[character.len_utf8()..];
                }
            }
            Ok(ids)
        }
    }

    impl CharOffsetEncoder for Chars {
        fn char_ends(&self, text: &str) -> Result<Vec<usize>, ModelError> {
            Ok((1..=text.chars().count()).collect())
        }
    }

    fn layout(max_len: usize) -> VonLayout {
        VonLayout {
            specials: VonSpecials {
                cls: CLS,
                sep: SEP,
                mask: MASK,
                pad: 0,
            },
            max_len,
            prior: ZeroShotPrior::default(),
        }
    }

    fn question(value: Value) -> Question {
        serde_json::from_value(value).expect("question")
    }

    #[test]
    fn renders_state_like_python_str() {
        let content = Content::Object(
            json!({"subject": "it's", "n": 1, "p": 19.99, "ok": true, "none": null,
                   "tags": ["a", "b"], "meta": {"k": "v", "b": false}})
            .as_object()
            .cloned()
            .expect("object"),
        );
        assert_eq!(
            layout(MAX_LEN).state_text(&content),
            "subject: it's\nn: 1\np: 19.99\nok: True\nnone: None\ntags: ['a', 'b']\nmeta: {'k': 'v', 'b': False}"
        );
        assert_eq!(
            layout(MAX_LEN).state_text(&Content::String("  raw  ".to_owned())),
            "  raw  "
        );
    }

    #[test]
    fn describes_options_like_von() {
        let l = layout(MAX_LEN);
        let choice = question(json!({"type": "choice", "instructions": "x",
            "criteria": {" a ": null, "b": " desc ", "c": {"k": 1}, "d": {}, "e": [], "f": "2.5"}}));
        assert_eq!(
            l.descriptions(&choice).expect("choice"),
            (
                vec![
                    "a".to_owned(),
                    "desc".to_owned(),
                    "{'k': 1}".to_owned(),
                    "d".to_owned(),
                    "e".to_owned(),
                    "2.5".to_owned()
                ],
                false
            )
        );
        let score = question(json!({"type": "score", "instructions": "x", "criteria": [
            " low ", {"what": "mid", "examples": ["a", "b"]}, {"what": "hi", "examples": "xy"},
            {"examples": [1, true]}, {"what": null}, {"what": "w", "examples": []}, "3"]}));
        assert_eq!(
            l.descriptions(&score).expect("score").0,
            [
                "low",
                "mid Examples: a, b",
                "hi Examples: x, y",
                "Examples: 1, True",
                "None",
                "w",
                "3"
            ]
        );
        let zero = question(
            json!({"type": "noul", "instructions": "x", "criteria": {"true": "", "false": null}}),
        );
        assert_eq!(
            l.descriptions(&zero).expect("zero"),
            (
                vec![NOUL_DEFAULT_TRUE.to_owned(), NOUL_DEFAULT_FALSE.to_owned()],
                true
            )
        );
        let one =
            question(json!({"type": "noul", "instructions": "x", "criteria": {"False": "no way"}}));
        assert_eq!(
            l.descriptions(&one).expect("one"),
            (
                vec![NOUL_DEFAULT_TRUE.to_owned(), "no way".to_owned()],
                false
            )
        );
    }

    #[test]
    fn packs_rows_like_pack_sequence() {
        let l = layout(MAX_LEN);
        let state = l
            .encode_state(&Chars, &Content::String(" s [MASK] t ".to_owned()))
            .expect("state");
        assert_eq!((state.tokens, state.text.as_str()), (6, " s   t "));
        let choice = question(json!({"type": "choice", "instructions": "Pick [MASK]?",
            "criteria": {"a": "first [MASK]", "b": null}}));
        let scoring = l
            .encode_question(&Chars, &state, "q", &choice)
            .expect("scoring");
        assert_eq!(scoring.rows.len(), 1);
        assert_eq!(
            l.pack(
                &state.text,
                &l.sanitize(&l.instructions("q", &choice)),
                &["first  ".to_owned(), "b".to_owned()]
            ),
            "Pick  ?  s   t [SEP] [MASK] first [MASK] b"
        );
        assert_eq!(scoring.rows[0].ids[0], CLS);
        assert_eq!(*scoring.rows[0].ids.last().expect("last"), SEP);
        assert_eq!(scoring.rows[0].markers.len(), 2);
        assert!(
            scoring.rows[0]
                .markers
                .iter()
                .all(|marker| scoring.rows[0].ids[*marker] == MASK)
        );
        assert!(!scoring.truncated);

        let zero = question(json!({"type": "noul", "instructions": ""}));
        let scoring = l.encode_question(&Chars, &state, "q", &zero).expect("zero");
        assert_eq!(scoring.rows.len(), 2);
        assert!(scoring.zero_shot);
        assert_eq!(
            l.pack(
                &state.text,
                "",
                &[NOUL_DEFAULT_TRUE.to_owned(), NOUL_DEFAULT_FALSE.to_owned()]
            ),
            "s   t [SEP] [MASK] Yes, condition holds true. [MASK] No, condition is false."
        );
    }

    #[test]
    fn cuts_the_state_to_fit() {
        let choice = question(json!({
            "type": "choice",
            "instructions": "Q",
            "criteria": {"a": null, "b": null}
        }));
        let fits = |max_len: usize, state_text: &str| {
            let l = layout(max_len);
            let state = l
                .encode_state(&Chars, &Content::String(state_text.to_owned()))
                .expect("state");
            l.encode_question(&Chars, &state, "q", &choice)
        };
        let scoring = fits(20, "0123456789").expect("fits");
        assert_eq!(scoring.rows[0].ids.len(), 20);
        assert!(scoring.truncated);
        assert!(!fits(22, "0123456789").expect("fits").truncated);
        assert!(matches!(
            fits(10, "0123456789"),
            Err(ModelError::Decision(DecisionError::Invalid { message }))
                if message.contains("alone take 11 tokens")
        ));
    }

    #[test]
    fn derives_option_logits() {
        let l = layout(MAX_LEN);
        let scoring = |qtype, zero_shot| Scoring {
            qtype,
            rows: Vec::new(),
            zero_shot,
            truncated: false,
        };
        assert_eq!(
            l.option_logits(
                &scoring(QuestionType::Choice, false),
                &[vec![1.0, 2.0, 3.0]]
            ),
            [1.0, 2.0, 3.0]
        );
        assert_eq!(
            l.option_logits(&scoring(QuestionType::Noul, false), &[vec![1.0, -1.0]]),
            [-1.0, 1.0]
        );
        let logits = l.option_logits(
            &scoring(QuestionType::Noul, true),
            &[vec![1.0, -1.0], vec![2.0, 0.0]],
        );
        assert_eq!(logits[0], -1.0);
        assert!((logits[1] - (1.0 - 0.7 * 2.0)).abs() < 1e-6);
    }
}
