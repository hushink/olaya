//! Decima (`decima-late-interaction-v1` / `-v2`): an encoder with a
//! late-interaction option scorer.
//!
//! Every question becomes one state row and one row per option, each encoded on
//! its own:
//!
//! ```text
//! state row   [cls] tok(prefix + normalize(text + "\n" + state))[..max-2] [sep]
//! option row  [cls] tok(option_text)[..max-2] [sep]
//! ```
//!
//! A two-layer scorer reads the option tokens through the state tokens, pools
//! them, and adds a bi-encoder cosine skip term. Choice and noul answers are a
//! softmax over the scores; score answers use the cumulative-link ordinal head.
//!
//! The encoder is BERT (decima-small's multilingual-e5-small) or ModernBERT
//! (decima-base/agent/guard); the manifest's `arch` selects it.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use crate::encoders::bert::{BertConfig, BertEncoder, read_config as read_bert_config};
use crate::encoders::modernbert::{EncoderConfig, ModernBertEncoder, StripModelRenamer};
use crate::tokenizer::HfTokenizer;
use crate::{Family, FamilyFactory, LoadContext, ModelError};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::softmax;
use candle_nn::var_builder::Rename;
use candle_nn::{LayerNorm, Linear, Module, VarBuilder, layer_norm, linear};
use indexmap::IndexMap;
use olaya_core::{
    Calibration, CalibrationFile, CalibrationMode, Content, DecisionError, FileRole, Question,
    QuestionType, State, SystemOneResponse, TokenEncoder, Usage, answer_from_logits, choice_answer,
    dumps_content, pyrepr,
};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

const LAYOUT: &str = "decima-late-interaction-v1";
const LAYOUT_FIT: &str = "decima-late-interaction-v2";
const NOUL_OPTIONS: [&str; 2] = ["yes", "no"];
const NOUL_CRITERIA: &str = "\nTrue if: {true}\nFalse if: {false}";
const NOUL_MISSING: &str = "\u{2014}";
const MAX_CHOICES: usize = 255;
const MIN_LEVELS: usize = 2;
const MAX_LEVELS: usize = 10;
const HEAD_LAYER_NORM_EPS: f64 = 1e-5;
const SCORER_TOKEN_BUDGET: usize = 65_536;
const MAX_SCORER_OPTIONS: usize = 64;

fn default_scorer_layers() -> usize {
    2
}

fn default_heads() -> usize {
    6
}

fn default_ffn() -> usize {
    1536
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize)]
struct DecimaConfig {
    #[serde(default = "default_scorer_layers")]
    scorer_layers: usize,
    #[serde(default = "default_heads")]
    heads: usize,
    #[serde(default = "default_ffn")]
    ffn: usize,
    max_state_tokens: usize,
    max_choice_tokens: usize,
    temperature: f64,
    #[serde(default)]
    question_in_state: bool,
    #[serde(default = "default_true")]
    question_in_choices: bool,
    #[serde(default)]
    choice_fit: bool,
    #[serde(default)]
    state_prefix: String,
    #[serde(default)]
    choice_prefix: String,
}

#[derive(Clone, Copy, Debug)]
struct DecimaTokens {
    cls: u32,
    sep: u32,
    pad: u32,
}

struct DecimaLayout {
    config: DecimaConfig,
    specials: DecimaTokens,
}

struct DecimaQuestion {
    qtype: QuestionType,
    state_ids: Vec<u32>,
    option_ids: Vec<Vec<u32>>,
    state_truncated: bool,
}

/// `(raw scores, ord_g, ord_gap)` per option.
type OptionOutputs = (Vec<f32>, Vec<f32>, Vec<f32>);

type WeightMaps = (HashMap<String, Tensor>, HashMap<String, Tensor>);

trait DecimaTokenizer: TokenEncoder {
    fn nfc(&self, text: &str) -> String;
}

impl DecimaTokenizer for HfTokenizer {
    fn nfc(&self, text: &str) -> String {
        HfTokenizer::nfc(self, text)
    }
}

impl DecimaLayout {
    fn layout_id(&self) -> &'static str {
        if self.config.choice_fit {
            LAYOUT_FIT
        } else {
            LAYOUT
        }
    }

    fn validate(&self) -> Result<(), ModelError> {
        if self.config.max_state_tokens < 3 || self.config.max_choice_tokens < 3 {
            return Err(ModelError::config(
                "max_state_tokens and max_choice_tokens must leave room for a token",
            ));
        }
        if !(self.config.temperature.is_finite() && self.config.temperature > 0.0) {
            return Err(ModelError::config(format!(
                "temperature {} is not positive",
                self.config.temperature
            )));
        }
        if self.config.scorer_layers == 0 || self.config.heads == 0 || self.config.ffn == 0 {
            return Err(ModelError::config(
                "scorer_layers, heads and ffn must be positive",
            ));
        }
        Ok(())
    }

    fn text_of(value: Option<&Content>) -> String {
        match value {
            None | Some(Content::Null) => String::new(),
            Some(Content::String(text)) => text.clone(),
            Some(other) => dumps_content(other),
        }
    }

    fn normalize(enc: &dyn DecimaTokenizer, text: &str) -> String {
        pyrepr::strip(&enc.nfc(text)).to_owned()
    }

    fn fill(template: &str, true_text: &str, false_text: &str) -> String {
        let mut out = String::with_capacity(template.len() + true_text.len() + false_text.len());
        let mut rest = template;
        while let Some(character) = rest.chars().next() {
            if let Some(tail) = rest.strip_prefix("{true}") {
                out.push_str(true_text);
                rest = tail;
            } else if let Some(tail) = rest.strip_prefix("{false}") {
                out.push_str(false_text);
                rest = tail;
            } else {
                out.push(character);
                rest = &rest[character.len_utf8()..];
            }
        }
        out
    }

    fn question(
        &self,
        qid: &str,
        question: &Question,
    ) -> Result<(String, Vec<String>), ModelError> {
        let mut body = question.instructions_or_id(qid);
        let options = match question {
            Question::Choice(choice) => {
                if choice.ordinal {
                    if !(MIN_LEVELS..=MAX_LEVELS).contains(&choice.criteria.len()) {
                        return Err(DecisionError::invalid(format!(
                            "question {qid:?}: ordinal choice needs {MIN_LEVELS}-{MAX_LEVELS} criteria"
                        ))
                        .into());
                    }
                } else if !(2..=MAX_CHOICES).contains(&choice.criteria.len()) {
                    return Err(DecisionError::invalid(format!(
                        "question {qid:?}: choice needs 2-{MAX_CHOICES} criteria"
                    ))
                    .into());
                }
                choice
                    .criteria
                    .iter()
                    .map(|(label, value)| {
                        let description = Self::text_of(Some(value));
                        if pyrepr::strip(&description).is_empty() {
                            label.clone()
                        } else {
                            format!("{label}: {description}")
                        }
                    })
                    .collect()
            }
            Question::Score(score) => {
                if !(MIN_LEVELS..=MAX_LEVELS).contains(&score.criteria.len()) {
                    return Err(DecisionError::invalid(format!(
                        "question {qid:?}: score needs {MIN_LEVELS}-{MAX_LEVELS} levels"
                    ))
                    .into());
                }
                score
                    .criteria
                    .iter()
                    .map(|level| Self::text_of(Some(level)))
                    .collect()
            }
            Question::Noul(noul) => {
                let (true_text, false_text) = match &noul.criteria {
                    Some(criteria) => (
                        Self::text_of(criteria.is_true.as_ref()),
                        Self::text_of(criteria.is_false.as_ref()),
                    ),
                    None => (String::new(), String::new()),
                };
                let true_text = pyrepr::strip(&true_text).to_owned();
                let false_text = pyrepr::strip(&false_text).to_owned();
                if !true_text.is_empty() || !false_text.is_empty() {
                    body.push_str(&Self::fill(
                        NOUL_CRITERIA,
                        if true_text.is_empty() {
                            NOUL_MISSING
                        } else {
                            &true_text
                        },
                        if false_text.is_empty() {
                            NOUL_MISSING
                        } else {
                            &false_text
                        },
                    ));
                }
                NOUL_OPTIONS
                    .iter()
                    .map(|option| (*option).to_owned())
                    .collect()
            }
            Question::Box(_) => {
                return Err(ModelError::unsupported(
                    "decima does not answer box questions",
                ));
            }
        };
        Ok((body, options))
    }

    fn ids(
        &self,
        enc: &dyn DecimaTokenizer,
        text: &str,
        max_len: usize,
    ) -> Result<(Vec<u32>, bool), ModelError> {
        let body = enc.encode(text)?;
        let keep = body.len().min(max_len - 2);
        let mut ids = Vec::with_capacity(keep + 2);
        ids.push(self.specials.cls);
        ids.extend_from_slice(&body[..keep]);
        ids.push(self.specials.sep);
        Ok((ids, keep < body.len()))
    }

    fn row(
        &self,
        enc: &dyn DecimaTokenizer,
        prefix: &str,
        body: &str,
        max_len: usize,
    ) -> Result<(Vec<u32>, bool), ModelError> {
        let normalized = Self::normalize(enc, body);
        self.ids(enc, &format!("{prefix}{normalized}"), max_len)
    }

    fn tokens(&self, enc: &dyn DecimaTokenizer, text: &str) -> Result<usize, ModelError> {
        Ok(enc.encode(text)?.len() + 2)
    }

    fn option_texts(
        &self,
        enc: &dyn DecimaTokenizer,
        question: &str,
        options: &[String],
    ) -> Result<Vec<String>, ModelError> {
        let prefix = self.config.choice_prefix.as_str();
        let max_len = self.config.max_choice_tokens;
        let full: Vec<String> = options
            .iter()
            .map(|option| {
                let text = format!("{question} {option}");
                format!("{prefix}{}", Self::normalize(enc, pyrepr::strip(&text)))
            })
            .collect();
        if !self.config.choice_fit {
            let mut rows = HashSet::with_capacity(full.len());
            for text in &full {
                rows.insert(self.ids(enc, text, max_len)?.0);
            }
            if rows.len() == full.len() {
                return Ok(full);
            }
        }
        let question = Self::normalize(enc, question);
        let words: Vec<&str> = pyrepr::split(&question).collect();
        let fitted = |head: usize, option: &str| {
            format!("{prefix}{} \u{2026} {option}", words[..head].join(" "))
        };
        let mut out = Vec::with_capacity(options.len());
        for (option, full) in options.iter().zip(full) {
            if self.tokens(enc, &full)? <= max_len {
                out.push(full);
                continue;
            }
            let option = Self::normalize(enc, option);
            let (mut lo, mut hi) = (0usize, words.len());
            while lo < hi {
                let mid = (lo + hi).div_ceil(2);
                if self.tokens(enc, &fitted(mid, &option))? <= max_len {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            out.push(if lo > 0 {
                fitted(lo, &option)
            } else {
                format!("{prefix}{option}")
            });
        }
        Ok(out)
    }

    fn encode_question(
        &self,
        enc: &dyn DecimaTokenizer,
        qid: &str,
        question: &Question,
        state: &str,
    ) -> Result<DecimaQuestion, ModelError> {
        let (body, options) = self.question(qid, question)?;
        let state_text = if self.config.question_in_state && !body.is_empty() {
            format!("{body}\n{state}")
        } else {
            state.to_owned()
        };
        let (state_ids, state_truncated) = self.row(
            enc,
            &self.config.state_prefix,
            &state_text,
            self.config.max_state_tokens,
        )?;
        let option_question = if self.config.question_in_choices {
            body.as_str()
        } else {
            ""
        };
        let option_ids = self
            .option_texts(enc, option_question, &options)?
            .iter()
            .map(|text| {
                self.ids(enc, text, self.config.max_choice_tokens)
                    .map(|(ids, _)| ids)
            })
            .collect::<Result<_, _>>()?;
        // An ordinal choice is read by the cumulative-link head; the answer
        // is still a choice over the same labels.
        let qtype = match question {
            Question::Choice(choice) if choice.ordinal => QuestionType::Score,
            question => question.question_type(),
        };
        Ok(DecimaQuestion {
            qtype,
            state_ids,
            option_ids,
            state_truncated,
        })
    }
}

fn softplus(value: f64) -> f64 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn log_softmax(values: &[f64]) -> Vec<f64> {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_sum = max
        + values
            .iter()
            .map(|value| (value - max).exp())
            .sum::<f64>()
            .ln();
    values.iter().map(|value| value - log_sum).collect()
}

/// `DecimaModel.ordinal_log_probs` for one question.
pub fn ordinal_log_probs(scores: &[f32], g: &[f32], gap: &[f32], temperature: f64) -> Vec<f64> {
    let k = scores.len();
    if k == 0 {
        return Vec::new();
    }
    let tempered: Vec<f64> = scores
        .iter()
        .map(|score| f64::from(*score) / temperature)
        .collect();
    let probs = log_softmax(&tempered);
    let expected = probs
        .iter()
        .enumerate()
        .map(|(index, log_prob)| log_prob.exp() * index as f64)
        .sum::<f64>()
        - (k as f64 - 1.0) / 2.0;
    let latent = g.iter().map(|&value| f64::from(value)).sum::<f64>() / k as f64 + expected;
    let gaps: Vec<f64> = gap
        .iter()
        .map(|&value| softplus(f64::from(value)) + 1e-3)
        .collect();
    if gaps.len() < k - 1 {
        // A malformed head returned fewer gap logits than boundaries; report
        // an empty distribution instead of panicking on the slice below.
        return Vec::new();
    }
    let half = gaps.iter().sum::<f64>() / 2.0;
    let mut above = Vec::with_capacity(k - 1);
    let mut threshold = 0.0;
    for gap in &gaps[..k - 1] {
        threshold += gap;
        above.push(sigmoid(latent - (threshold - half)));
    }
    let levels: Vec<f64> = (0..k)
        .map(|index| {
            let high = if index == 0 { 1.0 } else { above[index - 1] };
            let low = if index + 1 == k { 0.0 } else { above[index] };
            (high - low).max(1e-7).ln()
        })
        .collect();
    log_softmax(&levels)
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(vb: VarBuilder, hidden: usize, heads: usize) -> Result<Self, ModelError> {
        if !hidden.is_multiple_of(heads) {
            return Err(ModelError::config(format!(
                "hidden size {hidden} is not divisible by {heads} heads"
            )));
        }
        Ok(Self {
            q: linear(hidden, hidden, vb.pp("q"))?,
            k: linear(hidden, hidden, vb.pp("k"))?,
            v: linear(hidden, hidden, vb.pp("v"))?,
            o: linear(hidden, hidden, vb.pp("o"))?,
            heads,
            head_dim: hidden / heads,
        })
    }

    fn forward(&self, x: &Tensor, mem: &Tensor, mem_pad: &Tensor) -> Result<Tensor, ModelError> {
        let (batch, query_len, hidden) = x.dims3()?;
        let key_len = mem.dim(1)?;
        let q = self
            .q
            .forward(x)?
            .reshape((batch, query_len, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(mem)?
            .reshape((batch, key_len, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .transpose(2, 3)?
            .contiguous()?;
        let v = self
            .v
            .forward(mem)?
            .reshape((batch, key_len, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let att = q.matmul(&k)?;
        let att = att.affine(scale, 0.0)?;
        let pad = mem_pad
            .to_dtype(DType::F32)?
            .unsqueeze(1)?
            .unsqueeze(2)?
            .broadcast_as((batch, 1, query_len, key_len))?
            .affine(-1e4, 0.0)?;
        let att = att.broadcast_add(&pad)?;
        let att = softmax(&att, 3)?;
        let out = att.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, query_len, hidden))?;
        Ok(self.o.forward(&out)?)
    }
}

struct ScorerLayer {
    self_attn: Attention,
    cross_attn: Attention,
    ff0: Linear,
    ff3: Linear,
    n1: LayerNorm,
    n2: LayerNorm,
    n3: LayerNorm,
}

impl ScorerLayer {
    fn load(vb: VarBuilder, hidden: usize, heads: usize, ffn: usize) -> Result<Self, ModelError> {
        Ok(Self {
            self_attn: Attention::load(vb.pp("self_attn"), hidden, heads)?,
            cross_attn: Attention::load(vb.pp("cross_attn"), hidden, heads)?,
            ff0: linear(hidden, ffn, vb.pp("ff.0"))?,
            ff3: linear(ffn, hidden, vb.pp("ff.3"))?,
            n1: layer_norm(hidden, HEAD_LAYER_NORM_EPS, vb.pp("n1"))?,
            n2: layer_norm(hidden, HEAD_LAYER_NORM_EPS, vb.pp("n2"))?,
            n3: layer_norm(hidden, HEAD_LAYER_NORM_EPS, vb.pp("n3"))?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        mem: &Tensor,
        mem_pad: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let x_pad = mask.affine(-1.0, 1.0)?;
        let hidden = self.n1.forward(x)?;
        let x = (x + self.self_attn.forward(&hidden, &hidden, &x_pad)?)?;
        let hidden = self.n2.forward(&x)?;
        let x = (x + self.cross_attn.forward(&hidden, mem, mem_pad)?)?;
        let ff = self.n3.forward(&x)?;
        let ff = self.ff0.forward(&ff)?.gelu_erf()?;
        let ff = self.ff3.forward(&ff)?;
        Ok((x + ff)?)
    }
}

struct DecimaHead {
    layers: Vec<ScorerLayer>,
    out_norm: LayerNorm,
    score: Linear,
    ord_g: Linear,
    ord_gap: Linear,
    sim_scale: f32,
    sim_center: f32,
}

impl DecimaHead {
    fn load(vb: VarBuilder, config: &DecimaConfig, hidden: usize) -> Result<Self, ModelError> {
        let mut layers = Vec::with_capacity(config.scorer_layers);
        for index in 0..config.scorer_layers {
            layers.push(ScorerLayer::load(
                vb.pp(format!("layers.{index}")),
                hidden,
                config.heads,
                config.ffn,
            )?);
        }
        let sim_scale = vb.get_unchecked("sim_scale")?.to_scalar::<f32>()?;
        let sim_center = vb.get_unchecked("sim_center")?.to_scalar::<f32>()?;
        Ok(Self {
            layers,
            out_norm: layer_norm(hidden, HEAD_LAYER_NORM_EPS, vb.pp("out_norm"))?,
            score: linear(hidden, 1, vb.pp("score"))?,
            ord_g: linear(hidden, 1, vb.pp("ord_g"))?,
            ord_gap: linear(hidden, 1, vb.pp("ord_gap"))?,
            sim_scale,
            sim_center,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        mask: &Tensor,
        mem: &Tensor,
        mem_pad: &Tensor,
        state_mean: &Tensor,
    ) -> Result<OptionOutputs, ModelError> {
        let mut x = hidden.clone();
        for layer in &self.layers {
            x = layer.forward(&x, mask, mem, mem_pad)?;
        }
        let pooled = masked_mean(&self.out_norm.forward(&x)?, mask)?;
        let score = self.score.forward(&pooled)?.squeeze(1)?;
        let option_mean = masked_mean(hidden, mask)?;
        let batch = option_mean.dim(0)?;
        let state_mean = state_mean.broadcast_as((batch, state_mean.dim(1)?))?;
        let sim = cosine(&state_mean, &option_mean)?;
        let sim = sim.affine(
            f64::from(self.sim_scale),
            -f64::from(self.sim_center) * f64::from(self.sim_scale),
        )?;
        let logits = (&score + &sim)?;
        let ord_g = self.ord_g.forward(&pooled)?.squeeze(1)?;
        let ord_gap = self.ord_gap.forward(&pooled)?.squeeze(1)?;
        Ok((
            logits.to_dtype(DType::F32)?.to_vec1::<f32>()?,
            ord_g.to_dtype(DType::F32)?.to_vec1::<f32>()?,
            ord_gap.to_dtype(DType::F32)?.to_vec1::<f32>()?,
        ))
    }
}

fn masked_mean(hidden: &Tensor, mask: &Tensor) -> Result<Tensor, ModelError> {
    let mask = mask.to_dtype(hidden.dtype())?.unsqueeze(2)?;
    let sum = hidden.broadcast_mul(&mask)?.sum(1)?;
    let count = mask.sum(1)?.clamp(1.0, 1e30)?;
    Ok(sum.broadcast_div(&count)?)
}

fn cosine(left: &Tensor, right: &Tensor) -> Result<Tensor, ModelError> {
    let dot = (left * right)?.sum(1)?;
    let left_norm = left.sqr()?.sum(1)?.sqrt()?;
    let right_norm = right.sqr()?.sum(1)?.sqrt()?;
    let denom = (left_norm * right_norm)?.clamp(1e-8, 1e30)?;
    Ok((dot / denom)?)
}

enum Encoder {
    Bert(BertEncoder),
    Modern(ModernBertEncoder),
}

impl Encoder {
    fn forward(&self, ids: &Tensor, mask: &Tensor) -> Result<Tensor, ModelError> {
        match self {
            Self::Bert(encoder) => encoder.forward(ids, mask),
            Self::Modern(encoder) => encoder.forward(ids, mask),
        }
    }
}

struct DecimaFamily {
    tokenizer: HfTokenizer,
    encoder: Encoder,
    head: DecimaHead,
    layout: DecimaLayout,
    calibration: Calibration,
    device: Device,
}

impl DecimaFamily {
    fn score(
        &self,
        state_ids: &[u32],
        option_ids: &[Vec<u32>],
    ) -> Result<OptionOutputs, ModelError> {
        let state_len = state_ids.len();
        let state = Tensor::from_vec(state_ids.to_vec(), (1, state_len), &self.device)?;
        let state_mask = Tensor::ones((1, state_len), DType::F32, &self.device)?;
        // The scorer head stays F32 even when the encoder runs in half precision.
        let state_hidden = self
            .encoder
            .forward(&state, &state_mask)?
            .to_dtype(DType::F32)?;
        let state_mean = masked_mean(&state_hidden, &state_mask)?;
        let hidden_size = state_hidden.dim(2)?;

        let count = option_ids.len();
        let option_len = option_ids.iter().map(Vec::len).max().unwrap_or(1);
        let mut ids = vec![self.layout.specials.pad; count * option_len];
        let mut mask = vec![0f32; count * option_len];
        for (row, option) in option_ids.iter().enumerate() {
            let offset = row * option_len;
            ids[offset..offset + option.len()].copy_from_slice(option);
            mask[offset..offset + option.len()].fill(1.0);
        }
        let ids = Tensor::from_vec(ids, (count, option_len), &self.device)?;
        let mask = Tensor::from_vec(mask, (count, option_len), &self.device)?;
        let option_hidden = self.encoder.forward(&ids, &mask)?.to_dtype(DType::F32)?;

        let mut scores = Vec::with_capacity(count);
        let mut ord_g = Vec::with_capacity(count);
        let mut ord_gap = Vec::with_capacity(count);
        let chunk = (SCORER_TOKEN_BUDGET / state_len.max(1)).clamp(1, MAX_SCORER_OPTIONS);
        let mut start = 0;
        while start < count {
            let size = (count - start).min(chunk);
            let mem = state_hidden.broadcast_as((size, state_len, hidden_size))?;
            let mem_pad = Tensor::zeros((size, state_len), DType::U8, &self.device)?;
            let hidden = option_hidden.narrow(0, start, size)?;
            let mask = mask.narrow(0, start, size)?;
            let (batch_scores, batch_g, batch_gap) =
                self.head
                    .forward(&hidden, &mask, &mem, &mem_pad, &state_mean)?;
            scores.extend(batch_scores);
            ord_g.extend(batch_g);
            ord_gap.extend(batch_gap);
            start += size;
        }
        Ok((scores, ord_g, ord_gap))
    }

    fn answer_logits(
        &self,
        qtype: QuestionType,
        scores: &[f32],
        ord_g: &[f32],
        ord_gap: &[f32],
    ) -> Vec<f32> {
        match qtype {
            QuestionType::Choice => scores.to_vec(),
            QuestionType::Noul => scores.iter().rev().copied().collect(),
            QuestionType::Score => {
                ordinal_log_probs(scores, ord_g, ord_gap, self.layout.config.temperature)
                    .into_iter()
                    .map(|value| value as f32)
                    .collect()
            }
            QuestionType::Box => Vec::new(),
        }
    }
}

impl Family for DecimaFamily {
    fn id(&self) -> &'static str {
        "decima"
    }

    fn capabilities(&self) -> olaya_core::Capabilities {
        olaya_core::Capabilities::classification().with_ordinal()
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
        let state = match state.as_content() {
            Content::String(text) => text.clone(),
            Content::Object(_) | Content::Array(_) => dumps_content(state.as_content()),
            Content::Null => {
                return Err(
                    DecisionError::invalid("state must be a string, object or array").into(),
                );
            }
        };
        let state_tokens = self.tokenizer.encode(&state)?.len();
        let mut encodings = Vec::with_capacity(questions.len());
        let mut input_tokens = 0u64;
        for (qid, question) in questions.iter() {
            if matches!(question, Question::Box(_)) {
                return Err(ModelError::unsupported(
                    "decima does not answer box questions",
                ));
            }
            let encoding = self
                .layout
                .encode_question(&self.tokenizer, qid, question, &state)?;
            // A state that no longer fits is a validation error, not a
            // silently shortened answer.
            if encoding.state_truncated {
                return Err(DecisionError::StateTruncated.into());
            }
            input_tokens += encoding.state_ids.len() as u64
                + encoding
                    .option_ids
                    .iter()
                    .map(|ids| ids.len() as u64)
                    .sum::<u64>();
            encodings.push((qid, question, encoding));
        }
        let mut answers = IndexMap::with_capacity(questions.len());
        for (qid, question, encoding) in encodings {
            let (scores, ord_g, ord_gap) = self.score(&encoding.state_ids, &encoding.option_ids)?;
            let logits = self.answer_logits(encoding.qtype, &scores, &ord_g, &ord_gap);
            let answer = match question {
                Question::Choice(choice) if choice.ordinal => {
                    // `ordinal_log_probs` already carries the config
                    // temperature; `answer_from_logits` would apply the choice
                    // temperature a second time. Upstream renormalises the
                    // log-probabilities (`decima.probs`), which also removes
                    // the f32 rounding of the log-probabilities themselves.
                    let probabilities =
                        olaya_core::softmax(logits.iter().map(|&value| f64::from(value)));
                    let labels = choice.criteria.keys().cloned().collect::<Vec<_>>();
                    choice_answer(&labels, &probabilities)
                }
                Question::Choice(choice) => {
                    let labels = choice.criteria.keys().cloned().collect::<Vec<_>>();
                    answer_from_logits(
                        question,
                        &labels,
                        &[],
                        &logits,
                        &self.calibration,
                        state_tokens,
                    )?
                }
                Question::Score(score) => answer_from_logits(
                    question,
                    &[],
                    &score.criteria,
                    &logits,
                    &self.calibration,
                    state_tokens,
                )?,
                Question::Noul(_) => answer_from_logits(
                    question,
                    &["false".to_owned(), "true".to_owned()],
                    &[],
                    &logits,
                    &self.calibration,
                    state_tokens,
                )?,
                Question::Box(_) => {
                    return Err(ModelError::unsupported(
                        "decima does not answer box questions",
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

fn resolve_specials(tokenizer: &HfTokenizer) -> Result<DecimaTokens, ModelError> {
    // Prefer the model's sentencepiece specials: some tokenizers also expose
    // `[CLS]`/`[SEP]` as ordinary tokens, which would be the wrong ids.
    let cls = tokenizer
        .token_id("<bos>")
        .or_else(|| tokenizer.token_id("<s>"))
        .or_else(|| tokenizer.token_id("[CLS]"))
        .ok_or_else(|| ModelError::config("decima tokenizer has no CLS token"))?;
    let sep = tokenizer
        .token_id("<eos>")
        .or_else(|| tokenizer.token_id("</s>"))
        .or_else(|| tokenizer.token_id("[SEP]"))
        .ok_or_else(|| ModelError::config("decima tokenizer has no SEP token"))?;
    let pad = tokenizer
        .token_id("<pad>")
        .or_else(|| tokenizer.token_id("[PAD]"))
        .ok_or_else(|| ModelError::config("decima tokenizer has no PAD token"))?;
    Ok(DecimaTokens { cls, sep, pad })
}

fn weights_maps(ctx: &LoadContext<'_>) -> Result<WeightMaps, ModelError> {
    let weights = ctx
        .manifest
        .files
        .iter()
        .filter(|file| file.role == FileRole::Weights)
        .collect::<Vec<_>>();
    let encoder = weights
        .iter()
        .find(|file| file.path.ends_with("encoder/model.safetensors"))
        .or_else(|| {
            weights
                .iter()
                .find(|file| !file.path.ends_with("head.safetensors"))
        })
        .ok_or_else(|| ModelError::config("decima manifest has no encoder weights"))?;
    let head = weights
        .iter()
        .find(|file| file.path.ends_with("head.safetensors"))
        .ok_or_else(|| ModelError::config("decima manifest has no head weights"))?;
    let mut encoder_tensors =
        candle_core::safetensors::load(ctx.dir.join(&encoder.path), ctx.device)?;
    let mut head_tensors = candle_core::safetensors::load(ctx.dir.join(&head.path), ctx.device)?;
    for tensor in encoder_tensors.values_mut() {
        if tensor.dtype() != DType::F32 {
            *tensor = tensor.to_dtype(DType::F32)?;
        }
    }
    for tensor in head_tensors.values_mut() {
        if tensor.dtype() != DType::F32 {
            *tensor = tensor.to_dtype(DType::F32)?;
        }
    }
    Ok((encoder_tensors, head_tensors))
}

fn read_encoder_config(ctx: &LoadContext<'_>) -> Result<EncoderConfig, ModelError> {
    let file = ctx
        .manifest
        .file(FileRole::Config)
        .ok_or_else(|| ModelError::config("decima manifest has no encoder config"))?;
    read_json(&ctx.dir.join(&file.path))
}

fn read_bert_encoder_config(ctx: &LoadContext<'_>) -> Result<BertConfig, ModelError> {
    let file = ctx
        .manifest
        .file(FileRole::Config)
        .ok_or_else(|| ModelError::config("decima manifest has no encoder config"))?;
    read_bert_config(&ctx.dir.join(&file.path))
}

/// Loads a Decima checkpoint.
pub struct Factory;

impl FamilyFactory for Factory {
    fn family(&self) -> &'static str {
        "decima"
    }

    fn load(&self, ctx: &LoadContext<'_>) -> Result<Arc<dyn Family>, ModelError> {
        if ![LAYOUT, LAYOUT_FIT].contains(&ctx.manifest.layout.as_str()) {
            return Err(ModelError::config(format!(
                "decima serves {LAYOUT} / {LAYOUT_FIT}, got {:?}",
                ctx.manifest.layout
            )));
        }
        let config_file = ctx
            .manifest
            .file(FileRole::Decision)
            .ok_or_else(|| ModelError::config("decima manifest has no decima.json"))?;
        let config: DecimaConfig = read_json(&ctx.dir.join(&config_file.path))?;
        let tokenizer_file = ctx
            .manifest
            .file(FileRole::Tokenizer)
            .ok_or_else(|| ModelError::config("decima manifest has no tokenizer"))?;
        let tokenizer = HfTokenizer::from_file(ctx.dir.join(&tokenizer_file.path))?;
        let specials = resolve_specials(&tokenizer)?;
        let layout = DecimaLayout { config, specials };
        layout.validate()?;
        if ctx.manifest.layout != layout.layout_id() {
            return Err(ModelError::config(format!(
                "decima config choice_fit={} requires layout {}, got {:?}",
                layout.config.choice_fit,
                layout.layout_id(),
                ctx.manifest.layout
            )));
        }

        let (encoder_tensors, head_tensors) = weights_maps(ctx)?;
        // candle's BERT builds its attention mask with `f32::MIN`, which
        // saturates to -inf in F16/BF16; `0 * -inf` then turns every real key
        // into NaN. Keep the BERT checkpoint (decima-small) on F32 until that
        // encoder is dtype-safe.
        let encoder_dtype = if ctx.manifest.arch == "bert" {
            DType::F32
        } else {
            ctx.compute_dtype()
        };
        let (encoder, hidden) = match ctx.manifest.arch.as_str() {
            "bert" => {
                let config = read_bert_encoder_config(ctx)?;
                if config.max_position_embeddings < layout.config.max_state_tokens {
                    return Err(ModelError::config(format!(
                        "BERT config allows {} positions, decima needs {}",
                        config.max_position_embeddings, layout.config.max_state_tokens
                    )));
                }
                let hidden = config.hidden_size;
                let vb = VarBuilder::from_tensors(encoder_tensors, encoder_dtype, ctx.device);
                (Encoder::Bert(BertEncoder::load(vb, &config)?), hidden)
            }
            "modernbert" => {
                let mut config = read_encoder_config(ctx)?;
                config.max_position_embeddings = config
                    .max_position_embeddings
                    .max(layout.config.max_state_tokens);
                let hidden = config.hidden_size;
                let has_model_prefix = encoder_tensors.keys().any(|key| key.starts_with("model."));
                let vb = VarBuilder::from_tensors(encoder_tensors, encoder_dtype, ctx.device);
                let vb = if has_model_prefix {
                    vb
                } else {
                    VarBuilder::from_backend(
                        Box::new(Rename::new(vb, StripModelRenamer)),
                        encoder_dtype,
                        ctx.device.clone(),
                    )
                };
                (
                    Encoder::Modern(ModernBertEncoder::load(vb, &config)?),
                    hidden,
                )
            }
            other => {
                return Err(ModelError::config(format!(
                    "decima serves BERT or ModernBERT encoders, got arch {other:?}"
                )));
            }
        };
        let head_vb = VarBuilder::from_tensors(head_tensors, DType::F32, ctx.device);
        let head = DecimaHead::load(head_vb, &layout.config, hidden)?;
        let calibration = Calibration::from_file(
            &CalibrationFile {
                temperature: vec![
                    Value::from(layout.config.temperature),
                    Value::from(1.0),
                    Value::from(layout.config.temperature),
                ],
                ..CalibrationFile::default()
            },
            CalibrationMode::Raw,
        );
        Ok(Arc::new(DecimaFamily {
            tokenizer,
            encoder,
            head,
            layout,
            calibration,
            device: ctx.device.clone(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const SEP: u32 = 0x10_0001;
    const CLS: u32 = 0x10_0002;

    struct Chars;

    impl TokenEncoder for Chars {
        fn encode(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
            Ok(text.chars().map(u32::from).collect())
        }
    }

    impl DecimaTokenizer for Chars {
        fn nfc(&self, text: &str) -> String {
            text.replace("e\u{301}", "\u{e9}")
        }
    }

    fn layout(max_state: usize, max_option: usize) -> DecimaLayout {
        DecimaLayout {
            config: DecimaConfig {
                scorer_layers: 2,
                heads: 6,
                ffn: 1536,
                max_state_tokens: max_state,
                max_choice_tokens: max_option,
                temperature: 0.5,
                question_in_state: true,
                question_in_choices: true,
                choice_fit: false,
                state_prefix: "q: ".to_owned(),
                choice_prefix: "p: ".to_owned(),
            },
            specials: DecimaTokens {
                cls: CLS,
                sep: SEP,
                pad: 1,
            },
        }
    }

    fn questions(value: Value) -> olaya_core::Questions {
        let map: IndexMap<String, Question> = serde_json::from_value(value).expect("questions");
        map.into()
    }

    fn read(ids: &[u32]) -> String {
        assert_eq!((ids[0], ids[ids.len() - 1]), (CLS, SEP));
        ids[1..ids.len() - 1]
            .iter()
            .map(|&token| char::from_u32(token).unwrap())
            .collect()
    }

    #[test]
    fn maps_questions_like_systemone() {
        let layout = layout(512, 64);
        let qs = questions(json!({
            "c": {"type": "choice", "instructions": {"k": "\u{e9}"},
                  "criteria": {"a": null, "b": " ", "c": "desc", "e": {"x": [1, true]}}},
            "s": {"type": "score", "instructions": "", "criteria": ["low", {"what": "mid"}, null]},
            "n": {"type": "noul", "instructions": null, "criteria": {"true": " yes! ", "false": null}},
            "z": {"type": "noul", "instructions": "Zero?", "criteria": {"true": "", "false": null}}
        }));
        let q = |id: &str| layout.question(id, qs.get(id).unwrap()).unwrap();
        assert_eq!(
            q("c"),
            (
                "{\"k\": \"\u{e9}\"}".to_owned(),
                vec![
                    "a".to_owned(),
                    "b".to_owned(),
                    "c: desc".to_owned(),
                    "e: {\"x\": [1, true]}".to_owned()
                ]
            )
        );
        assert_eq!(q("s").1, ["low", "{\"what\": \"mid\"}", ""]);
        assert_eq!(q("n").0, "n\nTrue if: yes!\nFalse if: \u{2014}");
        assert_eq!(
            q("z"),
            ("Zero?".to_owned(), vec!["yes".to_owned(), "no".to_owned()])
        );
    }

    #[test]
    fn builds_rows_like_model_py() {
        let layout = layout(512, 64);
        let qs = questions(json!({
            "q": {"type": "choice", "instructions": " Re\u{301}sume? ",
                  "criteria": {"a": "x", "b": null}}
        }));
        let q = layout
            .encode_question(
                &Chars,
                "q",
                qs.get("q").unwrap(),
                "{\"t\": \"Caf\u{e9}\", \"n\": 1.0}",
            )
            .unwrap();
        assert_eq!(
            read(&q.state_ids),
            "q: R\u{e9}sume? \n{\"t\": \"Caf\u{e9}\", \"n\": 1.0}"
        );
        assert_eq!(read(&q.option_ids[0]), "p: R\u{e9}sume?  a: x");
        assert_eq!(read(&q.option_ids[1]), "p: R\u{e9}sume?  b");
        assert!(!q.state_truncated);
    }

    #[test]
    fn cuts_long_rows() {
        let layout = layout(10, 6);
        let qs = questions(json!({
            "q": {"type": "choice", "instructions": "Q",
                  "criteria": {"long option": null, "b": null}}
        }));
        let q = layout
            .encode_question(&Chars, "q", qs.get("q").unwrap(), "abcdefghij")
            .unwrap();
        assert_eq!(
            (read(&q.state_ids).as_str(), q.state_truncated),
            ("q: Q\nabc", true)
        );
        assert_eq!(read(&q.option_ids[0]), "p: l");
        assert_eq!(read(&q.option_ids[1]), "p: b");
    }

    #[test]
    fn ordinal_choice_uses_the_ordinal_head() {
        let layout = layout(512, 64);
        let qs = questions(json!({
            "rank": {"type": "choice", "ordinal": true, "instructions": "How urgent?",
                     "criteria": {"low": null, "mid": null, "high": null}},
            "topic": {"type": "choice", "instructions": "Which?", "criteria": {"a": null, "b": null}}
        }));
        let rank = layout
            .encode_question(&Chars, "rank", qs.get("rank").unwrap(), "s")
            .unwrap();
        assert_eq!(rank.qtype, QuestionType::Score);
        let topic = layout
            .encode_question(&Chars, "topic", qs.get("topic").unwrap(), "s")
            .unwrap();
        assert_eq!(topic.qtype, QuestionType::Choice);

        let too_few = questions(json!({
            "q": {"type": "choice", "ordinal": true, "criteria": {"only": null}}
        }));
        assert!(matches!(
            layout.encode_question(&Chars, "q", too_few.get("q").unwrap(), "s"),
            Err(ModelError::Decision(DecisionError::Invalid { .. }))
        ));
        let criteria = (0..11)
            .map(|index| (format!("l{index}"), json!(null)))
            .collect::<serde_json::Map<_, _>>();
        let too_many = questions(json!({
            "q": {"type": "choice", "ordinal": true, "criteria": criteria}
        }));
        assert!(matches!(
            layout.encode_question(&Chars, "q", too_many.get("q").unwrap(), "s"),
            Err(ModelError::Decision(DecisionError::Invalid { .. }))
        ));
    }

    #[test]
    fn fits_options_like_render_py() {
        let options = [
            "billing: charges and refunds".to_owned(),
            "billing: charges and refunds, VAT".to_owned(),
            "ops".to_owned(),
        ];
        let mut layout = layout(512, 32);
        let v1 = layout
            .option_texts(&Chars, "Pick the team for this ticket now", &options)
            .unwrap();
        layout.config.choice_fit = true;
        let v2 = layout
            .option_texts(&Chars, "Pick the team for this ticket now", &options)
            .unwrap();
        let want = [
            "p: billing: charges and refunds",
            "p: billing: charges and refunds, VAT",
            "p: Pick the team for … ops",
        ];
        assert_eq!(v1, want);
        assert_eq!(v2, want);
    }

    #[test]
    fn ordinal_head_matches_upstream() {
        let scores = [-4.1834, -2.6412, -1.8191];
        let g = [0.1262, 0.0516, -0.752];
        let gaps = [2.6582, 3.2905, 2.3794];
        let log_probs = ordinal_log_probs(&scores, &g, &gaps, 0.935_556_835_803_263_9);
        let want = [-2.0909, -0.3950, -1.5956];
        for (value, expected) in log_probs.iter().zip(want) {
            assert!((value - expected).abs() < 1e-3, "{log_probs:?}");
        }
        let total: f64 = log_probs.iter().map(|value| value.exp()).sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn validates_the_config() {
        assert!(layout(512, 64).validate().is_ok());
        assert!(layout(2, 64).validate().is_err());
        let mut bad = layout(512, 64);
        bad.config.temperature = 0.0;
        assert!(bad.validate().is_err());
    }
}
