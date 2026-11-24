//! Laya: ModernBERT encoder + option-marker decision head.
//!
//! The layout and head mirror `rl_common.py`:
//!
//! ```text
//! [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
//! ```
//!
//! Fitted temperatures live in `rl_agent_config.json`; the manifest selects
//! `CalibrationMode::Raw` so the checkpoint's values apply unchanged.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::encoders::modernbert::{EncoderConfig, ModernBertEncoder, load_encoder_from_tensors};
use crate::families::batch::groups;
use crate::{Backbone, BackboneOutput, Family, FamilyFactory, LoadContext, ModelError};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{
    Embedding, LayerNorm, Linear, Module, VarBuilder, embedding, layer_norm, linear, ops::softmax,
};
use olaya_core::{
    Answer, Calibration, CalibrationFile, Capabilities, Content, DecisionError, Encoded, QType,
    Question, Readout, Renderer, State, TokenEncoder, answer_from_logits, dumps_content,
};
use serde::Deserialize;
use serde_json::Value;

const MAX_OPTION_TOKENS: usize = 48;
const MIN_HEAD_ROOM: i64 = 16;
const MIN_INSTRUCTION_TOKENS: usize = 8;
const MIN_TOKENS_PER_OPTION: usize = 4;
const LAYER_NORM_EPS: f64 = 1e-5;

fn default_max_len() -> usize {
    512
}

fn default_head_max_len() -> usize {
    192
}

fn default_head_layers() -> usize {
    2
}

#[derive(Clone, Debug, Deserialize)]
struct LayaConfig {
    #[serde(default = "default_max_len")]
    max_len: usize,
    #[serde(default = "default_head_max_len")]
    head_max_len: usize,
    #[serde(default = "default_head_layers")]
    head_layers: usize,
    #[serde(default)]
    temperature: Vec<Value>,
    #[serde(default)]
    temperature_by_options: HashMap<String, Value>,
}

// ---------------------------------------------------------------------------
// Rendering

#[derive(Clone, Copy, Debug)]
struct LayaSpecialTokens {
    cls: u32,
    sep: u32,
    mask: u32,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct LayaTokenizerConfig {
    #[serde(default)]
    cls_token: Option<String>,
    #[serde(default)]
    sep_token: Option<String>,
    #[serde(default)]
    mask_token: Option<String>,
}

impl LayaSpecialTokens {
    fn resolve(
        config: &EncoderConfig,
        tokenizer_config: &LayaTokenizerConfig,
        encoder: &dyn TokenEncoder,
    ) -> Result<Self, ModelError> {
        let resolve = |configured: Option<&str>, fallback: Option<u32>, legacy: &str| {
            configured
                .and_then(|token| encoder.token_id(token))
                .or(fallback)
                .or_else(|| encoder.token_id(legacy))
        };
        let cls = resolve(
            tokenizer_config.cls_token.as_deref(),
            config.cls_token_id,
            "[CLS]",
        )
        .ok_or_else(|| ModelError::config("tokenizer has no [CLS]/<bos> token"))?;
        let sep = resolve(
            tokenizer_config.sep_token.as_deref(),
            config.sep_token_id,
            "[SEP]",
        )
        .ok_or_else(|| ModelError::config("tokenizer has no [SEP]/<eos> token"))?;
        let mask = resolve(
            tokenizer_config.mask_token.as_deref(),
            config.mask_token_id,
            "[MASK]",
        )
        .ok_or_else(|| ModelError::config("tokenizer has no [MASK]/<mask> token"))?;
        Ok(Self { cls, sep, mask })
    }
}

fn read_tokenizer_config(path: &Path) -> Result<LayaTokenizerConfig, ModelError> {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map_err(|error| ModelError::config(format!("{}: {error}", path.display()))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(LayaTokenizerConfig::default())
        }
        Err(error) => Err(ModelError::Io(error)),
    }
}

struct LayaRenderer {
    max_len: usize,
    head_max_len: usize,
    special: LayaSpecialTokens,
}

impl Renderer for LayaRenderer {
    fn encode_state(
        &self,
        state: &State,
        encoder: &dyn TokenEncoder,
    ) -> Result<Vec<u32>, DecisionError> {
        let text = match state.as_content() {
            Content::String(text) => text.clone(),
            content => dumps_content(content),
        };
        encoder.encode(&sanitize(&text))
    }

    fn render(
        &self,
        state_ids: &[u32],
        question_id: &str,
        question: &Question,
        encoder: &dyn TokenEncoder,
    ) -> Result<Encoded, DecisionError> {
        let qtype = QType::of(question);
        if qtype == QType::Box {
            return Err(DecisionError::unsupported(
                "laya does not answer box questions",
            ));
        }

        let cls = self.special.cls;
        let sep = self.special.sep;
        let mask = self.special.mask;

        let instructions = question.instructions_or_id(question_id);
        let head_text = format!("{} question: {}", qtype.name(), sanitize(&instructions));
        let mut head_ids = encoder.encode(&head_text)?;

        let options = render_options(question);
        if options.is_empty() {
            return Err(DecisionError::invalid("question has no options"));
        }

        let mut option_ids = Vec::with_capacity(options.len());
        for option in &options {
            let mut ids = vec![mask];
            let text = format!(" {}", sanitize(option));
            let encoded = encoder.encode(&text)?;
            let take = encoded.len().min(MAX_OPTION_TOKENS);
            ids.extend_from_slice(&encoded[..take]);
            option_ids.push(ids);
        }

        let mut option_budget =
            self.head_max_len as i64 - option_ids.iter().map(|ids| ids.len()).sum::<usize>() as i64;
        if option_budget < MIN_HEAD_ROOM {
            let per = std::cmp::max(
                MIN_TOKENS_PER_OPTION as i64,
                (self.head_max_len as i64 - MIN_HEAD_ROOM) / option_ids.len().max(1) as i64,
            ) as usize;
            for ids in &mut option_ids {
                ids.truncate(per);
            }
            option_budget = self.head_max_len as i64
                - option_ids.iter().map(|ids| ids.len()).sum::<usize>() as i64;
        }
        head_ids.truncate(std::cmp::max(MIN_INSTRUCTION_TOKENS as i64, option_budget) as usize);

        let mut ids = Vec::with_capacity(self.max_len);
        ids.push(cls);
        ids.extend_from_slice(&head_ids);
        ids.push(sep);

        let mut markers = Vec::with_capacity(option_ids.len());
        for option in &option_ids {
            markers.push(ids.len());
            ids.extend_from_slice(option);
        }
        ids.push(sep);

        let room = self.max_len.saturating_sub(ids.len() + 1);
        let state_used = &state_ids[..state_ids.len().min(room)];
        ids.extend_from_slice(state_used);
        ids.push(sep);
        ids.truncate(self.max_len);

        let markers: Vec<usize> = markers
            .into_iter()
            .filter(|marker| *marker < ids.len())
            .collect();
        if markers.len() != options.len() {
            return Err(DecisionError::TooManyOptions {
                question: question_id.to_owned(),
                options: options.len(),
                limit: markers.len(),
            });
        }

        Ok(Encoded {
            ids,
            markers,
            qtype,
            state_tokens: state_used.len(),
            truncated: state_used.len() < state_ids.len(),
        })
    }
}

fn render_options(question: &Question) -> Vec<String> {
    match question {
        Question::Choice(choice) => choice
            .criteria
            .iter()
            .map(|(label, description)| {
                if is_empty(description) {
                    label.clone()
                } else {
                    format!("{label}: {}", render_content(description))
                }
            })
            .collect(),
        Question::Score(score) => score
            .criteria
            .iter()
            .enumerate()
            .map(|(index, level)| format!("level {index}: {}", render_content(level)))
            .collect(),
        Question::Noul(noul) => {
            let (is_true, is_false) = match &noul.criteria {
                Some(criteria) => (
                    criteria
                        .is_true
                        .as_ref()
                        .map(render_content)
                        .unwrap_or_else(|| "yes, the statement holds".to_owned()),
                    criteria
                        .is_false
                        .as_ref()
                        .map(render_content)
                        .unwrap_or_else(|| "no, the statement does not hold".to_owned()),
                ),
                None => (
                    "yes, the statement holds".to_owned(),
                    "no, the statement does not hold".to_owned(),
                ),
            };
            vec![format!("false: {is_false}"), format!("true: {is_true}")]
        }
        Question::Box(_) => Vec::new(),
    }
}

fn render_content(content: &Content) -> String {
    match content {
        Content::String(text) => text.clone(),
        other => dumps_content(other),
    }
}

fn is_empty(content: &Content) -> bool {
    match content {
        Content::Null => true,
        Content::String(text) => text.is_empty(),
        Content::Object(map) => map.is_empty(),
        Content::Array(items) => items.is_empty(),
    }
}

fn sanitize(text: &str) -> String {
    text.replace("[MASK]", " ").replace("<mask>", " ")
}

// ---------------------------------------------------------------------------
// Readout

struct LayaReadout;

impl Readout for LayaReadout {
    fn read(
        &self,
        question: &Question,
        logits: &[f32],
        calibration: &Calibration,
        state_tokens: usize,
    ) -> Result<Answer, DecisionError> {
        match question {
            Question::Choice(choice) => {
                let labels: Vec<String> = choice.criteria.keys().cloned().collect();
                answer_from_logits(question, &labels, &[], logits, calibration, state_tokens)
            }
            Question::Score(score) => answer_from_logits(
                question,
                &[],
                &score.criteria,
                logits,
                calibration,
                state_tokens,
            ),
            Question::Noul(_) => answer_from_logits(
                question,
                &["false".to_owned(), "true".to_owned()],
                &[],
                logits,
                calibration,
                state_tokens,
            ),
            Question::Box(_) => Err(DecisionError::unsupported(
                "laya does not answer box questions",
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Backbone

struct LayaBackbone {
    encoder: ModernBertEncoder,
    type_emb: Embedding,
    head: Vec<HeadLayer>,
    scorer: Scorer,
    act_head: ActHead,
    device: Device,
}

impl LayaBackbone {
    fn load(
        ctx: &LoadContext<'_>,
        head_layers: usize,
        encoder_config: &EncoderConfig,
    ) -> Result<Self, ModelError> {
        let weights = ctx.dir.join("model.safetensors");
        let tensors = candle_core::safetensors::load(&weights, ctx.device)?;
        let encoder =
            load_encoder_from_tensors(&tensors, encoder_config, ctx.device, ctx.compute_dtype())?;
        // The decision head is tiny, so it stays F32 even when the encoder runs
        // in half precision; this keeps the readout numerically identical.
        let vb = VarBuilder::from_tensors(tensors, DType::F32, ctx.device);
        let hidden = encoder_config.hidden_size;
        let num_heads = (hidden / 64).max(1);

        let type_emb = embedding(3, hidden, vb.pp("type_emb"))?;
        let mut head = Vec::with_capacity(head_layers);
        for index in 0..head_layers {
            head.push(HeadLayer::load(
                vb.pp(format!("head.layers.{index}")),
                hidden,
                num_heads,
            )?);
        }
        let scorer = Scorer::load(vb.pp("scorer"), hidden)?;
        let act_head = ActHead::load(vb.pp("act_head"), hidden)?;

        Ok(Self {
            encoder,
            type_emb,
            head,
            scorer,
            act_head,
            device: ctx.device.clone(),
        })
    }
}

const BATCH_TOKEN_BUDGET: usize = 32_768;

impl LayaBackbone {
    fn forward_rows(&self, rows: &[Encoded]) -> Result<Vec<BackboneOutput>, ModelError> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let lengths = rows.iter().map(|row| row.ids.len()).collect::<Vec<_>>();
        let mut outputs: Vec<Option<BackboneOutput>> = (0..rows.len()).map(|_| None).collect();
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
            let mut hidden = self.encoder.forward(&ids, &mask)?.to_dtype(DType::F32)?;
            let qtypes = Tensor::from_vec(
                batch
                    .iter()
                    .map(|row| row.qtype.index() as u32)
                    .collect::<Vec<u32>>(),
                batch.len(),
                &self.device,
            )?;
            hidden = hidden.broadcast_add(&self.type_emb.forward(&qtypes)?.unsqueeze(1)?)?;
            // The decision head has no causal constraint: pad keys are masked so
            // a row's real tokens see exactly their own row.
            let attention_mask = mask
                .affine(-1.0, 1.0)?
                .affine(f32::MIN as f64, 0.0)?
                .unsqueeze(1)?
                .unsqueeze(1)?;
            for layer in &self.head {
                hidden = layer.forward(&hidden, &attention_mask)?;
            }

            for (offset, row) in batch.iter().enumerate() {
                let row_hidden = hidden.i(offset)?;
                let markers = Tensor::from_vec(
                    row.markers
                        .iter()
                        .map(|marker| *marker as u32)
                        .collect::<Vec<u32>>(),
                    row.markers.len(),
                    &self.device,
                )?;
                let marker_states = row_hidden.index_select(&markers, 0)?.unsqueeze(0)?;
                let logits = self.scorer.forward(&marker_states)?;

                let probabilities = softmax(&logits, 1)?;
                let values = probabilities
                    .flatten_all()?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                let k = values.len().max(2) as f32;
                let entropy = -values
                    .iter()
                    .map(|probability| probability * probability.max(1e-9).ln())
                    .sum::<f32>()
                    / k.ln();
                let mut sorted = values;
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let top1 = sorted.first().copied().unwrap_or(0.0);
                let top2 = sorted.get(1).copied().unwrap_or(0.0);
                let features = Tensor::from_vec(
                    vec![top1, top1 - top2, entropy, k / 255.0],
                    (1, 4),
                    &self.device,
                )?;
                let pooled = row_hidden.i(0)?.unsqueeze(0)?;
                let act_input = Tensor::cat(&[pooled, features], 1)?;
                let act = self.act_head.forward(&act_input)?;
                outputs[range.start + offset] = Some(BackboneOutput {
                    logits,
                    act: Some(act),
                });
            }
        }
        outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| candle_core::Error::Msg("laya row was not scored".to_owned()))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(ModelError::from)
    }
}

impl Backbone for LayaBackbone {
    fn forward(&self, encoded: &Encoded) -> Result<BackboneOutput, ModelError> {
        self.forward_rows(std::slice::from_ref(encoded))?
            .pop()
            .ok_or_else(|| {
                ModelError::from(candle_core::Error::Msg(
                    "laya row was not scored".to_owned(),
                ))
            })
    }

    fn forward_batch(&self, batch: &[Encoded]) -> Result<Vec<BackboneOutput>, ModelError> {
        self.forward_rows(batch)
    }
}

struct HeadLayer {
    norm1: LayerNorm,
    in_proj: Linear,
    out_proj: Linear,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl HeadLayer {
    fn load(vb: VarBuilder, hidden: usize, num_heads: usize) -> Result<Self, ModelError> {
        let weight = vb.get((3 * hidden, hidden), "self_attn.in_proj_weight")?;
        let bias = vb.get(3 * hidden, "self_attn.in_proj_bias")?;
        Ok(Self {
            norm1: layer_norm(hidden, LAYER_NORM_EPS, vb.pp("norm1"))?,
            in_proj: Linear::new(weight, Some(bias)),
            out_proj: linear(hidden, hidden, vb.pp("self_attn.out_proj"))?,
            norm2: layer_norm(hidden, LAYER_NORM_EPS, vb.pp("norm2"))?,
            linear1: linear(hidden, 4 * hidden, vb.pp("linear1"))?,
            linear2: linear(4 * hidden, hidden, vb.pp("linear2"))?,
            num_heads,
            head_dim: hidden / num_heads,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: &Tensor) -> Result<Tensor, ModelError> {
        let residual = xs.clone();
        let normed = self.norm1.forward(xs)?;
        let attention = self.attention(&normed, attention_mask)?;
        let xs = (residual + attention)?;

        let residual = xs.clone();
        let normed = self.norm2.forward(&xs)?;
        let ffn = self.linear1.forward(&normed)?.relu()?;
        let ffn = self.linear2.forward(&ffn)?;
        Ok((residual + ffn)?)
    }

    fn attention(&self, xs: &Tensor, attention_mask: &Tensor) -> Result<Tensor, ModelError> {
        let (batch, len, hidden) = xs.dims3()?;
        let qkv = self.in_proj.forward(xs)?;
        let chunks = qkv.chunk(3, 2)?;
        let q = chunks[0]
            .reshape((batch, len, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let k = chunks[1]
            .reshape((batch, len, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let v = chunks[2]
            .reshape((batch, len, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attention = q
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .affine(scale, 0.0)?
            .broadcast_add(attention_mask)?;
        let attention = softmax(&attention, 3)?;
        let out = attention
            .matmul(&v)?
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((batch, len, hidden))?;
        Ok(self.out_proj.forward(&out)?)
    }
}

struct Scorer {
    norm: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

impl Scorer {
    fn load(vb: VarBuilder, hidden: usize) -> Result<Self, ModelError> {
        Ok(Self {
            norm: layer_norm(hidden, LAYER_NORM_EPS, vb.pp("0"))?,
            linear1: linear(hidden, hidden, vb.pp("1"))?,
            linear2: linear(hidden, 1, vb.pp("3"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, ModelError> {
        let xs = self.norm.forward(xs)?;
        let xs = self.linear1.forward(&xs)?.gelu_erf()?;
        let xs = self.linear2.forward(&xs)?;
        Ok(xs.squeeze(2)?)
    }
}

struct ActHead {
    linear1: Linear,
    linear2: Linear,
}

impl ActHead {
    fn load(vb: VarBuilder, hidden: usize) -> Result<Self, ModelError> {
        Ok(Self {
            linear1: linear(hidden + 4, 256, vb.pp("0"))?,
            linear2: linear(256, 2, vb.pp("2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, ModelError> {
        let xs = self.linear1.forward(xs)?.gelu_erf()?;
        Ok(self.linear2.forward(&xs)?)
    }
}

// ---------------------------------------------------------------------------
// Family

struct LayaFamily {
    tokenizer: Arc<dyn TokenEncoder>,
    renderer: LayaRenderer,
    backbone: LayaBackbone,
    readout: LayaReadout,
    calibration: Calibration,
}

impl Family for LayaFamily {
    fn id(&self) -> &'static str {
        "laya"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::classification()
    }

    fn tokenizer(&self) -> &dyn TokenEncoder {
        self.tokenizer.as_ref()
    }

    fn renderer(&self) -> Option<&dyn Renderer> {
        Some(&self.renderer)
    }

    fn backbone(&self) -> Option<&dyn Backbone> {
        Some(&self.backbone)
    }

    fn readout(&self) -> Option<&dyn Readout> {
        Some(&self.readout)
    }

    fn calibration(&self) -> &Calibration {
        &self.calibration
    }
}

/// Loads the laya checkpoint.
pub struct Factory;

impl FamilyFactory for Factory {
    fn family(&self) -> &'static str {
        "laya"
    }

    fn load(&self, ctx: &LoadContext<'_>) -> Result<Arc<dyn Family>, ModelError> {
        let config_path = ctx.dir.join("rl_agent_config.json");
        let config: LayaConfig = serde_json::from_slice(&std::fs::read(&config_path)?)
            .map_err(|error| ModelError::config(format!("{}: {error}", config_path.display())))?;
        // These values size per-request buffers; reject a malformed checkpoint
        // instead of attempting an enormous allocation.
        const MAX_LAYOUT_TOKENS: usize = 1_048_576;
        if config.max_len == 0 || config.max_len > MAX_LAYOUT_TOKENS {
            return Err(ModelError::config(format!(
                "laya max_len must be between 1 and {MAX_LAYOUT_TOKENS}"
            )));
        }
        if config.head_max_len == 0 || config.head_max_len > MAX_LAYOUT_TOKENS {
            return Err(ModelError::config(format!(
                "laya head_max_len must be between 1 and {MAX_LAYOUT_TOKENS}"
            )));
        }
        if config.head_layers == 0 || config.head_layers > 128 {
            return Err(ModelError::config(
                "laya head_layers must be between 1 and 128",
            ));
        }
        let calibration_file = CalibrationFile {
            temperature: config.temperature.clone(),
            temperature_by_options: config.temperature_by_options.clone(),
            ..CalibrationFile::default()
        };
        let calibration = Calibration::from_file(&calibration_file, ctx.manifest.calibration_mode);
        let encoder_config = EncoderConfig::from_file(&ctx.dir.join("encoder/config.json"))?;
        let tokenizer_config =
            read_tokenizer_config(&ctx.dir.join("tokenizer/tokenizer_config.json"))?;
        let special =
            LayaSpecialTokens::resolve(&encoder_config, &tokenizer_config, ctx.tokenizer.as_ref())?;
        let backbone = LayaBackbone::load(ctx, config.head_layers, &encoder_config)?;

        Ok(Arc::new(LayaFamily {
            tokenizer: ctx.tokenizer.clone(),
            renderer: LayaRenderer {
                max_len: config.max_len,
                head_max_len: config.head_max_len,
                special,
            },
            backbone,
            readout: LayaReadout,
            calibration,
        }))
    }
}
