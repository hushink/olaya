//! Qwen3.5 text trunk: hybrid Gated DeltaNet (linear attention) + gated
//! full attention.
//!
//! Layer order follows `transformers.models.qwen3_5`:
//!
//! * linear-attention layers run a depthwise causal conv plus the chunked
//!   gated-delta rule (64-token chunks, UT transform, inter-chunk scan);
//! * full-attention layers are GQA with an output gate and partial RoPE;
//! * every RMSNorm is zero-centered (`weight + 1`).
//!
//! The trunk is stateless and causal, so right padding is inert. Text-only
//! mRoPE reduces to plain partial RoPE on the first `rotary_dim` channels.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::attention::{AttnMask, flash_attn};
use candle_nn::ops::{sigmoid, silu, softmax_last_dim};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder, embedding, linear_b, linear_no_bias};
use serde::Deserialize;
use serde_json::Value;

use crate::ModelError;

const CHUNK: usize = 64;

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_hidden_act() -> String {
    "silu".to_owned()
}

fn default_true() -> bool {
    true
}

fn default_conv_kernel_dim() -> usize {
    4
}

fn default_linear_key_head_dim() -> usize {
    128
}

fn default_linear_value_head_dim() -> usize {
    128
}

fn default_linear_key_heads() -> usize {
    16
}

fn default_linear_value_heads() -> usize {
    16
}

fn default_full_attention_interval() -> usize {
    4
}

fn default_max_positions() -> usize {
    262_144
}

fn default_rope_theta() -> f64 {
    10_000_000.0
}

fn default_partial_rotary_factor() -> f64 {
    1.0
}

#[derive(Clone, Debug, Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
}

impl Default for RopeParameters {
    fn default() -> Self {
        Self {
            rope_theta: default_rope_theta(),
            partial_rotary_factor: default_partial_rotary_factor(),
        }
    }
}

/// The text half of a Qwen3.5 `config.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct Qwen35Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attn_output_gate: bool,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_conv_kernel_dim")]
    pub linear_conv_kernel_dim: usize,
    #[serde(default = "default_linear_key_head_dim")]
    pub linear_key_head_dim: usize,
    #[serde(default = "default_linear_value_head_dim")]
    pub linear_value_head_dim: usize,
    #[serde(default = "default_linear_key_heads")]
    pub linear_num_key_heads: usize,
    #[serde(default = "default_linear_value_heads")]
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: usize,
    #[serde(default = "default_max_positions")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub rope_parameters: RopeParameters,
}

impl Qwen35Config {
    /// Parse a text config, or the `text_config` object of a multimodal one.
    pub fn from_value(value: Value) -> std::result::Result<Self, ModelError> {
        // Some multimodal releases put `tie_word_embeddings` only at the root;
        // the text config inherits it.
        let root_tie = value.get("tie_word_embeddings").cloned();
        let mut text = value.get("text_config").cloned().unwrap_or(value);
        if let (Some(map), Some(root_tie)) = (text.as_object_mut(), root_tie)
            && !map.contains_key("tie_word_embeddings")
        {
            map.insert("tie_word_embeddings".to_owned(), root_tie);
        }
        serde_json::from_value(text)
            .map_err(|error| ModelError::config(format!("Qwen3.5 config: {error}")))
    }

    pub fn from_file(path: &Path) -> std::result::Result<Self, ModelError> {
        let bytes = std::fs::read(path)?;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|error| ModelError::config(format!("{}: {error}", path.display())))?;
        Self::from_value(value)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn rotary_dim(&self) -> usize {
        ((self.head_dim() as f64) * self.rope_parameters.partial_rotary_factor) as usize
    }

    pub fn is_full_attention(&self, layer: usize) -> bool {
        match &self.layer_types {
            Some(types) => types[layer] == "full_attention",
            None => (layer + 1).is_multiple_of(self.full_attention_interval),
        }
    }

    pub fn validate(&self) -> std::result::Result<(), ModelError> {
        if self.vocab_size == 0
            || self.hidden_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.intermediate_size == 0
        {
            return Err(ModelError::config(
                "Qwen3.5 dimensions and layer counts must be positive",
            ));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(ModelError::config(
                "Qwen3.5 num_attention_heads is not a multiple of num_key_value_heads",
            ));
        }
        let head_dim = self.head_dim();
        let rotary = self.rotary_dim();
        if head_dim == 0 || rotary == 0 || rotary > head_dim || !rotary.is_multiple_of(2) {
            return Err(ModelError::config(format!(
                "Qwen3.5 rotary_dim {rotary} must be even and at most head_dim {head_dim}"
            )));
        }
        if self.linear_conv_kernel_dim == 0
            || self.linear_key_head_dim == 0
            || self.linear_value_head_dim == 0
            || self.linear_num_key_heads == 0
            || self.linear_num_value_heads == 0
        {
            return Err(ModelError::config(
                "Qwen3.5 linear-attention sizes must be positive",
            ));
        }
        if !self
            .linear_num_value_heads
            .is_multiple_of(self.linear_num_key_heads)
        {
            return Err(ModelError::config(
                "Qwen3.5 linear_num_value_heads is not a multiple of linear_num_key_heads",
            ));
        }
        if self.hidden_act != "silu" {
            return Err(ModelError::config(format!(
                "unsupported Qwen3.5 hidden_act {:?}; expected silu",
                self.hidden_act
            )));
        }
        if let Some(types) = &self.layer_types {
            if types.len() != self.num_hidden_layers {
                return Err(ModelError::config(format!(
                    "Qwen3.5 layer_types has {} entries for {} layers",
                    types.len(),
                    self.num_hidden_layers
                )));
            }
            for kind in types {
                if kind != "linear_attention" && kind != "full_attention" {
                    return Err(ModelError::config(format!(
                        "unsupported Qwen3.5 layer type {kind:?}"
                    )));
                }
            }
        }
        if self.max_position_embeddings == 0 {
            return Err(ModelError::config(
                "Qwen3.5 max_position_embeddings is zero",
            ));
        }
        Ok(())
    }
}

fn prefix_key(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_owned()
    } else {
        format!("{prefix}.{name}")
    }
}

fn detect_prefix(tensors: &HashMap<String, Tensor>) -> &'static str {
    for prefix in ["model.language_model", "model", ""] {
        if tensors.contains_key(&prefix_key(prefix, "embed_tokens.weight")) {
            return prefix;
        }
    }
    ""
}

fn zero_centered_norm(dim: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let weight = vb.get(dim, "weight")?;
    let weight = (weight + 1.0)?;
    Ok(RmsNorm::new(weight, eps))
}

struct Rotary {
    cos: Tensor,
    sin: Tensor,
    dim: usize,
}

impl Rotary {
    fn new(config: &Qwen35Config, dtype: DType, device: &Device) -> Result<Self> {
        let dim = config.rotary_dim();
        let max = config.max_position_embeddings;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|index| {
                1f32 / config
                    .rope_parameters
                    .rope_theta
                    .powf(index as f64 / dim as f64) as f32
            })
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, dim / 2), device)?.to_dtype(DType::F32)?;
        let positions = Tensor::arange(0u32, max as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max, 1))?;
        let freqs = positions.matmul(&inv_freq)?;
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
            dim,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq = q.dim(2)?;
        let head_dim = q.dim(3)?;
        let cos = self.cos.narrow(0, 0, seq)?;
        let sin = self.sin.narrow(0, 0, seq)?;
        let rotate = |x: &Tensor| -> Result<Tensor> {
            if self.dim == head_dim {
                return candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin);
            }
            let rotated = x.narrow(3, 0, self.dim)?;
            let passed = x.narrow(3, self.dim, head_dim - self.dim)?;
            let rotated = candle_nn::rotary_emb::rope(&rotated.contiguous()?, &cos, &sin)?;
            Tensor::cat(&[rotated, passed], 3)
        };
        Ok((rotate(q)?, rotate(k)?))
    }
}

fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (batch, heads, seq, dim) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((batch, heads, groups, seq, dim))?
        .reshape((batch, heads * groups, seq, dim))
}

fn repeat_heads(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (batch, seq, heads, dim) = x.dims4()?;
    x.unsqueeze(3)?
        .broadcast_as((batch, seq, heads, groups, dim))?
        .reshape((batch, seq, heads * groups, dim))
}

/// Score elements (`batch * heads * chunk * seq`) held by one GPU attention
/// chunk. 8M f32 is 32 MiB, small enough for a 12 GiB card at 16k contexts.
const ATTENTION_SCORE_BUDGET: usize = 8 * 1024 * 1024;

fn causal_attention(q: &Tensor, k: &Tensor, v: &Tensor, device: &Device) -> Result<Tensor> {
    let head_dim = q.dim(3)?;
    let scale = 1.0 / (head_dim as f32).sqrt();
    if device.is_cpu() {
        // CPU flash takes (B, S, H, D) and returns (B, H, S, D).
        flash_attn::<f32>(
            &q.transpose(1, 2)?.contiguous()?,
            &k.transpose(1, 2)?.contiguous()?,
            &v.transpose(1, 2)?.contiguous()?,
            scale,
            AttnMask::causal(),
            None,
            None,
        )
    } else {
        causal_attention_windowed(q, k, v, scale, ATTENTION_SCORE_BUDGET)
    }
}

/// Materialize attention one query chunk at a time. Softmax still runs over
/// the full key sequence, so this is exact; only the `[chunk, seq]` score
/// matrix is bounded instead of `[seq, seq]`.
fn causal_attention_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    budget: usize,
) -> Result<Tensor> {
    let (batch, heads, seq, _) = q.dims4()?;
    let key = k.transpose(2, 3)?.contiguous()?;
    let rows = budget.max(1) / (batch * heads * seq).max(1);
    let chunk = rows.clamp(1, seq);
    let mut contexts = Vec::with_capacity(seq.div_ceil(chunk));
    let mut start = 0;
    while start < seq {
        let length = chunk.min(seq - start);
        let scores = (q.narrow(2, start, length)?.contiguous()?.matmul(&key)? * scale as f64)?;
        let mask = causal_window_mask(seq, start, length, scores.dtype(), scores.device())?;
        let probabilities = softmax_last_dim(&scores.broadcast_add(&mask)?)?;
        contexts.push(probabilities.matmul(v)?);
        start += length;
    }
    Tensor::cat(&contexts, 2)
}

#[cfg(test)]
fn causal_mask(seq: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    causal_window_mask(seq, 0, seq, dtype, device)
}

/// `[1, 1, rows, seq]` mask for query rows `start..start + rows`. Built on the
/// target device so a long-context pass does not copy a mask per chunk.
fn causal_window_mask(
    seq: usize,
    start: usize,
    rows: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let query_rows = Tensor::arange(start as u32, (start + rows) as u32, device)?.unsqueeze(1)?;
    let key_columns = Tensor::arange(0u32, seq as u32, device)?.unsqueeze(0)?;
    let allowed = key_columns.broadcast_le(&query_rows)?;
    let zeros = Tensor::zeros((rows, seq), dtype, device)?;
    let masked = Tensor::full(f32::NEG_INFINITY, (rows, seq), device)?.to_dtype(dtype)?;
    allowed
        .where_cond(&zeros, &masked)?
        .unsqueeze(0)?
        .unsqueeze(0)
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    let positive = x.clamp(0f64, f64::MAX)?;
    let complement = ((x.abs()? * -1.0)?.exp()? + 1.0)?.log()?;
    positive + complement
}

struct FullAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    gated: bool,
    device: Device,
}

impl FullAttention {
    fn load(config: &Qwen35Config, vb: VarBuilder) -> Result<Self> {
        let hidden = config.hidden_size;
        let heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim();
        let gated = config.attn_output_gate;
        let projection = |input: usize, output: usize, name: &str| {
            if config.attention_bias {
                linear_b(input, output, true, vb.pp(name))
            } else {
                linear_no_bias(input, output, vb.pp(name))
            }
        };
        Ok(Self {
            q_proj: projection(
                hidden,
                heads * head_dim * if gated { 2 } else { 1 },
                "q_proj",
            )?,
            k_proj: projection(hidden, kv_heads * head_dim, "k_proj")?,
            v_proj: projection(hidden, kv_heads * head_dim, "v_proj")?,
            o_proj: projection(heads * head_dim, hidden, "o_proj")?,
            q_norm: zero_centered_norm(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: zero_centered_norm(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?,
            heads,
            kv_heads,
            head_dim,
            gated,
            device: vb.device().clone(),
        })
    }

    fn forward(&self, x: &Tensor, rotary: &Rotary) -> Result<Tensor> {
        let (batch, seq, _) = x.dims3()?;
        let projected = self.q_proj.forward(x)?;
        let (query, gate) = if self.gated {
            let shaped = projected.reshape((batch, seq, self.heads, self.head_dim * 2))?;
            (
                shaped.narrow(3, 0, self.head_dim)?,
                Some(shaped.narrow(3, self.head_dim, self.head_dim)?.reshape((
                    batch,
                    seq,
                    self.heads * self.head_dim,
                ))?),
            )
        } else {
            (
                projected.reshape((batch, seq, self.heads, self.head_dim))?,
                None,
            )
        };
        let query = self.q_norm.forward(&query.transpose(1, 2)?.contiguous()?)?;
        let key = self.k_norm.forward(
            &self
                .k_proj
                .forward(x)?
                .reshape((batch, seq, self.kv_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?,
        )?;
        let value = self
            .v_proj
            .forward(x)?
            .reshape((batch, seq, self.kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let (query, key) = rotary.apply(&query, &key)?;
        let groups = self.heads / self.kv_heads;
        let key = repeat_kv(&key, groups)?;
        let value = repeat_kv(&value, groups)?;
        let context = causal_attention(&query, &key, &value, &self.device)?;
        let context = context.transpose(1, 2)?.contiguous()?.reshape((
            batch,
            seq,
            self.heads * self.head_dim,
        ))?;
        let context = match gate {
            Some(gate) => (context * sigmoid(&gate)?)?,
            None => context,
        };
        self.o_proj.forward(&context)
    }
}

struct DeltaMasks {
    lower: Tensor,
    pairwise: Tensor,
    levels: Vec<Tensor>,
}

impl DeltaMasks {
    fn new(device: &Device) -> Result<Self> {
        let mut lower = Vec::with_capacity(CHUNK * CHUNK);
        let mut pairwise = Vec::with_capacity(CHUNK * CHUNK);
        for row in 0..CHUNK {
            for column in 0..CHUNK {
                lower.push(if column < row { 1.0f32 } else { 0.0 });
                pairwise.push(if column <= row {
                    0.0f32
                } else {
                    f32::NEG_INFINITY
                });
            }
        }
        let mut masks = Vec::new();
        let mut step = 1;
        while step < CHUNK {
            let mut values = Vec::with_capacity(CHUNK * CHUNK);
            for row in 0..CHUNK {
                for column in 0..CHUNK {
                    let same = row / (2 * step) == column / (2 * step);
                    let lower_left = row % (2 * step) >= step && column % (2 * step) < step;
                    values.push(if same && lower_left { 1.0f32 } else { 0.0 });
                }
            }
            masks.push(Tensor::from_vec(values, (CHUNK, CHUNK), device)?);
            step *= 2;
        }
        Ok(Self {
            lower: Tensor::from_vec(lower, (CHUNK, CHUNK), device)?,
            pairwise: Tensor::from_vec(pairwise, (CHUNK, CHUNK), device)?,
            levels: masks,
        })
    }
}

/// `(I + A)^-1` for a strictly lower-triangular `A`, by block doubling.
fn unit_lower_inverse(strict_lower: &Tensor, masks: &DeltaMasks) -> Result<Tensor> {
    let device = strict_lower.device();
    let mut inverse = Tensor::eye(CHUNK, DType::F32, device)?.broadcast_as(strict_lower.dims())?;
    for mask in &masks.levels {
        let masked = strict_lower.broadcast_mul(mask)?.contiguous()?;
        let term = inverse.matmul(&masked.matmul(&inverse)?)?;
        inverse = (inverse - term)?;
    }
    Ok(inverse)
}

fn pad_zeros(x: &Tensor, axis: usize, amount: usize) -> Result<Tensor> {
    if amount == 0 {
        return Ok(x.clone());
    }
    let mut dims = x.dims().to_vec();
    dims[axis] = amount;
    let zeros = Tensor::zeros(dims, x.dtype(), x.device())?;
    Tensor::cat(&[x, &zeros], axis)
}

/// The chunked gated-delta rule (`torch_chunk_gated_delta_rule` with qk L2
/// normalization, no initial state, no final state).
fn gated_delta_rule(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    masks: &DeltaMasks,
) -> Result<Tensor> {
    let (batch, seq, heads, key_dim) = key.dims4()?;
    let value_dim = value.dim(3)?;
    let padded = seq.div_ceil(CHUNK) * CHUNK;
    let pad = padded - seq;
    // candle's CPU matmul handles at most one leading batch dimension, so the
    // (batch, head) pair is flattened for the chunk phase and the scan.
    let batch_heads = batch * heads;
    let query = pad_zeros(query, 1, pad)?
        .transpose(1, 2)?
        .to_dtype(DType::F32)?
        .reshape((batch_heads, padded, key_dim))?;
    let key = pad_zeros(key, 1, pad)?
        .transpose(1, 2)?
        .to_dtype(DType::F32)?
        .reshape((batch_heads, padded, key_dim))?;
    let value = pad_zeros(value, 1, pad)?
        .transpose(1, 2)?
        .to_dtype(DType::F32)?
        .reshape((batch_heads, padded, value_dim))?;
    let beta = pad_zeros(beta, 1, pad)?
        .transpose(1, 2)?
        .to_dtype(DType::F32)?
        .reshape((batch_heads, padded))?;
    let g = pad_zeros(g, 1, pad)?
        .transpose(1, 2)?
        .to_dtype(DType::F32)?
        .reshape((batch_heads, padded))?;

    let normalize = |x: &Tensor| -> Result<Tensor> {
        let norm = (x.sqr()?.sum_keepdim(2)? + 1e-6)?.sqrt()?;
        x.broadcast_div(&norm)
    };
    let query = (normalize(&query)? * ((key_dim as f64).powf(-0.5)))?.contiguous()?;
    let key = normalize(&key)?.contiguous()?;

    let chunks = padded / CHUNK;
    let query = query.reshape((batch_heads, chunks, CHUNK, key_dim))?;
    let key = key.reshape((batch_heads, chunks, CHUNK, key_dim))?;
    let value = value.reshape((batch_heads, chunks, CHUNK, value_dim))?;
    let beta = beta.reshape((batch_heads, chunks, CHUNK))?;
    let g = g.reshape((batch_heads, chunks, CHUNK))?;

    let beta_expanded = beta.unsqueeze(3)?;
    let value_beta = value.broadcast_mul(&beta_expanded)?.contiguous()?;
    let key_beta = key.broadcast_mul(&beta_expanded)?.contiguous()?;

    let cumulative = g.cumsum(2)?;
    let pair = cumulative
        .unsqueeze(3)?
        .broadcast_sub(&cumulative.unsqueeze(2)?)?
        .broadcast_add(&masks.pairwise)?
        .exp()?;
    let ut = key_beta
        .matmul(&key.transpose(2, 3)?.contiguous()?)?
        .broadcast_mul(&pair)?
        .contiguous()?;
    let strict_lower = ut.broadcast_mul(&masks.lower)?.contiguous()?;
    let inverse = unit_lower_inverse(&strict_lower, masks)?;
    let intra = query
        .matmul(&key.transpose(2, 3)?.contiguous()?)?
        .broadcast_mul(&pair)?
        .contiguous()?;
    let decayed_key_beta = key_beta
        .broadcast_mul(&cumulative.exp()?.unsqueeze(3)?)?
        .contiguous()?;
    let new_values = inverse.matmul(&value_beta)?;
    let key_cumdecay = inverse.matmul(&decayed_key_beta)?;
    let query = query
        .broadcast_mul(&cumulative.exp()?.unsqueeze(3)?)?
        .contiguous()?;
    let key = key
        .broadcast_mul(
            &cumulative
                .narrow(2, CHUNK - 1, 1)?
                .broadcast_sub(&cumulative)?
                .exp()?
                .unsqueeze(3)?,
        )?
        .contiguous()?;
    let chunk_decay = cumulative.narrow(2, CHUNK - 1, 1)?.exp()?.squeeze(2)?;

    let mut state = Tensor::zeros(
        (batch_heads, key_dim, value_dim),
        DType::F32,
        value.device(),
    )?;
    let mut outputs = Vec::with_capacity(chunks);
    for index in 0..chunks {
        let new_values = new_values.narrow(1, index, 1)?.squeeze(1)?;
        let key_cumdecay = key_cumdecay.narrow(1, index, 1)?.squeeze(1)?;
        let query = query.narrow(1, index, 1)?.squeeze(1)?;
        let intra = intra.narrow(1, index, 1)?.squeeze(1)?;
        let key = key.narrow(1, index, 1)?.squeeze(1)?;
        let decay = chunk_decay.narrow(1, index, 1)?.squeeze(1)?;
        let prediction = key_cumdecay.matmul(&state)?;
        let v_new = (new_values - prediction)?;
        let out = (query.matmul(&state)? + intra.matmul(&v_new)?)?;
        state = (state.broadcast_mul(&decay.unsqueeze(1)?.unsqueeze(2)?)?
            + key.transpose(1, 2)?.contiguous()?.matmul(&v_new)?)?;
        outputs.push(out);
    }
    let output = Tensor::stack(&outputs, 1)?;
    let output = output
        .reshape((batch_heads, padded, value_dim))?
        .narrow(1, 0, seq)?;
    output
        .reshape((batch, heads, seq, value_dim))?
        .transpose(1, 2)?
        .contiguous()
}

struct GatedDeltaNet {
    in_qkv: Linear,
    in_z: Linear,
    in_b: Linear,
    in_a: Linear,
    conv_weight: Tensor,
    out_proj: Linear,
    norm_weight: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    value_heads: usize,
    value_head_dim: usize,
    key_head_dim: usize,
    key_heads: usize,
    value_dim: usize,
    conv_kernel: usize,
    eps: f64,
    masks: std::sync::Arc<DeltaMasks>,
    device: Device,
}

impl GatedDeltaNet {
    fn load(config: &Qwen35Config, vb: VarBuilder) -> Result<Self> {
        let key_heads = config.linear_num_key_heads;
        let value_heads = config.linear_num_value_heads;
        let key_head_dim = config.linear_key_head_dim;
        let value_head_dim = config.linear_value_head_dim;
        let key_dim = key_heads * key_head_dim;
        let value_dim = value_heads * value_head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        Ok(Self {
            in_qkv: linear_no_bias(config.hidden_size, conv_dim, vb.pp("in_proj_qkv"))?,
            in_z: linear_no_bias(config.hidden_size, value_dim, vb.pp("in_proj_z"))?,
            in_b: linear_no_bias(config.hidden_size, value_heads, vb.pp("in_proj_b"))?,
            in_a: linear_no_bias(config.hidden_size, value_heads, vb.pp("in_proj_a"))?,
            conv_weight: vb.get(
                (conv_dim, 1, config.linear_conv_kernel_dim),
                "conv1d.weight",
            )?,
            out_proj: linear_no_bias(value_dim, config.hidden_size, vb.pp("out_proj"))?,
            norm_weight: vb.get(value_head_dim, "norm.weight")?,
            a_log: vb.get(value_heads, "A_log")?,
            dt_bias: vb.get(value_heads, "dt_bias")?,
            value_heads,
            value_head_dim,
            key_head_dim,
            key_heads,
            value_dim,
            conv_kernel: config.linear_conv_kernel_dim,
            eps: config.rms_norm_eps,
            masks: std::sync::Arc::new(DeltaMasks::new(vb.device())?),
            device: vb.device().clone(),
        })
    }

    fn causal_conv(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq, channels) = x.dims3()?;
        let x = x.transpose(1, 2)?.contiguous()?;
        let padding = Tensor::zeros(
            (batch, channels, self.conv_kernel - 1),
            x.dtype(),
            &self.device,
        )?;
        let padded = Tensor::cat(&[&padding, &x], 2)?;
        let mut out = Tensor::zeros((batch, channels, seq), x.dtype(), &self.device)?;
        for index in 0..self.conv_kernel {
            let slice = padded.narrow(2, index, seq)?;
            let weight = self
                .conv_weight
                .narrow(2, index, 1)?
                .squeeze(2)?
                .reshape((1, channels, 1))?;
            out = (out + slice.broadcast_mul(&weight)?)?;
        }
        silu(&out.transpose(1, 2)?.contiguous()?)
    }

    fn gated_norm(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(1)?;
        let normalized = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let weight = self.norm_weight.reshape((1, self.value_head_dim))?;
        let normalized = normalized.broadcast_mul(&weight)?;
        normalized.mul(&silu(&gate.to_dtype(DType::F32)?)?)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq, _) = x.dims3()?;
        let mixed = self.causal_conv(&self.in_qkv.forward(x)?)?;
        let z =
            self.in_z
                .forward(x)?
                .reshape((batch, seq, self.value_heads, self.value_head_dim))?;
        let beta = sigmoid(&self.in_b.forward(x)?)?;
        let a = self.in_a.forward(x)?;
        let a_log = self
            .a_log
            .to_dtype(DType::F32)?
            .reshape((1, 1, self.value_heads))?;
        let dt_bias = self
            .dt_bias
            .to_dtype(DType::F32)?
            .reshape((1, 1, self.value_heads))?;
        let decay = (softplus(&a.broadcast_add(&dt_bias)?)?.broadcast_mul(&a_log.exp()?)? * -1.0)?;

        let key_dim = self.key_heads * self.key_head_dim;
        let query = mixed.narrow(2, 0, key_dim)?.reshape((
            batch,
            seq,
            self.key_heads,
            self.key_head_dim,
        ))?;
        let key = mixed.narrow(2, key_dim, key_dim)?.reshape((
            batch,
            seq,
            self.key_heads,
            self.key_head_dim,
        ))?;
        let value = mixed.narrow(2, 2 * key_dim, self.value_dim)?.reshape((
            batch,
            seq,
            self.value_heads,
            self.value_head_dim,
        ))?;
        let groups = self.value_heads / self.key_heads;
        let query = repeat_heads(&query, groups)?;
        let key = repeat_heads(&key, groups)?;

        let core = gated_delta_rule(&query, &key, &value, &decay, &beta, &self.masks)?;
        let core = core.reshape((batch, seq, self.value_dim))?;
        let z = z.reshape((batch, seq, self.value_dim))?;
        let tokens = batch * seq;
        let core = core.reshape((tokens, self.value_heads, self.value_head_dim))?;
        let z = z.reshape((tokens, self.value_heads, self.value_head_dim))?;
        let normalized = self.gated_norm(
            &core.reshape((tokens * self.value_heads, self.value_head_dim))?,
            &z.reshape((tokens * self.value_heads, self.value_head_dim))?,
        )?;
        self.out_proj
            .forward(&normalized.reshape((batch, seq, self.value_dim))?)
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn load(config: &Qwen35Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: linear_no_bias(
                config.hidden_size,
                config.intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up: linear_no_bias(
                config.hidden_size,
                config.intermediate_size,
                vb.pp("up_proj"),
            )?,
            down: linear_no_bias(
                config.intermediate_size,
                config.hidden_size,
                vb.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        (silu(&self.gate.forward(x)?)? * self.up.forward(x)?)?.apply(&self.down)
    }
}

enum Mixer {
    Linear(GatedDeltaNet),
    Full(FullAttention),
}

struct Layer {
    input_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    mixer: Mixer,
    mlp: Mlp,
}

impl Layer {
    fn load(config: &Qwen35Config, layer: usize, vb: VarBuilder) -> Result<Self> {
        let mixer = if config.is_full_attention(layer) {
            Mixer::Full(FullAttention::load(config, vb.pp("self_attn"))?)
        } else {
            Mixer::Linear(GatedDeltaNet::load(config, vb.pp("linear_attn"))?)
        };
        Ok(Self {
            input_norm: zero_centered_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_norm: zero_centered_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mixer,
            mlp: Mlp::load(config, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor, rotary: &Rotary) -> Result<Tensor> {
        let h = self.input_norm.forward(x)?;
        let h = match &self.mixer {
            Mixer::Linear(linear) => linear.forward(&h)?,
            Mixer::Full(attention) => attention.forward(&h, rotary)?,
        };
        let x = (x + h)?;
        let h = self.post_attention_norm.forward(&x)?;
        x + self.mlp.forward(&h)
    }
}

/// A stateless Qwen3.5 text trunk plus its (optional) untied LM head.
pub struct Qwen35Encoder {
    embed_tokens: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Option<Linear>,
    rotary: Rotary,
    device: Device,
}

impl Qwen35Encoder {
    pub fn load(
        config: &Qwen35Config,
        mut tensors: HashMap<String, Tensor>,
        device: &Device,
    ) -> std::result::Result<Self, ModelError> {
        config.validate()?;
        let prefix = detect_prefix(&tensors);
        if !prefix.is_empty() {
            // Multimodal checkpoints carry a vision tower and MTP tensors that
            // the text trunk never reads; keep only the language model.
            let language = format!("{prefix}.");
            tensors.retain(|key, _| key.starts_with(&language) || key.starts_with("lm_head."));
        }
        for tensor in tensors.values_mut() {
            if tensor.dtype() != DType::F32 {
                *tensor = tensor.to_dtype(DType::F32)?;
            }
        }
        let root = VarBuilder::from_tensors(tensors.clone(), DType::F32, device);
        let vb = if prefix.is_empty() {
            root
        } else {
            root.pp(prefix)
        };
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| Layer::load(config, index, vb.pp(format!("layers.{index}"))))
            .collect::<Result<Vec<_>>>()?;
        let norm = zero_centered_norm(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            let weight = tensors
                .get(&prefix_key(prefix, "lm_head.weight"))
                .or_else(|| tensors.get("lm_head.weight"))
                .cloned()
                .ok_or_else(|| ModelError::config("Qwen3.5 has no lm_head.weight"))?;
            Some(Linear::new(weight, None))
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary: Rotary::new(config, DType::F32, device)?,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Hidden states for a right-padded batch, `[batch, seq, hidden]`.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let mut hidden = self.embed_tokens.forward(ids)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &self.rotary)?;
        }
        self.norm.forward(&hidden)
    }

    /// Logits of `hidden` (`[batch, hidden]`) over a candidate token subset.
    pub fn candidate_logits(&self, hidden: &Tensor, candidates: &[u32]) -> Result<Tensor> {
        let weights = match &self.lm_head {
            Some(head) => head.weight().clone(),
            None => self.embed_tokens.embeddings().clone(),
        };
        let indices = Tensor::from_vec(candidates.to_vec(), candidates.len(), &self.device)?;
        let rows = weights.index_select(&indices, 0)?.to_dtype(DType::F32)?;
        hidden
            .to_dtype(DType::F32)?
            .matmul(&rows.transpose(0, 1)?.contiguous()?)
    }
}

fn layer_prefix(prefix: &str, index: usize) -> String {
    if prefix.is_empty() {
        format!("layers.{index}.")
    } else {
        format!("{prefix}.layers.{index}.")
    }
}

#[derive(Clone, Debug)]
struct TensorInfo {
    file: usize,
    start: u64,
    end: u64,
    shape: Vec<usize>,
    dtype: DType,
}

/// Sharded safetensors read tensor by tensor, without mapping or loading whole
/// shards. The header is parsed once; each `load` seeks to one tensor's byte
/// range, widens BF16/F32 to fp32, and returns an owned tensor.
struct ShardedSafetensors {
    files: Vec<Mutex<File>>,
    data_start: Vec<u64>,
    tensors: HashMap<String, TensorInfo>,
}

#[derive(Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

impl ShardedSafetensors {
    fn open(paths: &[PathBuf]) -> std::result::Result<Self, ModelError> {
        let mut files = Vec::with_capacity(paths.len());
        let mut data_start = Vec::with_capacity(paths.len());
        let mut tensors = HashMap::new();
        for (index, path) in paths.iter().enumerate() {
            let mut file = File::open(path)?;
            let file_len = file.metadata()?.len();
            let mut length_bytes = [0u8; 8];
            file.read_exact(&mut length_bytes)?;
            let header_len = u64::from_le_bytes(length_bytes);
            if header_len > file_len.saturating_sub(8) {
                return Err(ModelError::config(format!(
                    "{}: safetensors header is longer than the file",
                    path.display()
                )));
            }
            let mut header = vec![0u8; header_len as usize];
            file.read_exact(&mut header)?;
            let entries: HashMap<String, serde_json::Value> = serde_json::from_slice(&header)
                .map_err(|error| {
                    ModelError::config(format!("{}: safetensors header: {error}", path.display()))
                })?;
            for (name, value) in entries {
                if name == "__metadata__" {
                    continue;
                }
                let raw: RawTensor = serde_json::from_value(value).map_err(|error| {
                    ModelError::config(format!("{}: tensor {name:?}: {error}", path.display()))
                })?;
                let dtype = match raw.dtype.as_str() {
                    "F32" => DType::F32,
                    "BF16" => DType::BF16,
                    other => {
                        return Err(ModelError::config(format!(
                            "{}: tensor {name:?} has unsupported dtype {other}",
                            path.display()
                        )));
                    }
                };
                let expected = raw.shape.iter().product::<usize>();
                let bytes = raw.data_offsets[1].saturating_sub(raw.data_offsets[0]);
                if bytes != (expected * dtype.size_in_bytes()) as u64 {
                    return Err(ModelError::config(format!(
                        "{}: tensor {name:?} has {} bytes for {expected} values",
                        path.display(),
                        bytes
                    )));
                }
                if raw.data_offsets[1] > file_len.saturating_sub(8 + header_len) {
                    return Err(ModelError::config(format!(
                        "{}: tensor {name:?} is out of bounds",
                        path.display()
                    )));
                }
                tensors.insert(
                    name,
                    TensorInfo {
                        file: index,
                        start: raw.data_offsets[0],
                        end: raw.data_offsets[1],
                        shape: raw.shape,
                        dtype,
                    },
                );
            }
            files.push(Mutex::new(file));
            data_start.push(8 + header_len);
        }
        Ok(Self {
            files,
            data_start,
            tensors,
        })
    }

    fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    fn load(&self, name: &str, device: &Device) -> Result<Tensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| candle_core::Error::CannotFindTensor {
                path: name.to_owned(),
            })?;
        let mut file = self.files[info.file]
            .lock()
            .map_err(|_| candle_core::Error::Msg("safetensors file lock poisoned".to_owned()))?;
        file.seek(SeekFrom::Start(self.data_start[info.file] + info.start))?;
        let mut bytes = vec![0u8; (info.end - info.start) as usize];
        file.read_exact(&mut bytes)?;
        let values = match info.dtype {
            DType::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect::<Vec<_>>(),
            DType::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| f32::from_bits(u32::from(u16::from_le_bytes(*chunk)) << 16))
                .collect::<Vec<_>>(),
            other => {
                return Err(candle_core::Error::Msg(format!(
                    "{name}: unsupported dtype {other:?}"
                )));
            }
        };
        Tensor::from_vec(values, info.shape.clone(), device)
    }
}

/// A Qwen3.5 text trunk that materializes one layer at a time.
///
struct LayerCache {
    budget: usize,
    used: usize,
    layers: HashMap<usize, Vec<(String, Tensor)>>,
}

impl LayerCache {
    fn new(budget: usize) -> Self {
        Self {
            budget,
            used: 0,
            layers: HashMap::new(),
        }
    }

    fn get(&self, index: usize) -> Option<Vec<(String, Tensor)>> {
        self.layers.get(&index).cloned()
    }

    fn insert(&mut self, index: usize, tensors: &[(String, Tensor)]) {
        if self.budget == 0 || self.layers.contains_key(&index) {
            return;
        }
        let bytes = tensors
            .iter()
            .map(|(_, tensor)| tensor.elem_count() * tensor.dtype().size_in_bytes())
            .sum::<usize>();
        if self.used.saturating_add(bytes) <= self.budget {
            self.used += bytes;
            self.layers.insert(index, tensors.to_vec());
        }
    }
}

/// The 9B Clef backbone does not fit in fp32 RAM together with its untied LM
/// head. Only the token embedding and output embedding tables stay resident;
/// each transformer layer is read from the safetensors shards, widened to
/// fp32, run, and dropped. `load_with_cache` pins whole layers up to a byte
/// budget (the transformer layers are visited in order, so a prefix cache is
/// enough) and the OS page cache keeps the rest fast. Numerics are identical
/// to [`Qwen35Encoder`].
pub struct Qwen35StreamingEncoder {
    config: Qwen35Config,
    embed_tokens: Embedding,
    lm_head: Option<Linear>,
    norm: RmsNorm,
    rotary: Rotary,
    weights: ShardedSafetensors,
    /// `(absolute key, key within the layer)` for every tensor of each layer.
    layer_keys: Vec<Vec<(String, String)>>,
    cache: Mutex<LayerCache>,
    device: Device,
}

impl Qwen35StreamingEncoder {
    pub fn load(
        config: &Qwen35Config,
        paths: &[PathBuf],
        device: &Device,
    ) -> std::result::Result<Self, ModelError> {
        Self::load_with_cache(config, paths, device, 0)
    }

    /// `cache_bytes` bounds the f32 layer weights kept resident; 0 disables
    /// the cache and streams every layer.
    pub fn load_with_cache(
        config: &Qwen35Config,
        paths: &[PathBuf],
        device: &Device,
        cache_bytes: usize,
    ) -> std::result::Result<Self, ModelError> {
        config.validate()?;
        if paths.is_empty() {
            return Err(ModelError::config(
                "Qwen3.5 sharded checkpoint has no files",
            ));
        }
        tracing::debug!(
            shards = paths.len(),
            layers = config.num_hidden_layers,
            hidden = config.hidden_size,
            cache_bytes,
            "opening Qwen3.5 streaming checkpoint"
        );
        let weights = ShardedSafetensors::open(paths)?;
        let names = weights.names().cloned().collect::<Vec<_>>();
        let prefix = ["model.language_model", "model", ""]
            .into_iter()
            .find(|prefix| names.contains(&prefix_key(prefix, "embed_tokens.weight")))
            .ok_or_else(|| ModelError::config("Qwen3.5 checkpoint has no embed_tokens.weight"))?;
        let load = |name: &str| -> std::result::Result<Tensor, ModelError> {
            Ok(weights
                .load(&prefix_key(prefix, name), device)?
                .to_dtype(DType::F32)?)
        };
        let embed_tokens = Embedding::new(load("embed_tokens.weight")?, config.hidden_size);
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            // Multimodal checkpoints keep the untied head at the root, next to
            // the language-model prefix rather than inside it.
            let name = if names.contains(&prefix_key(prefix, "lm_head.weight")) {
                prefix_key(prefix, "lm_head.weight")
            } else {
                "lm_head.weight".to_owned()
            };
            Some(Linear::new(
                weights.load(&name, device)?.to_dtype(DType::F32)?,
                None,
            ))
        };
        let norm_weight = (load("norm.weight")? + 1.0)?;
        let norm = RmsNorm::new(norm_weight, config.rms_norm_eps);
        let layer_keys = (0..config.num_hidden_layers)
            .map(|index| {
                let full = layer_prefix(prefix, index);
                let keys = names
                    .iter()
                    .filter_map(|name| {
                        name.strip_prefix(&full)
                            .map(|local| (name.clone(), local.to_owned()))
                    })
                    .collect::<Vec<_>>();
                if keys.is_empty() {
                    return Err(ModelError::config(format!(
                        "Qwen3.5 checkpoint has no tensors under {full}"
                    )));
                }
                Ok(keys)
            })
            .collect::<std::result::Result<Vec<_>, ModelError>>()?;
        Ok(Self {
            config: config.clone(),
            embed_tokens,
            lm_head,
            norm,
            rotary: Rotary::new(config, DType::F32, device)?,
            weights,
            layer_keys,
            cache: Mutex::new(LayerCache::new(cache_bytes)),
            device: device.clone(),
        })
    }

    /// Hidden states for one sequence, `[batch, seq, hidden]`.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let mut hidden = self.embed_tokens.forward(ids)?;
        for (index, keys) in self.layer_keys.iter().enumerate() {
            let tensors = {
                let mut cache = self
                    .cache
                    .lock()
                    .map_err(|_| candle_core::Error::Msg("layer cache lock poisoned".to_owned()))?;
                match cache.get(index) {
                    Some(tensors) => tensors,
                    None => {
                        tracing::trace!(
                            layer = index,
                            tensors = keys.len(),
                            "streaming Qwen3.5 layer from disk"
                        );
                        let mut tensors = HashMap::with_capacity(keys.len());
                        for (absolute, local) in keys {
                            tensors.insert(
                                local.clone(),
                                self.weights
                                    .load(absolute, &self.device)?
                                    .to_dtype(DType::F32)?,
                            );
                        }
                        let tensors = tensors.into_iter().collect::<Vec<_>>();
                        cache.insert(index, &tensors);
                        tensors
                    }
                }
            };
            let tensors = tensors.into_iter().collect::<HashMap<_, _>>();
            let vb = VarBuilder::from_tensors(tensors, DType::F32, &self.device);
            let layer = Layer::load(&self.config, index, vb)?;
            hidden = layer.forward(&hidden, &self.rotary)?;
        }
        self.norm.forward(&hidden)
    }

    /// The token embedding table used by the LM head (tied or untied).
    pub fn output_embedding_weight(&self) -> Tensor {
        match &self.lm_head {
            Some(head) => head.weight().clone(),
            None => self.embed_tokens.embeddings().clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(shape: (usize, usize, usize, usize), seed: usize) -> Tensor {
        let count = shape.0 * shape.1 * shape.2 * shape.3;
        let data = (0..count)
            .map(|index| ((index * 37 + seed * 11) % 19) as f32 / 19.0 - 0.5)
            .collect::<Vec<_>>();
        Tensor::from_vec(data, shape, &Device::Cpu).expect("values")
    }

    #[test]
    fn windowed_attention_matches_the_full_matrix() {
        let device = Device::Cpu;
        let (batch, heads, seq, dim) = (2, 3, 11, 4);
        let query = values((batch, heads, seq, dim), 1);
        let key = values((batch, heads, seq, dim), 2);
        let value = values((batch, heads, seq, dim), 3);
        let scale = 1.0 / (dim as f32).sqrt();
        let scores = (query
            .matmul(&key.transpose(2, 3).expect("key").contiguous().expect("key"))
            .expect("scores")
            * scale as f64)
            .expect("scale");
        let mask = causal_mask(seq, DType::F32, &device).expect("mask");
        let full = softmax_last_dim(&scores.broadcast_add(&mask).expect("masked"))
            .expect("softmax")
            .matmul(&value)
            .expect("context");

        for budget in [1, 7, 64, 1_000_000] {
            let chunked =
                causal_attention_windowed(&query, &key, &value, scale, budget).expect("chunked");
            let diff = (full.clone() - chunked)
                .expect("diff")
                .abs()
                .expect("abs")
                .flatten_all()
                .expect("flatten")
                .max(0)
                .expect("max")
                .to_scalar::<f32>()
                .expect("scalar");
            println!("budget {budget}: max |Δattention| = {diff:e}");
            assert!(diff < 1e-6, "budget {budget}: diff {diff}");
        }
    }
}
