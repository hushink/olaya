//! ModernBERT encoder shared by laya, NLI, Von, and Decima.
//!
//! The laya checkpoint stores the encoder weights under `encoder.*` while the
//! canonical names are `model.*`; [`EncoderRenamer`] bridges the two without
//! copying the weights.
//!
//! The forward pass is adapted from candle-transformers' `modernbert` model
//! (MIT/Apache-2.0), which builds its 4-D attention masks as F32 unconditionally.
//! Keeping the masks in the weight dtype is what makes F16/BF16 inference
//! possible on CUDA and Metal, so the handful of layers needed here live in
//! this module instead.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::ops::softmax;
use candle_nn::var_builder::{Rename, Renamer};
use candle_nn::{
    Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm_no_bias, linear_no_bias,
};
use serde::Deserialize;

use crate::ModelError;

/// The subset of `encoder/config.json` that the encoder needs.
#[derive(Clone, Debug, Deserialize)]
pub struct EncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    #[serde(default)]
    pub pad_token_id: u32,
    #[serde(default)]
    pub cls_token_id: Option<u32>,
    #[serde(default)]
    pub sep_token_id: Option<u32>,
    #[serde(default)]
    pub mask_token_id: Option<u32>,
    pub global_attn_every_n_layers: usize,
    pub local_attention: usize,
    /// Older configs spell the RoPE bases as two scalar fields; newer ones use
    /// `rope_parameters`.
    #[serde(default)]
    pub global_rope_theta: Option<f64>,
    #[serde(default)]
    pub local_rope_theta: Option<f64>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RopeParameters {
    pub full_attention: RopeParameter,
    pub sliding_attention: RopeParameter,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RopeParameter {
    pub rope_theta: f64,
}

impl EncoderConfig {
    pub fn from_file(path: &Path) -> std::result::Result<Self, ModelError> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes)
            .map_err(|error| ModelError::config(format!("{}: {error}", path.display())))
    }

    fn rope_thetas(&self) -> (f64, f64) {
        match &self.rope_parameters {
            Some(rope) => (
                rope.full_attention.rope_theta,
                rope.sliding_attention.rope_theta,
            ),
            None => (
                self.global_rope_theta.unwrap_or(160_000.0),
                self.local_rope_theta.unwrap_or(10_000.0),
            ),
        }
    }
}

/// Renames the `model.*` keys the encoder asks for to the checkpoint's `encoder.*`.
pub struct EncoderRenamer;

impl Renamer for EncoderRenamer {
    fn rename(&self, value: &str) -> Cow<'_, str> {
        match value.strip_prefix("model.") {
            Some(rest) => Cow::Owned(format!("encoder.{rest}")),
            None => Cow::Owned(value.to_owned()),
        }
    }
}

/// Strips the `model.` prefix the encoder asks for, for checkpoints saved as the
/// bare `ModernBertModel` (`embeddings.*`, `layers.*`, `final_norm.*`).
pub struct StripModelRenamer;

impl Renamer for StripModelRenamer {
    fn rename(&self, value: &str) -> Cow<'_, str> {
        match value.strip_prefix("model.") {
            Some(rest) => Cow::Owned(rest.to_owned()),
            None => Cow::Owned(value.to_owned()),
        }
    }
}

/// The most negative finite value a dtype can hold, matching `torch.finfo`.
fn dtype_min(dtype: DType) -> f32 {
    match dtype {
        DType::F16 => -65504.0,
        // f32::MIN rounds to -inf in BF16; use BF16's own finite minimum.
        DType::BF16 => -3.389_531_4e38,
        _ => f32::MIN,
    }
}

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, config: &EncoderConfig, rope_theta: f64, dev: &Device) -> Result<Self> {
        let dim = config.hidden_size / config.num_attention_heads;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let max_seq_len = config.max_position_embeddings;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &self.cos, &self.sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &self.cos, &self.sin)?;
        Ok((q_embed, k_embed))
    }
}

struct Attention {
    qkv: Linear,
    proj: Linear,
    num_attention_heads: usize,
    attention_head_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl Attention {
    fn load(
        vb: VarBuilder,
        config: &EncoderConfig,
        rotary_emb: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let qkv = linear_no_bias(config.hidden_size, config.hidden_size * 3, vb.pp("Wqkv"))?;
        let proj = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("Wo"))?;
        Ok(Self {
            qkv,
            proj,
            num_attention_heads,
            attention_head_size,
            rotary_emb,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let xs = hidden_states.clone();
        let (b, seq_len, d) = xs.dims3()?;
        let qkv = xs
            .apply(&self.qkv)?
            .reshape((
                b,
                seq_len,
                3,
                self.num_attention_heads,
                self.attention_head_size,
            ))?
            .permute((2, 0, 3, 1, 4))?;

        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;

        let (q, k) = self.rotary_emb.apply_rotary_emb_qkv(&q, &k)?;

        let scale = (self.attention_head_size as f64).powf(-0.5);
        let q = (q * scale)?;

        let att = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let att = att.broadcast_add(attention_mask)?;
        let att = softmax(&att, D::Minus1)?;

        let xs = att.matmul(&v)?;
        let xs = xs.transpose(1, 2)?.reshape((b, seq_len, d))?;
        xs.apply(&self.proj)
    }
}

struct Mlp {
    wi: Linear,
    wo: Linear,
}

impl Mlp {
    fn load(vb: VarBuilder, config: &EncoderConfig) -> Result<Self> {
        let wi = linear_no_bias(
            config.hidden_size,
            config.intermediate_size * 2,
            vb.pp("Wi"),
        )?;
        let wo = linear_no_bias(config.intermediate_size, config.hidden_size, vb.pp("Wo"))?;
        Ok(Self { wi, wo })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.wi)?;
        let xs = xs.chunk(2, D::Minus1)?;
        (&xs[0].gelu_erf()? * &xs[1])?.apply(&self.wo)
    }
}

struct Layer {
    attn: Attention,
    mlp: Mlp,
    attn_norm: Option<LayerNorm>,
    mlp_norm: LayerNorm,
    uses_local_attention: bool,
}

impl Layer {
    fn load(
        vb: VarBuilder,
        config: &EncoderConfig,
        rotary_emb: Arc<RotaryEmbedding>,
        uses_local_attention: bool,
    ) -> Result<Self> {
        let attn = Attention::load(vb.pp("attn"), config, rotary_emb)?;
        let mlp = Mlp::load(vb.pp("mlp"), config)?;
        let attn_norm = layer_norm_no_bias(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("attn_norm"),
        )
        .ok();
        let mlp_norm =
            layer_norm_no_bias(config.hidden_size, config.layer_norm_eps, vb.pp("mlp_norm"))?;
        Ok(Self {
            attn,
            mlp,
            attn_norm,
            mlp_norm,
            uses_local_attention,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        global_attention_mask: &Tensor,
        local_attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let mut xs = xs.clone();
        if let Some(norm) = &self.attn_norm {
            xs = xs.apply(norm)?;
        }
        let attention_mask = if self.uses_local_attention {
            &global_attention_mask.broadcast_add(local_attention_mask)?
        } else {
            global_attention_mask
        };
        let xs = self.attn.forward(&xs, attention_mask)?;
        let xs = (xs + residual)?;
        let mlp_out = xs.apply(&self.mlp_norm)?.apply(&self.mlp)?;
        xs + mlp_out
    }
}

/// Global attention mask calculated from padded token inputs, in `dtype`.
fn prepare_4d_attention_mask(mask: &Tensor, dtype: DType) -> Result<Tensor> {
    let bsz = mask.dim(0)?;
    let src_len = mask.dim(1)?;
    let expanded_mask = mask
        .unsqueeze(1)?
        .unsqueeze(2)?
        .expand((bsz, 1, src_len, src_len))?
        .to_dtype(DType::F32)?;
    let inverted_mask = (1.0 - expanded_mask)?;
    (inverted_mask * dtype_min(dtype) as f64)?.to_dtype(dtype)
}

/// Attention mask caused by the sliding window, in `dtype`.
fn get_local_attention_mask(
    seq_len: usize,
    max_distance: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let min = dtype_min(dtype);
    let mask: Vec<_> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len).map(move |j| {
                if (j as i32 - i as i32).abs() > max_distance as i32 {
                    min
                } else {
                    0.
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (seq_len, seq_len), device)?.to_dtype(dtype)
}

struct ModernBert {
    word_embeddings: Embedding,
    norm: LayerNorm,
    layers: Vec<Layer>,
    final_norm: LayerNorm,
    local_attention_size: usize,
    dtype: DType,
}

impl ModernBert {
    fn load(vb: VarBuilder, config: &EncoderConfig) -> Result<Self> {
        let dtype = vb.dtype();
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embeddings.tok_embeddings"),
        )?;
        let norm = layer_norm_no_bias(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("model.embeddings.norm"),
        )?;
        let (global_theta, local_theta) = config.rope_thetas();
        let global_rotary_emb = Arc::new(RotaryEmbedding::new(
            dtype,
            config,
            global_theta,
            vb.device(),
        )?);
        let local_rotary_emb = Arc::new(RotaryEmbedding::new(
            dtype,
            config,
            local_theta,
            vb.device(),
        )?);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_id in 0..config.num_hidden_layers {
            let local = layer_id % config.global_attn_every_n_layers != 0;
            layers.push(Layer::load(
                vb.pp(format!("model.layers.{layer_id}")),
                config,
                if local {
                    local_rotary_emb.clone()
                } else {
                    global_rotary_emb.clone()
                },
                local,
            )?);
        }
        let final_norm = layer_norm_no_bias(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("model.final_norm"),
        )?;
        Ok(Self {
            word_embeddings,
            norm,
            layers,
            final_norm,
            local_attention_size: config.local_attention,
            dtype,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let seq_len = xs.shape().dims()[1];
        let global_attention_mask =
            prepare_4d_attention_mask(mask, self.dtype)?.to_device(xs.device())?;
        let local_attention_mask = get_local_attention_mask(
            seq_len,
            self.local_attention_size / 2,
            self.dtype,
            xs.device(),
        )?;
        let mut xs = xs.apply(&self.word_embeddings)?.apply(&self.norm)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &global_attention_mask, &local_attention_mask)?;
        }
        xs.apply(&self.final_norm)
    }
}

/// A loaded ModernBERT encoder.
pub struct ModernBertEncoder {
    model: ModernBert,
}

impl ModernBertEncoder {
    pub fn load(vb: VarBuilder, config: &EncoderConfig) -> std::result::Result<Self, ModelError> {
        tracing::debug!(
            dtype = ?vb.dtype(),
            hidden = config.hidden_size,
            layers = config.num_hidden_layers,
            heads = config.num_attention_heads,
            vocab = config.vocab_size,
            "loading ModernBERT encoder"
        );
        Ok(Self {
            model: ModernBert::load(vb, config)?,
        })
    }

    pub fn dtype(&self) -> DType {
        self.model.dtype
    }

    pub fn forward(&self, ids: &Tensor, mask: &Tensor) -> std::result::Result<Tensor, ModelError> {
        Ok(self.model.forward(ids, mask)?)
    }
}

/// Load a laya-style checkpoint (`encoder.*` keys) as an encoder in `dtype`.
pub fn load_encoder(
    weights: &Path,
    config: &EncoderConfig,
    device: &Device,
    dtype: DType,
) -> std::result::Result<ModernBertEncoder, ModelError> {
    let tensors = candle_core::safetensors::load(weights, device)?;
    let vb = VarBuilder::from_tensors(tensors, dtype, device);
    let renamed = VarBuilder::from_backend(
        Box::new(Rename::new(vb, EncoderRenamer)),
        dtype,
        device.clone(),
    );
    ModernBertEncoder::load(renamed, config)
}

/// Build an encoder from an already-loaded tensor map, without re-reading the
/// checkpoint, using `encoder.*` names.
pub fn load_encoder_from_tensors(
    tensors: &HashMap<String, Tensor>,
    config: &EncoderConfig,
    device: &Device,
    dtype: DType,
) -> std::result::Result<ModernBertEncoder, ModelError> {
    let vb = VarBuilder::from_tensors(tensors.clone(), dtype, device);
    let renamed = VarBuilder::from_backend(
        Box::new(Rename::new(vb, EncoderRenamer)),
        dtype,
        device.clone(),
    );
    ModernBertEncoder::load(renamed, config)
}

#[cfg(test)]
mod tests {
    use super::EncoderConfig;
    use serde_json::json;

    fn config(extra: serde_json::Value) -> EncoderConfig {
        let mut config = json!({
            "vocab_size": 50371,
            "hidden_size": 384,
            "num_hidden_layers": 10,
            "num_attention_heads": 6,
            "intermediate_size": 576,
            "max_position_embeddings": 7999,
            "layer_norm_eps": 1e-5,
            "global_attn_every_n_layers": 3,
            "local_attention": 128
        });
        for (key, value) in extra.as_object().expect("object") {
            config
                .as_object_mut()
                .expect("config object")
                .insert(key.clone(), value.clone());
        }
        serde_json::from_value(config).expect("config")
    }

    #[test]
    fn scalar_rope_thetas_override_the_defaults() {
        let config = config(json!({
            "global_rope_theta": 160_000.0,
            "local_rope_theta": 160_000.0
        }));
        assert_eq!(config.rope_thetas(), (160_000.0, 160_000.0));
    }

    #[test]
    fn nested_rope_parameters_win() {
        let config = config(json!({
            "global_rope_theta": 1.0,
            "local_rope_theta": 2.0,
            "rope_parameters": {
                "full_attention": {"rope_theta": 160_000.0},
                "sliding_attention": {"rope_theta": 10_000.0}
            }
        }));
        assert_eq!(config.rope_thetas(), (160_000.0, 10_000.0));
    }
}
