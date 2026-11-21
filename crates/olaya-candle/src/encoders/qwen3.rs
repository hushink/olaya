//! Qwen3 decoder trunk (Qwen3-0.6B/4B/8B family).
//!
//! One stateless causal forward per row: embeddings, RMSNorm, GQA attention
//! with per-head QK RMSNorm and RoPE, SwiGLU MLP, final RMSNorm. The optional
//! LM head is tied unless the checkpoint ships an untied one; families that
//! read only a few candidate tokens gather those embedding rows instead of
//! projecting the whole vocabulary.
//!
//! Follows `candle_transformers::models::qwen3` (which uses the same layer
//! order and RoPE layout) but keeps no KV cache: a decision is one forward pass
//! and concurrency must not share attention state.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::attention::{AttnMask, flash_attn};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{
    Activation, Embedding, Linear, RmsNorm, VarBuilder, embedding, linear_b, linear_no_bias,
    rms_norm,
};
use serde::Deserialize;

use crate::ModelError;

fn default_max_positions() -> usize {
    32_768
}

fn default_rope_theta() -> f64 {
    1_000_000.0
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_hidden_act() -> String {
    "silu".to_owned()
}

/// The HF `config.json` of a Qwen3 checkpoint.
#[derive(Clone, Debug, Deserialize)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_max_positions")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub max_window_layers: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
}

fn default_kv_heads() -> usize {
    0
}

impl Qwen3Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    fn kv_heads(&self) -> usize {
        if self.num_key_value_heads == 0 {
            self.num_attention_heads
        } else {
            self.num_key_value_heads
        }
    }

    pub fn validate(&self) -> std::result::Result<(), ModelError> {
        if self.hidden_size == 0
            || self.num_attention_heads == 0
            || self.num_hidden_layers == 0
            || self.intermediate_size == 0
        {
            return Err(ModelError::config(
                "Qwen3 hidden_size, heads, layers and intermediate_size must be positive",
            ));
        }
        if self.head_dim() * self.num_attention_heads == 0 {
            return Err(ModelError::config("Qwen3 head_dim must be positive"));
        }
        if self.kv_heads() == 0 || !self.num_attention_heads.is_multiple_of(self.kv_heads()) {
            return Err(ModelError::config(format!(
                "Qwen3 num_attention_heads {} is not a multiple of num_key_value_heads {}",
                self.num_attention_heads,
                self.kv_heads()
            )));
        }
        if self.use_sliding_window || self.sliding_window.is_some() {
            return Err(ModelError::config(
                "Qwen3 sliding-window attention is not supported",
            ));
        }
        if self.max_position_embeddings == 0 {
            return Err(ModelError::config("Qwen3 max_position_embeddings is zero"));
        }
        Ok(())
    }

    fn activation(&self) -> std::result::Result<Activation, ModelError> {
        match self.hidden_act.as_str() {
            "silu" => Ok(Activation::Silu),
            other => Err(ModelError::config(format!(
                "unsupported Qwen3 hidden_act {other:?}; expected silu"
            ))),
        }
    }
}

struct Rotary {
    cos: Tensor,
    sin: Tensor,
}

impl Rotary {
    fn new(config: &Qwen3Config, dtype: DType, device: &Device) -> Result<Self> {
        let dim = config.head_dim();
        let max = config.max_position_embeddings;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|index| 1f32 / config.rope_theta.powf(index as f64 / dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, dim / 2), device)?.to_dtype(DType::F32)?;
        let positions = Tensor::arange(0u32, max as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max, 1))?;
        let freqs = positions.matmul(&inv_freq)?;
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq = q.dim(2)?;
        let cos = self.cos.narrow(0, 0, seq)?;
        let sin = self.sin.narrow(0, 0, seq)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary: Rotary,
    device: Device,
}

impl Attention {
    fn load(config: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let heads = config.num_attention_heads;
        let kv_heads = config.kv_heads();
        let head_dim = config.head_dim();
        let bias = config.attention_bias;
        let projection = |input, output, name: &str| {
            if bias {
                linear_b(input, output, true, vb.pp(name))
            } else {
                linear_no_bias(input, output, vb.pp(name))
            }
        };
        let q_proj = projection(config.hidden_size, heads * head_dim, "q_proj")?;
        let k_proj = projection(config.hidden_size, kv_heads * head_dim, "k_proj")?;
        let v_proj = projection(config.hidden_size, kv_heads * head_dim, "v_proj")?;
        let o_proj = projection(heads * head_dim, config.hidden_size, "o_proj")?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: rms_norm(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?,
            heads,
            kv_heads,
            head_dim,
            rotary: Rotary::new(config, vb.dtype(), vb.device())?,
            device: vb.device().clone(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq, _) = x.dims3()?;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((batch, seq, self.heads, self.head_dim))?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((batch, seq, self.kv_heads, self.head_dim))?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((batch, seq, self.kv_heads, self.head_dim))?;
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let q = self.q_norm.forward(&q.flatten(0, 2)?)?.reshape((
            batch,
            self.heads,
            seq,
            self.head_dim,
        ))?;
        let k = self.k_norm.forward(&k.flatten(0, 2)?)?.reshape((
            batch,
            self.kv_heads,
            seq,
            self.head_dim,
        ))?;
        let (q, k) = self.rotary.apply(&q, &k)?;

        let groups = self.heads / self.kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let context = if self.device.is_cpu() && q.dtype() == DType::F32 {
            // CPU flash takes (B, S, H, D) and returns (B, H, S, D).
            flash_attn::<f32>(
                &q.transpose(1, 2)?.contiguous()?,
                &k.transpose(1, 2)?.contiguous()?,
                &v.transpose(1, 2)?.contiguous()?,
                scale,
                AttnMask::causal(),
                None,
                None,
            )?
        } else {
            standard_attention(&q, &k, &v, scale)?
        };
        context
            .transpose(1, 2)?
            .reshape((batch, seq, self.heads * self.head_dim))?
            .apply(&self.o_proj)
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

fn standard_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    let (_, _, seq, _) = q.dims4()?;
    let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale as f64)?;
    let mask = causal_mask(seq, scores.dtype(), scores.device())?;
    let scores = scores.broadcast_add(&mask)?;
    softmax_last_dim(&scores)?.matmul(v)
}

fn causal_mask(seq: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mut values = Vec::with_capacity(seq * seq);
    for row in 0..seq {
        for column in 0..seq {
            values.push(if column <= row {
                0.0f32
            } else {
                f32::NEG_INFINITY
            });
        }
    }
    Tensor::from_vec(values, (seq, seq), device)?
        .to_dtype(dtype)?
        .unsqueeze(0)?
        .unsqueeze(0)
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
    activation: Activation,
}

impl Mlp {
    fn load(config: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
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
            activation: config
                .activation()
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(x)?.apply(&self.activation)?;
        let up = self.up.forward(x)?;
        (gate * up)?.apply(&self.down)
    }
}

struct Layer {
    input_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    attention: Attention,
    mlp: Mlp,
}

impl Layer {
    fn load(config: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_norm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_norm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            attention: Attention::load(config, vb.pp("self_attn"))?,
            mlp: Mlp::load(config, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.input_norm.forward(x)?;
        let x = (x + self.attention.forward(&h)?)?;
        let h = self.post_attention_norm.forward(&x)?;
        x + self.mlp.forward(&h)
    }
}

/// A stateless Qwen3 trunk plus its (optional) untied LM head.
pub struct Qwen3Encoder {
    embed_tokens: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Option<Linear>,
    device: Device,
    dtype: DType,
}

impl Qwen3Encoder {
    pub fn load(config: &Qwen3Config, vb: VarBuilder) -> std::result::Result<Self, ModelError> {
        config.validate()?;
        tracing::debug!(
            dtype = ?vb.dtype(),
            hidden = config.hidden_size,
            layers = config.num_hidden_layers,
            heads = config.num_attention_heads,
            kv_heads = config.kv_heads(),
            vocab = config.vocab_size,
            "loading Qwen3 decoder"
        );
        let embed_tokens = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| Layer::load(config, vb.pp(format!("model.layers.{index}"))))
            .collect::<Result<Vec<_>>>()?;
        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(linear_no_bias(
                config.hidden_size,
                config.vocab_size,
                vb.pp("lm_head"),
            )?)
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Hidden states for a right-padded batch, `[batch, seq, hidden]`.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let mut hidden = self.embed_tokens.forward(ids)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        self.norm.forward(&hidden)
    }

    /// Logits of `hidden` (`[batch, hidden]`) over a candidate token subset.
    ///
    /// The tied LM head is the embedding table, so instead of projecting the
    /// whole vocabulary this gathers the candidate rows and dots them with the
    /// hidden state.
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
