//! DeBERTa-v2/v3 encoder shared by NLI and GLiClass.
//!
//! candle-transformers ships the disentangled-attention encoder; this module
//! adapts the Hugging Face `config.json` shape and the `deberta.*` weight
//! prefix used by `DebertaV2ForSequenceClassification` checkpoints.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2;
use serde::Deserialize;
use serde::de::Deserializer;

use crate::ModelError;

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum StringOrVec {
    String(String),
    Vec(Vec<String>),
}

#[derive(Clone, Debug, Default)]
pub struct PosAttType(pub Vec<String>);

impl<'de> Deserialize<'de> for PosAttType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match StringOrVec::deserialize(deserializer)? {
            StringOrVec::String(value) => Ok(Self(
                value.split('|').map(str::to_owned).collect::<Vec<_>>(),
            )),
            StringOrVec::Vec(values) => Ok(Self(values)),
        }
    }
}

fn default_hidden_act() -> String {
    "gelu".to_owned()
}

fn default_dropout() -> f64 {
    0.1
}

fn default_max_position_embeddings() -> usize {
    512
}

fn default_initializer_range() -> f64 {
    0.02
}

fn default_layer_norm_eps() -> f64 {
    1e-7
}

fn default_true() -> bool {
    true
}

fn default_max_relative_positions() -> isize {
    -1
}

fn default_pos_att_type() -> PosAttType {
    PosAttType(vec!["p2c".to_owned(), "c2p".to_owned()])
}

/// The Hugging Face DeBERTa-v2/v3 config fields candle reads.
#[derive(Clone, Debug, Deserialize)]
pub struct DebertaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_dropout")]
    pub hidden_dropout_prob: f64,
    #[serde(default = "default_dropout")]
    pub attention_probs_dropout_prob: f64,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub type_vocab_size: usize,
    #[serde(default = "default_initializer_range")]
    pub initializer_range: f64,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_true")]
    pub relative_attention: bool,
    #[serde(default = "default_max_relative_positions")]
    pub max_relative_positions: isize,
    #[serde(default)]
    pub pad_token_id: Option<usize>,
    #[serde(default)]
    pub position_biased_input: bool,
    #[serde(default = "default_pos_att_type")]
    pub pos_att_type: PosAttType,
    #[serde(default)]
    pub position_buckets: Option<isize>,
    #[serde(default)]
    pub share_att_key: Option<bool>,
    #[serde(default)]
    pub attention_head_size: Option<usize>,
    #[serde(default)]
    pub embedding_size: Option<usize>,
    #[serde(default)]
    pub norm_rel_ebd: Option<String>,
    #[serde(default)]
    pub conv_kernel_size: Option<usize>,
    #[serde(default)]
    pub conv_groups: Option<usize>,
    #[serde(default)]
    pub conv_act: Option<String>,
    #[serde(default)]
    pub id2label: Option<HashMap<u32, String>>,
    #[serde(default)]
    pub label2id: Option<HashMap<String, u32>>,
    #[serde(default)]
    pub pooler_dropout: Option<f64>,
    #[serde(default)]
    pub pooler_hidden_act: Option<String>,
    #[serde(default)]
    pub pooler_hidden_size: Option<usize>,
    #[serde(default)]
    pub cls_dropout: Option<f64>,
}

impl DebertaConfig {
    pub fn from_file(path: &std::path::Path) -> Result<Self, ModelError> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes)
            .map_err(|error| ModelError::config(format!("{}: {error}", path.display())))
    }

    fn activation(value: &str) -> Result<debertav2::HiddenAct, ModelError> {
        match value {
            "gelu" => Ok(debertav2::HiddenAct::Gelu),
            "gelu_new" | "gelu_fast" => Ok(debertav2::HiddenAct::GeluApproximate),
            "relu" => Ok(debertav2::HiddenAct::Relu),
            other => Err(ModelError::config(format!(
                "unsupported DeBERTa activation {other:?}"
            ))),
        }
    }

    fn to_candle(&self) -> Result<debertav2::Config, ModelError> {
        let pooler_hidden_act = match &self.pooler_hidden_act {
            Some(value) => Some(Self::activation(value)?),
            None => Some(debertav2::HiddenAct::Gelu),
        };
        Ok(debertav2::Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            intermediate_size: self.intermediate_size,
            hidden_act: Self::activation(&self.hidden_act)?,
            hidden_dropout_prob: self.hidden_dropout_prob,
            attention_probs_dropout_prob: self.attention_probs_dropout_prob,
            max_position_embeddings: self.max_position_embeddings,
            type_vocab_size: self.type_vocab_size,
            initializer_range: self.initializer_range,
            layer_norm_eps: self.layer_norm_eps,
            relative_attention: self.relative_attention,
            max_relative_positions: self.max_relative_positions,
            pad_token_id: self.pad_token_id,
            position_biased_input: self.position_biased_input,
            pos_att_type: self.pos_att_type.0.clone(),
            position_buckets: self.position_buckets,
            share_att_key: self.share_att_key,
            attention_head_size: self.attention_head_size,
            embedding_size: self.embedding_size,
            norm_rel_ebd: self.norm_rel_ebd.clone(),
            conv_kernel_size: self.conv_kernel_size,
            conv_groups: self.conv_groups,
            conv_act: self.conv_act.clone(),
            id2label: self.id2label.clone(),
            label2id: self.label2id.clone(),
            pooler_dropout: self.pooler_dropout,
            pooler_hidden_act,
            pooler_hidden_size: self.pooler_hidden_size,
            cls_dropout: self.cls_dropout,
        })
    }
}

/// A loaded DeBERTa-v2/v3 encoder.
pub struct DebertaEncoder {
    model: debertav2::DebertaV2Model,
}

impl DebertaEncoder {
    /// Load from an F32 tensor map whose encoder keys live under `prefix`.
    pub fn load_with_prefix(
        tensors: &HashMap<String, Tensor>,
        config: &DebertaConfig,
        device: &Device,
        prefix: &str,
    ) -> Result<Self, ModelError> {
        let vb = VarBuilder::from_tensors(tensors.clone(), DType::F32, device).pp(prefix);
        tracing::debug!(
            prefix,
            hidden = config.hidden_size,
            layers = config.num_hidden_layers,
            heads = config.num_attention_heads,
            "loading DeBERTa encoder"
        );
        let model = debertav2::DebertaV2Model::load(vb, &config.to_candle()?)?;
        Ok(Self { model })
    }

    /// Load from an F32 tensor map whose keys are `deberta.*`.
    pub fn load(
        tensors: &HashMap<String, Tensor>,
        config: &DebertaConfig,
        device: &Device,
    ) -> Result<Self, ModelError> {
        Self::load_with_prefix(tensors, config, device, "deberta")
    }

    /// Load a whole safetensors checkpoint as F32.
    pub fn from_file(
        weights: &std::path::Path,
        config: &DebertaConfig,
        device: &Device,
    ) -> Result<Self, ModelError> {
        let mut tensors: HashMap<String, Tensor> = candle_core::safetensors::load(weights, device)?;
        for tensor in tensors.values_mut() {
            if tensor.dtype() != DType::F32 {
                *tensor = tensor.to_dtype(DType::F32)?;
            }
        }
        Self::load(&tensors, config, device)
    }

    pub fn forward(&self, ids: &Tensor, mask: &Tensor) -> Result<Tensor, ModelError> {
        Ok(self.model.forward(ids, None, Some(mask.clone()))?)
    }
}

/// A convenience wrapper for callers that only need the config.
pub fn parse_config(value: &serde_json::Value) -> Result<DebertaConfig, ModelError> {
    serde_json::from_value(value.clone())
        .map_err(|error: serde_json::Error| ModelError::config(error.to_string()))
}
