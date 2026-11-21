//! BERT encoder (Decima's multilingual-e5-small, and other BERT checkpoints).
//!
//! Wraps `candle-transformers`' BERT model with the same ids + mask interface
//! as the other encoders.

use std::path::Path;

use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};

use crate::ModelError;

pub use candle_transformers::models::bert::Config as BertConfig;

/// Read a Hugging Face BERT `config.json`.
pub fn read_config(path: &Path) -> Result<Config, ModelError> {
    let bytes = std::fs::read(path)?;
    serde_json::from_slice(&bytes)
        .map_err(|error| ModelError::config(format!("{}: {error}", path.display())))
}

/// A loaded BERT encoder.
pub struct BertEncoder {
    model: BertModel,
}

impl BertEncoder {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self, ModelError> {
        tracing::debug!(
            dtype = ?vb.dtype(),
            hidden = config.hidden_size,
            layers = config.num_hidden_layers,
            heads = config.num_attention_heads,
            "loading BERT encoder"
        );
        Ok(Self {
            model: BertModel::load(vb, config)?,
        })
    }

    pub fn forward(&self, ids: &Tensor, mask: &Tensor) -> Result<Tensor, ModelError> {
        let token_types = Tensor::zeros_like(ids)?;
        Ok(self.model.forward(ids, &token_types, Some(mask))?)
    }
}
