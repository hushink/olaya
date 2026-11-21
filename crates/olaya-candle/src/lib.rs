//! candle family SDK.
//!
//! A model family composes three pieces: a [`Renderer`] that turns a state and
//! a question into an [`Encoded`] row, a [`Backbone`] that turns that row into
//! option logits, and a [`Readout`] that turns the logits into a typed answer.
//!
//! Third-party families depend on this crate and [`olaya_core`] only; they do
//! not depend on the `olaya` facade, its hub, or its built-in families.

#![forbid(unsafe_code)]

use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use olaya_core::{
    Answer, Calibration, Capabilities, DecisionError, Encoded, Questions, Readout, Renderer,
    RepoManifest, State, SystemOneResponse, TokenEncoder, Usage,
};

pub mod device;

pub mod families;

/// PyTorch `torch.save` checkpoint reader (used by Von).
#[cfg(feature = "torch")]
pub mod torch;

#[cfg(any(
    feature = "modernbert",
    feature = "deberta",
    feature = "bert",
    feature = "qwen3",
    feature = "qwen35"
))]
pub mod encoders;

#[cfg(feature = "tokenizer")]
pub mod tokenizer;

/// Errors raised while loading or running a family.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[error(transparent)]
    Decision(#[from] DecisionError),
    #[error("invalid model configuration: {0}")]
    Config(String),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
}

impl ModelError {
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config(message.into())
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported(message.into())
    }
}

/// Everything a factory needs to build a family.
pub struct LoadContext<'a> {
    /// The model directory (the directory that holds `olaya.manifest.json`).
    pub dir: &'a Path,
    pub manifest: &'a RepoManifest,
    pub device: &'a Device,
    pub dtype: DType,
    pub tokenizer: Arc<dyn TokenEncoder>,
}

impl LoadContext<'_> {
    /// The dtype a family should build its weights and activations in.
    ///
    /// Candle executes half-precision CPU kernels by widening to F32, which is
    /// slower than running F32 end to end, so a CPU device always computes in
    /// F32. CUDA and Metal honor the requested F16/BF16.
    pub fn compute_dtype(&self) -> DType {
        if self.device.is_cpu() || (self.dtype == DType::BF16 && !self.device.supports_bf16()) {
            DType::F32
        } else {
            self.dtype
        }
    }
}

/// The result of one backbone forward pass.
pub struct BackboneOutput {
    /// Option logits, one value per option.
    pub logits: Tensor,
    /// Optional action head output (laya's act head).
    pub act: Option<Tensor>,
}

/// A candle model that turns a rendered row into logits.
pub trait Backbone: Send + Sync {
    fn forward(&self, encoded: &Encoded) -> Result<BackboneOutput, ModelError>;

    /// Batched inference. The default preserves correctness by running each row.
    fn forward_batch(&self, batch: &[Encoded]) -> Result<Vec<BackboneOutput>, ModelError> {
        batch.iter().map(|encoded| self.forward(encoded)).collect()
    }
}

/// A complete decision model family.
pub trait Family: Send + Sync {
    fn id(&self) -> &'static str;

    fn capabilities(&self) -> Capabilities;

    fn tokenizer(&self) -> &dyn TokenEncoder;

    /// Optional renderer for the default `renderer → backbone → readout`
    /// composition. Families with a different row structure return `None` and
    /// override [`Family::predict`].
    fn renderer(&self) -> Option<&dyn Renderer> {
        None
    }

    /// Optional backbone for the default composition.
    fn backbone(&self) -> Option<&dyn Backbone> {
        None
    }

    /// Optional readout for the default composition.
    fn readout(&self) -> Option<&dyn Readout> {
        None
    }

    fn calibration(&self) -> &Calibration;

    /// Run every question against `state`.
    ///
    /// The default composes the renderer, backbone, and readout. Families that
    /// can share a forward pass across questions — or need several rows per
    /// question, as NLI does — override this.
    fn predict(
        &self,
        state: &olaya_core::State,
        questions: &Questions,
    ) -> Result<SystemOneResponse, ModelError> {
        let (Some(renderer), Some(backbone), Some(readout)) =
            (self.renderer(), self.backbone(), self.readout())
        else {
            return Err(ModelError::unsupported(format!(
                "family {:?} must override predict or provide renderer/backbone/readout",
                self.id()
            )));
        };
        if questions.iter().any(|(_, question)| question.is_ordinal()) {
            return Err(DecisionError::unsupported(format!(
                "family {:?} does not support ordinal choices",
                self.id()
            ))
            .into());
        }
        let tokenizer = self.tokenizer();
        let state_ids = renderer.encode_state(state, tokenizer)?;
        let mut answers = indexmap::IndexMap::with_capacity(questions.len());
        let mut input_tokens = 0u64;

        let mut encoded_rows = Vec::with_capacity(questions.len());
        for (id, question) in questions.iter() {
            let encoded = renderer.render(&state_ids, id, question, tokenizer)?;
            if encoded.truncated {
                return Err(DecisionError::StateTruncated.into());
            }
            input_tokens += encoded.ids.len() as u64;
            encoded_rows.push(encoded);
        }
        // Rows are independent (their decoders are masked), so one padded
        // backbone call replaces one call per question.
        let outputs = backbone.forward_batch(&encoded_rows)?;
        if outputs.len() != encoded_rows.len() {
            return Err(ModelError::Candle(candle_core::Error::Msg(format!(
                "backbone returned {} outputs for {} rows",
                outputs.len(),
                encoded_rows.len()
            ))));
        }
        for ((id, question), (encoded, output)) in questions
            .iter()
            .zip(encoded_rows.iter().zip(outputs.iter()))
        {
            let logits = output
                .logits
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let answer: Answer =
                readout.read(question, &logits, self.calibration(), encoded.state_tokens)?;
            answers.insert(id.clone(), answer);
        }

        Ok(SystemOneResponse::new(
            self.id(),
            answers,
            Usage::new(input_tokens, 0),
        ))
    }

    /// Run several requests in one call, preserving order. The default keeps
    /// the per-request behavior; families whose backbone accepts a row batch
    /// override this to share a forward pass.
    fn predict_batch(
        &self,
        requests: &[(&State, &Questions)],
    ) -> Vec<Result<SystemOneResponse, ModelError>> {
        requests
            .iter()
            .map(|(state, questions)| self.predict(state, questions))
            .collect()
    }
}

/// Loads a family from a manifest and a model directory.
pub trait FamilyFactory: Send + Sync {
    fn family(&self) -> &'static str;

    fn load(&self, ctx: &LoadContext<'_>) -> Result<Arc<dyn Family>, ModelError>;
}
