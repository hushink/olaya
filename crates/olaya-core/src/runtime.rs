use indexmap::IndexMap;

use crate::manifest::RepoManifest;
use crate::metrics::RuntimeMetrics;
use crate::model::ModelMetadata;
use crate::questions::Questions;
use crate::request::Asset;
use crate::response::SystemOneResponse;
use crate::state::State;

/// The engine boundary the HTTP server sees.
///
/// `olaya::Engine` implements this; third-party engines may implement it too.
/// The typed library API lives on the engine itself.
pub trait DecisionRuntime: Send + Sync {
    /// Run one decision request. The returned `model` is canonical.
    fn predict(
        &self,
        model: &str,
        state: &State,
        questions: &Questions,
    ) -> Result<SystemOneResponse, RuntimeError>;

    /// Run several requests for the same model, preserving input order. The
    /// default runs them one at a time; runtimes with a batched backbone can
    /// share one forward pass.
    fn predict_batch(
        &self,
        model: &str,
        requests: &[(&State, &Questions)],
    ) -> Vec<Result<SystemOneResponse, RuntimeError>> {
        requests
            .iter()
            .map(|(state, questions)| self.predict(model, state, questions))
            .collect()
    }

    /// Run one request with the assets the server received (inline base64 or
    /// multipart parts). Runtimes that do not read images can keep the default,
    /// which drops them and calls [`DecisionRuntime::predict`].
    fn predict_with_assets(
        &self,
        model: &str,
        state: &State,
        questions: &Questions,
        assets: &IndexMap<String, Asset>,
    ) -> Result<SystemOneResponse, RuntimeError> {
        let _ = assets;
        self.predict(model, state, questions)
    }

    /// Names and aliases exposed by `GET /v1/models`.
    fn list_models(&self) -> Vec<ModelMetadata>;

    /// The catalog entry for a name or alias.
    fn resolve(&self, name: &str) -> Option<RepoManifest>;

    /// Load a catalog model into memory.
    fn load(&self, name: &str) -> Result<(), RuntimeError> {
        let _ = name;
        Ok(())
    }

    /// Unload a resident model, returning whether it was resident.
    fn unload(&self, name: &str) -> bool {
        let _ = name;
        false
    }

    /// Canonical names of currently resident models.
    fn loaded_models(&self) -> Vec<String> {
        Vec::new()
    }

    /// Pull a model reference into the local catalog.
    fn pull(&self, reference: &str) -> Result<RepoManifest, RuntimeError> {
        let _ = reference;
        Err(RuntimeError::Unsupported {
            message: "pull is not supported by this runtime".to_owned(),
        })
    }

    /// Remove a model from the local catalog and delete its blobs.
    fn delete(&self, name: &str) -> Result<bool, RuntimeError> {
        let _ = name;
        Ok(false)
    }

    /// Shared metrics handle. Runtimes that keep counters return a clone so the
    /// HTTP server can add its own request/latency metrics to the same snapshot.
    fn runtime_metrics(&self) -> Option<RuntimeMetrics> {
        None
    }
}

/// Errors raised by the engine or a model family.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum RuntimeError {
    #[error("unknown model: {model}")]
    ModelNotFound { model: String },
    #[error("invalid request: {message}")]
    InvalidRequest { message: String },
    #[error("too many options for `{question}`: {options} > {limit}")]
    TooManyOptions {
        question: String,
        options: usize,
        limit: usize,
    },
    #[error("input too long: {tokens} tokens")]
    InputTooLong { tokens: usize },
    #[error("part of the state was dropped to fit the context of {model}")]
    StateTruncated { model: String },
    #[error("model load failed: {message}")]
    LoadFailed { message: String },
    #[error("inference failed: {message}")]
    Inference { message: String },
    #[error("unsupported: {message}")]
    Unsupported { message: String },
    #[error("family `{family}` is not registered")]
    FamilyNotRegistered { family: String },
    #[error("registry error: {message}")]
    Registry { message: String },
    #[error("storage error: {message}")]
    Storage { message: String },
}

impl RuntimeError {
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
        }
    }

    pub fn load_failed(message: impl Into<String>) -> Self {
        Self::LoadFailed {
            message: message.into(),
        }
    }

    pub fn inference(message: impl Into<String>) -> Self {
        Self::Inference {
            message: message.into(),
        }
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported {
            message: message.into(),
        }
    }
}
