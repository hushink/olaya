use crate::answer::AnswerType;
use crate::decision::DecisionError;
use crate::manifest::ManifestError;
use crate::runtime::RuntimeError;
use crate::validation::ValidationReport;

/// Registry and in-process inference errors.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("unknown model: {model}")]
    ModelNotFound { model: String },
    #[error("model `{model}` failed: {source}")]
    Model {
        model: String,
        #[source]
        source: RuntimeError,
    },
    #[error(transparent)]
    Validation(#[from] ValidationReport),
    #[error(transparent)]
    Decision(#[from] DecisionError),
    #[error(transparent)]
    Manifest(#[from] ManifestError),
}

/// A typed answer lookup failure.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum AnswerError {
    #[error("answer `{id}` is missing")]
    Missing { id: String },
    #[error("answer `{id}` has type {actual:?}, expected {expected:?}")]
    WrongType {
        id: String,
        expected: AnswerType,
        actual: AnswerType,
    },
}
