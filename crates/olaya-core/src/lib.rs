//! Engine-agnostic System One protocol, primitives, decision layer, and manifests.
//!
//! This crate owns the wire types of `POST /v1/systemone` and `GET /v1/models`,
//! the typed question/answer primitives, the renderer/readout boundary that
//! every model family implements, and the model manifest. It has no dependency
//! on candle, HTTP, or a download engine.

#![forbid(unsafe_code)]

pub mod answer;
pub mod answers;
pub mod catalog;
pub mod content;
pub mod decision;
pub mod error;
pub mod manifest;
pub mod metrics;
pub mod model;
pub mod parse;
mod printable;
pub mod pyjson;
pub mod pyrepr;
pub mod question;
pub mod questions;
pub mod registry;
pub mod request;
pub mod response;
pub mod runtime;
pub mod state;
pub mod store;
pub mod validation;

pub use answer::{
    Answer, AnswerType, BoxAnswer, BoxCoordinates, ChoiceAnswer, NoulAnswer, ScoreAnswer,
};
pub use answers::Answers;
pub use catalog::{CATALOG_SCHEMA_VERSION, CatalogBundle};
pub use content::{Content, ContentError};
pub use decision::{
    Calibration, CalibrationFile, CalibrationMode, Capabilities, DecisionError, Encoded, QType,
    Readout, Renderer, TemperatureMap, TokenEncoder, answer_from_logits, argmax, bucket,
    choice_answer, clamp_temperature, noul_answer, score_answer, softmax, temperature_range,
    typesafe_confidence,
};
pub use error::{AnswerError, Error};
pub use manifest::{
    ANNOTATION_PATH, ANNOTATION_ROLE, Descriptor, FileRole, MEDIA_CALIBRATION, MEDIA_CONFIG,
    MEDIA_DECISION, MEDIA_LICENSE, MEDIA_PICKLE, MEDIA_README, MEDIA_SAFETENSORS, MEDIA_TOKENIZER,
    MEDIA_TYPE_MANIFEST, ManifestError, ManifestFile, ModelConfig, OciManifest, REPO_MANIFEST_FILE,
    REPO_MANIFEST_SCHEMA, RepoManifest, RouterConfig,
};
pub use metrics::{HistogramSnapshot, MetricsSnapshot, RuntimeMetrics};
pub use model::{ModelMetadata, ModelMetadataList, is_valid_release_date};
pub use parse::{parse_request_slice, parse_request_value};
pub use pyjson::{
    dumps_content, dumps_content_canonical, dumps_value, dumps_value_canonical, float_repr,
};
pub use question::{
    BoxQuestion, ChoiceQuestion, NoulCriteria, NoulQuestion, Question, QuestionType, ScoreQuestion,
};
pub use questions::Questions;
pub use registry::{ManifestRegistry, RegistryError};
pub use request::{Asset, AssetFormat, AssetKind, SystemOneRequest};
pub use response::{SystemOneResponse, Usage};
pub use runtime::{DecisionRuntime, RuntimeError};
pub use state::{State, StateError};
pub use store::{ManifestStore, StoreError, StoredModel};
pub use validation::{
    ErrorDetail, ErrorResponse, LocSegment, MessageDetail, ValidationError, ValidationReport,
    error_type,
};

/// Official decision endpoint.
pub const SYSTEMONE_PATH: &str = "/v1/systemone";
/// Official model-list endpoint.
pub const MODELS_PATH: &str = "/v1/models";
/// Response header used for request correlation.
pub const REQUEST_ID_HEADER: &str = "x-typesafe-request-id";
/// Opt-in header required for box questions and image assets.
pub const DANGER_ZONE_IMAGES_HEADER: &str = "typesafe-danger-zone-images";
/// Opt-in header for long-context requests.
pub const DANGER_ZONE_LONG_CONTEXT_HEADER: &str = "typesafe-danger-zone-long-context";

/// Maximum number of questions in one request.
pub const MAX_QUESTIONS: usize = 256;
/// Maximum number of choice options.
pub const MAX_CHOICE_OPTIONS: usize = 255;
/// Maximum number of score levels.
pub const MAX_SCORE_LEVELS: usize = 10;
/// Maximum state length in tokens (server boundary).
pub const MAX_STATE_TOKENS: usize = 65_536;
/// Maximum request body size in bytes.
pub const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;

/// A fresh request id for correlation headers.
pub fn new_request_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("req_{nanos:x}{n:04x}")
}
