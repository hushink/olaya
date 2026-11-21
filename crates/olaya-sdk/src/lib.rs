//! A typed Rust client SDK for the TypeSafe/Jev System One API.
//!
//! This crate is independent of the other `olaya` crates: it only depends on
//! general-purpose HTTP and serialization libraries. It speaks the official
//! TypeSafe wire format and can talk to `https://api.typesafe.ai`, an
//! OpenAI-compatible gateway, or a local `olaya serve` instance.
//!
//! See the [crate README](https://github.com/hushink/olaya/blob/HEAD/crates/olaya-sdk/README.md)
//! for complete examples and differences from the official Python SDK.
//!
//! # Quickstart
//!
//! ```no_run
//! use olaya_sdk::{ChoiceQuestion, Client, NoulQuestion, ScoreQuestion, questions};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), olaya_sdk::Error> {
//!     // Reads TYPESAFE_API_KEY, TYPESAFE_BASE_URL, TYPESAFE_DEFAULT_MODEL.
//!     let client = Client::from_env()?;
//!     let response = client
//!         .system_one(
//!             "I was charged twice for my subscription this month. Please refund it.",
//!             questions! {
//!                 "billing" => NoulQuestion::new("Is this message about billing?"),
//!                 "tone" => ChoiceQuestion::new([
//!                     ("calm", None),
//!                     ("frustrated", None),
//!                     ("angry", None),
//!                 ]),
//!                 "urgency" => ScoreQuestion::new([
//!                     "Can wait",
//!                     "Needs attention this week",
//!                     "Needs attention today",
//!                 ]),
//!             },
//!         )
//!         .await?;
//!
//!     println!("billing: {}", response.noul("billing").expect("answer").noul);
//!     println!("tone: {}", response.choice("tone").expect("answer").choice);
//!     println!("urgency: {}", response.score("urgency").expect("answer").score);
//!     println!("request id: {:?}", response.request_id);
//!     Ok(())
//! }
//! ```
//!
//! # What is included
//!
//! - Typed questions ([`NoulQuestion`], [`ChoiceQuestion`], [`ScoreQuestion`],
//!   [`BoxQuestion`]) and answers ([`Answer`]).
//! - Typed error bodies, including official 422 validation details.
//! - Retries with exponential backoff, jitter, `Retry-After` support, and a
//!   total per-call deadline.
//! - Per-request timeouts, custom headers, extra request fields, and request
//!   ids.
//! - Inline base64 images and multipart image uploads.
//! - An optional blocking client behind the `blocking` feature.
//! - Optional gzip request-body compression behind the `gzip` feature.
//! - `tracing` debug logs with request/response summaries and retry warnings;
//!   bodies and credentials are never logged.
//! - Forward compatibility: unknown answer kinds are preserved in
//!   [`Answer::Unknown`], and unknown response fields are ignored.
//!
//! # Environment
//!
//! | Variable | Purpose | Default |
//! | --- | --- | --- |
//! | `TYPESAFE_API_KEY` | API key | required |
//! | `TYPESAFE_BASE_URL` | API root | `https://api.typesafe.ai` |
//! | `TYPESAFE_DEFAULT_MODEL` | Default model | `jev-latest` |
//!
//! An explicit builder value takes precedence over the environment. Explicitly
//! empty values are rejected rather than falling back. Use
//! [`ClientBuilder::allow_unauthenticated`] for local servers that do not
//! require a key, and [`ClientBuilder::without_api_key`] to suppress a key
//! inherited from the environment.
//!
//! # Blocking client
//!
//! With the `blocking` feature:
//!
//! ```ignore
//! use olaya_sdk::blocking::Client;
//! use olaya_sdk::{NoulQuestion, questions};
//!
//! let client = Client::from_env()?;
//! let response = client
//!     .system_one(
//!         "I was charged twice.",
//!         questions! { "billing" => NoulQuestion::new("Is this about billing?") },
//!     )
//!     .send()?;
//! # Ok::<(), olaya_sdk::Error>(())
//! ```

#![forbid(unsafe_code)]

pub mod answer;
#[cfg(feature = "blocking")]
pub mod blocking;
pub mod client;
pub mod config;
pub mod content;
pub mod error;
pub mod image;
pub mod question;
pub mod request;
pub mod retry;

mod transport;

pub use answer::{
    Answer, AnswerType, BoxAnswer, BoxCoordinates, ChoiceAnswer, ListModelsResponse, ModelMetadata,
    NoulAnswer, ScoreAnswer, SystemOneResponse, Usage,
};
#[cfg(feature = "blocking")]
pub use blocking::Client as BlockingClient;
pub use client::{Client, ClientBuilder, Models, SystemOneCall};
pub use config::{
    API_KEY_ENV, BASE_URL_ENV, Compression, DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_MODEL_ENV,
    DEFAULT_TIMEOUT,
};
pub use content::{Content, ContentError, State, StateError};
pub use error::{
    ApiError, ApiErrorKind, Error, ErrorBody, ErrorDetail, ErrorResponse, LocSegment,
    ValidationError,
};
pub use image::{Image, ImageFormat};
pub use question::{
    BoxQuestion, ChoiceQuestion, NoulCriteria, NoulQuestion, Question, QuestionType, ScoreQuestion,
};
pub use request::{
    Asset, Assets, MAX_CHOICE_OPTIONS, MAX_SCORE_LEVELS, Questions, SystemOneRequest,
};
pub use retry::RetryPolicy;

pub use reqwest::StatusCode;

/// Build a [`Questions`] map from `"name" => question` pairs.
///
/// ```
/// use olaya_sdk::{NoulQuestion, questions};
///
/// let questions = questions! {
///     "billing" => NoulQuestion::new("Is this about billing?"),
/// };
/// assert_eq!(questions.len(), 1);
/// ```
#[macro_export]
macro_rules! questions {
    () => {
        $crate::Questions::new()
    };
    ($($name:expr => $question:expr),+ $(,)?) => {{
        let mut questions = $crate::Questions::new();
        $(
            questions.insert(
                ::std::string::String::from($name),
                $crate::Question::from($question),
            );
        )+
        questions
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn questions_macro_preserves_order_and_accepts_trailing_commas() {
        let questions = crate::questions! {
            "a" => NoulQuestion::new("a?"),
            "b" => ChoiceQuestion::labels(["x", "y"]),
            "c" => ScoreQuestion::new(["low", "high"]),
        };
        assert_eq!(
            questions.keys().map(String::as_str).collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
        assert_eq!(questions.get("b").expect("b").type_name(), "choice");
    }

    #[test]
    fn questions_macro_can_be_empty() {
        assert!(crate::questions!().is_empty());
    }
}
