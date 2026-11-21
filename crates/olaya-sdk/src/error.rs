//! SDK errors and the TypeSafe `{"detail": ...}` error envelope.

use std::fmt;
use std::time::Duration;

use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::retry::RetryPolicy;

/// An SDK failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("the TypeSafe API key is missing; set TYPESAFE_API_KEY or call ClientBuilder::api_key")]
    MissingApiKey,
    #[error("the TypeSafe API key is invalid: {reason}")]
    InvalidApiKey { reason: String },
    #[error("invalid client configuration: {message}")]
    Config { message: String },
    #[error("invalid request: {message}")]
    InvalidRequest { message: String },
    #[error("image error: {message}")]
    Image { message: String },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("HTTP transport error for {method} {endpoint} after {attempts} attempt(s): {source}")]
    Transport {
        method: &'static str,
        endpoint: String,
        attempts: u32,
        #[source]
        source: reqwest::Error,
    },
    #[error("request to {method} {endpoint} timed out after {timeout:?} (attempt {attempts})")]
    Timeout {
        method: &'static str,
        endpoint: String,
        timeout: Duration,
        attempts: u32,
        #[source]
        source: Option<Box<reqwest::Error>>,
    },
    #[error(transparent)]
    Api(#[from] Box<ApiError>),
    #[error("could not decode the response from {method} {endpoint}: {source}")]
    Decode {
        method: &'static str,
        endpoint: String,
        field_path: Option<String>,
        body: String,
        request_id: Option<String>,
        #[source]
        source: serde_json::Error,
    },
}

impl From<ApiError> for Error {
    fn from(value: ApiError) -> Self {
        Self::Api(Box::new(value))
    }
}

impl Error {
    /// The request id attached to an API, decode, or timeout error, when any.
    pub fn request_id(&self) -> Option<&str> {
        match self {
            Self::Api(error) => error.request_id.as_deref(),
            Self::Decode { request_id, .. } => request_id.as_deref(),
            _ => None,
        }
    }

    /// The HTTP status of an API error.
    pub fn status(&self) -> Option<StatusCode> {
        match self {
            Self::Api(error) => Some(error.status),
            _ => None,
        }
    }

    /// Whether the failure was a request timeout.
    pub fn is_timeout(&self) -> bool {
        matches!(self, Self::Timeout { .. })
    }

    /// Whether this error is retried by `policy`.
    pub fn is_retryable(&self, policy: &RetryPolicy) -> bool {
        match self {
            Self::Api(error) => error.is_retryable(policy),
            Self::Timeout { .. } => policy.retry_timeouts,
            Self::Transport { .. } => policy.retry_connection_errors,
            _ => false,
        }
    }

    /// The server's requested retry delay, if this is an API error that
    /// supplied one.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::Api(error) => error.retry_after,
            _ => None,
        }
    }

    pub(crate) fn set_attempts(&mut self, attempts: u32) {
        match self {
            Self::Api(error) => error.attempts = attempts,
            Self::Transport {
                attempts: value, ..
            }
            | Self::Timeout {
                attempts: value, ..
            } => *value = attempts,
            _ => {}
        }
    }
}

/// A non-successful HTTP response.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    /// The request method, such as `POST`.
    pub method: &'static str,
    /// The request URL, without credentials.
    pub endpoint: String,
    /// The parsed error body, or the raw text when the body is not the
    /// official envelope.
    pub body: ErrorBody,
    /// The `x-typesafe-request-id` response header, when present.
    pub request_id: Option<String>,
    /// The server's requested retry delay, when present.
    pub retry_after: Option<Duration>,
    /// Total attempts made, including the initial attempt.
    pub attempts: u32,
}

impl ApiError {
    pub fn kind(&self) -> ApiErrorKind {
        ApiErrorKind::from_status(self.status)
    }

    /// The server's requested retry delay in milliseconds.
    pub fn retry_after_ms(&self) -> Option<u64> {
        self.retry_after
            .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
    }

    /// Whether this status is retried by `policy`.
    pub fn is_retryable(&self, policy: &RetryPolicy) -> bool {
        policy.retry_statuses.contains(&self.status.as_u16())
    }

    /// A human-readable detail, combining all validation errors when present.
    pub fn message(&self) -> Option<String> {
        self.body.message()
    }

    /// The validation errors of a 422 response, when present.
    pub fn validation_errors(&self) -> Option<&[ValidationError]> {
        self.body.validation()
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "TypeSafe API error {} {} for {} {}",
            self.status.as_u16(),
            self.status.canonical_reason().unwrap_or("Unknown"),
            self.method,
            self.endpoint
        )?;
        if let Some(message) = self.message() {
            write!(formatter, ": {message}")?;
        }
        if let Some(request_id) = &self.request_id {
            write!(formatter, " (request id {request_id})")?;
        }
        Ok(())
    }
}

impl std::error::Error for ApiError {}

/// A coarse classification of an API error status.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ApiErrorKind {
    BadRequest,
    Authentication,
    PermissionDenied,
    NotFound,
    PayloadTooLarge,
    UnsupportedMediaType,
    UnprocessableEntity,
    RateLimit,
    Overloaded,
    InternalServer,
    Other,
}

impl ApiErrorKind {
    pub fn from_status(status: StatusCode) -> Self {
        match status.as_u16() {
            400 => Self::BadRequest,
            401 => Self::Authentication,
            403 => Self::PermissionDenied,
            404 => Self::NotFound,
            413 => Self::PayloadTooLarge,
            415 => Self::UnsupportedMediaType,
            422 => Self::UnprocessableEntity,
            429 => Self::RateLimit,
            529 => Self::Overloaded,
            500..=599 => Self::InternalServer,
            _ => Self::Other,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::BadRequest => "bad_request",
            Self::Authentication => "authentication",
            Self::PermissionDenied => "permission_denied",
            Self::NotFound => "not_found",
            Self::PayloadTooLarge => "payload_too_large",
            Self::UnsupportedMediaType => "unsupported_media_type",
            Self::UnprocessableEntity => "unprocessable_entity",
            Self::RateLimit => "rate_limit",
            Self::Overloaded => "overloaded",
            Self::InternalServer => "internal_server",
            Self::Other => "other",
        }
    }
}

/// The body of an unsuccessful response.
#[derive(Clone, Debug, PartialEq)]
pub enum ErrorBody {
    Empty,
    Json(ErrorResponse),
    Text(String),
}

impl ErrorBody {
    /// Parse an error response body, falling back to the raw text.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self::Empty;
        }
        if let Ok(response) = serde_json::from_slice::<ErrorResponse>(bytes) {
            return Self::Json(response);
        }
        let text = String::from_utf8_lossy(bytes).trim().to_owned();
        if text.is_empty() {
            Self::Empty
        } else {
            Self::Text(text)
        }
    }

    pub fn detail(&self) -> Option<&ErrorDetail> {
        match self {
            Self::Json(response) => Some(&response.detail),
            _ => None,
        }
    }

    pub fn validation(&self) -> Option<&[ValidationError]> {
        match self.detail() {
            Some(ErrorDetail::Validation(errors)) => Some(errors),
            _ => None,
        }
    }

    /// A human-readable message. Validation errors are joined in order.
    pub fn message(&self) -> Option<String> {
        match self {
            Self::Empty => None,
            Self::Text(text) => Some(text.clone()),
            Self::Json(response) => Some(match &response.detail {
                ErrorDetail::Message(message) => message.clone(),
                ErrorDetail::Object(object) => object
                    .get("message")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .unwrap_or_else(|| {
                        serde_json::to_string(object)
                            .unwrap_or_else(|_| "invalid error object".into())
                    }),
                ErrorDetail::Validation(errors) => errors
                    .iter()
                    .map(|error| {
                        let path = error.field_path();
                        if path.is_empty() {
                            error.msg.clone()
                        } else {
                            format!("{path}: {}", error.msg)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("; "),
            }),
        }
    }
}

/// The official error envelope: `{"detail": ...}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub detail: ErrorDetail,
}

/// The official union of error details.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ErrorDetail {
    Validation(Vec<ValidationError>),
    Message(String),
    Object(Map<String, Value>),
}

/// A request validation error at a specific field or array element.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ValidationError {
    /// Path to the invalid value, such as `["body", "questions", "tone"]`.
    pub loc: Vec<LocSegment>,
    /// A human-readable explanation.
    pub msg: String,
    /// A machine-readable error code, such as `missing` or `type_error`.
    #[serde(rename = "type")]
    pub kind: String,
    /// A preview of the rejected input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    /// Additional structured context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctx: Option<Map<String, Value>>,
}

impl ValidationError {
    /// Join the location segments with dots, such as
    /// `body.questions.tone.criteria`.
    pub fn field_path(&self) -> String {
        self.loc
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(".")
    }
}

/// One segment of a validation error location.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LocSegment {
    String(String),
    Integer(i64),
}

impl fmt::Display for LocSegment {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String(value) => formatter.write_str(value),
            Self::Integer(value) => write!(formatter, "{value}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_a_string_detail() {
        let body = ErrorBody::from_bytes(br#"{"detail":"unknown model: nope"}"#);
        assert_eq!(body.message().as_deref(), Some("unknown model: nope"));
        assert!(body.validation().is_none());
    }

    #[test]
    fn parses_validation_details() {
        let body = ErrorBody::from_bytes(
            br#"{"detail":[{"loc":["body","questions","urgency","criteria"],"msg":"List should have at least 1 item","type":"too_short"}]}"#,
        );
        let errors = body.validation().expect("validation");
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, "too_short");
        assert_eq!(errors[0].field_path(), "body.questions.urgency.criteria");
        assert_eq!(
            body.message().as_deref(),
            Some("body.questions.urgency.criteria: List should have at least 1 item")
        );
    }

    #[test]
    fn parses_an_object_detail() {
        let body = ErrorBody::from_bytes(br#"{"detail":{"message":"bad key","code":7}}"#);
        assert_eq!(body.message().as_deref(), Some("bad key"));
        let ErrorBody::Json(response) = &body else {
            panic!("expected a JSON error body");
        };
        let ErrorDetail::Object(object) = &response.detail else {
            panic!("expected an object detail");
        };
        assert_eq!(object["code"], 7);
    }

    #[test]
    fn falls_back_to_text() {
        let body = ErrorBody::from_bytes(b"gateway timeout");
        assert_eq!(body, ErrorBody::Text("gateway timeout".to_owned()));
        assert_eq!(body.message().as_deref(), Some("gateway timeout"));
    }

    #[test]
    fn loc_segments_support_integers() {
        let error: ValidationError = serde_json::from_value(json!({
            "loc": ["body", "questions", 0, "criteria"],
            "msg": "bad",
            "type": "value_error"
        }))
        .expect("validation");
        assert_eq!(error.field_path(), "body.questions.0.criteria");
    }

    #[test]
    fn api_error_display_includes_status_and_request_id() {
        let error = ApiError {
            status: StatusCode::UNAUTHORIZED,
            method: "POST",
            endpoint: "https://api.typesafe.ai/v1/systemone".to_owned(),
            body: ErrorBody::from_bytes(br#"{"detail":"invalid key"}"#),
            request_id: Some("req-1".to_owned()),
            retry_after: None,
            attempts: 1,
        };
        let text = error.to_string();
        assert!(text.contains("401 Unauthorized"));
        assert!(text.contains("invalid key"));
        assert!(text.contains("req-1"));
        assert_eq!(error.kind(), ApiErrorKind::Authentication);
    }

    #[test]
    fn retry_classification_uses_the_policy() {
        let policy = RetryPolicy::default();
        let mut error = ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            method: "POST",
            endpoint: "http://localhost".to_owned(),
            body: ErrorBody::Empty,
            request_id: None,
            retry_after: None,
            attempts: 1,
        };
        assert!(error.is_retryable(&policy));
        error.status = StatusCode::BAD_REQUEST;
        assert!(!error.is_retryable(&policy));
        assert!(!error.is_retryable(&RetryPolicy::disabled()));
    }
}
