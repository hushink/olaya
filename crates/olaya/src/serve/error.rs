use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use olaya_core::{
    ErrorResponse, LocSegment, RuntimeError, ValidationError, ValidationReport, error_type,
};
use serde_json::{Map, Value};

/// An error response in the official TypeSafe `{"detail": ...}` shape.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub body: ErrorResponse,
}

impl ApiError {
    pub fn new(status: StatusCode, body: ErrorResponse) -> Self {
        Self { status, body }
    }

    pub fn validation(report: ValidationReport) -> Self {
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            body: ErrorResponse::validation(report.into_errors()),
        }
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            body: ErrorResponse::bad_request(message),
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            body: ErrorResponse::bad_request(message),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: ErrorResponse::bad_request(message),
        }
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            body: ErrorResponse::bad_request(message),
        }
    }

    pub fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            body: ErrorResponse::bad_request(
                "missing or invalid API key; send Authorization: Bearer <key>",
            ),
        }
    }

    pub fn payload_too_large() -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            body: ErrorResponse::bad_request("request body is too large"),
        }
    }

    /// Map an engine error to the official shape.
    pub fn runtime(error: RuntimeError) -> Self {
        match error {
            RuntimeError::ModelNotFound { model } => {
                Self::not_found(format!("unknown model: {model}"))
            }
            RuntimeError::InvalidRequest { message } => Self::bad_request(message),
            RuntimeError::TooManyOptions {
                question,
                options,
                limit,
            } => Self::validation(ValidationReport::single(ValidationError::new(
                [
                    LocSegment::key("body"),
                    LocSegment::key("questions"),
                    LocSegment::key(question),
                    LocSegment::key("criteria"),
                ],
                format!("{options} options exceed the model limit of {limit}"),
                error_type::TOO_MANY_OPTIONS,
            ))),
            RuntimeError::InputTooLong { tokens } => {
                Self::validation(ValidationReport::single(ValidationError::new(
                    [LocSegment::key("body"), LocSegment::key("state")],
                    format!("state is {tokens} tokens long"),
                    error_type::TOO_LONG,
                )))
            }
            RuntimeError::StateTruncated { model } => {
                let mut ctx = Map::new();
                ctx.insert("model".to_owned(), Value::String(model.clone()));
                Self::validation(ValidationReport::single(
                    ValidationError::new(
                        [LocSegment::key("body"), LocSegment::key("state")],
                        format!("part of the state was dropped to fit the context of {model}"),
                        error_type::STATE_TRUNCATED,
                    )
                    .with_ctx(ctx),
                ))
            }
            RuntimeError::FamilyNotRegistered { family } => {
                Self::validation(ValidationReport::single(ValidationError::new(
                    [LocSegment::key("body"), LocSegment::key("model")],
                    format!("family {family:?} is not registered; rebuild with the family crate"),
                    error_type::UNSUPPORTED_FAMILY,
                )))
            }
            RuntimeError::LoadFailed { message } => {
                Self::internal(format!("model load failed: {message}"))
            }
            RuntimeError::Inference { message } => {
                Self::internal(format!("inference failed: {message}"))
            }
            RuntimeError::Storage { message } => {
                Self::internal(format!("storage error: {message}"))
            }
            RuntimeError::Registry { message } => {
                Self::internal(format!("registry error: {message}"))
            }
            RuntimeError::Unsupported { message } => Self::unsupported(message),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn state_truncation_is_a_validation_error() {
        let error = ApiError::runtime(RuntimeError::StateTruncated {
            model: "decima-guard".to_owned(),
        });
        assert_eq!(error.status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(
            serde_json::to_value(&error.body).expect("json"),
            json!({
                "detail": [{
                    "loc": ["body", "state"],
                    "msg": "part of the state was dropped to fit the context of decima-guard",
                    "type": "state_truncated",
                    "ctx": {"model": "decima-guard"}
                }]
            })
        );
    }
}
