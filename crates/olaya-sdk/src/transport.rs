//! Helpers shared by the async and blocking transports.

use reqwest::StatusCode;
use reqwest::header::HeaderMap;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::config::Compression;
use crate::error::{ApiError, Error, ErrorBody};
use crate::question::Question;
use crate::request::SystemOneRequest;

pub(crate) const REQUEST_ID_HEADER: &str = "x-typesafe-request-id";
pub(crate) const DANGER_ZONE_IMAGES: &str = "typesafe-danger-zone-images";
pub(crate) const DANGER_ZONE_LONG_CONTEXT: &str = "typesafe-danger-zone-long-context";

/// Read the request id from response headers.
pub(crate) fn request_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get(REQUEST_ID_HEADER)
        .or_else(|| headers.get("x-request-id"))
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

/// Build an [`ApiError`] from a non-success response.
pub(crate) fn api_error(
    status: StatusCode,
    method: &'static str,
    endpoint: &str,
    headers: &HeaderMap,
    bytes: &[u8],
    attempts: u32,
) -> ApiError {
    ApiError {
        status,
        method,
        endpoint: endpoint.to_owned(),
        body: ErrorBody::from_bytes(bytes),
        request_id: request_id(headers),
        retry_after: crate::retry::parse_retry_after(headers),
        attempts,
    }
}

/// Deserialize a successful response, reporting the failing field path.
pub(crate) fn decode_response<T: DeserializeOwned>(
    method: &'static str,
    endpoint: &str,
    headers: &HeaderMap,
    bytes: &[u8],
) -> Result<T, Error> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    match serde_path_to_error::deserialize(&mut deserializer) {
        Ok(value) => Ok(value),
        Err(error) => {
            let path = error.path().to_string();
            let field_path = if path.is_empty() || path == "." {
                None
            } else {
                Some(path)
            };
            Err(Error::Decode {
                method,
                endpoint: endpoint.to_owned(),
                field_path,
                body: String::from_utf8_lossy(bytes).into_owned(),
                request_id: request_id(headers),
                source: error.into_inner(),
            })
        }
    }
}

/// Whether the request needs `TypeSafe-Danger-Zone-Images: true`.
pub(crate) fn requires_image_header(request: &SystemOneRequest) -> bool {
    !request.assets.is_empty()
        || request
            .questions
            .values()
            .any(|question| matches!(question, Question::Box(_)))
}

/// Serialize a JSON body, applying gzip when configured and worthwhile.
pub(crate) fn encode_json_body(
    json: &Value,
    compression: Compression,
) -> Result<(Vec<u8>, bool), Error> {
    let bytes = serde_json::to_vec(json).map_err(|error| Error::Config {
        message: format!("serializing the request body: {error}"),
    })?;
    #[cfg(feature = "gzip")]
    if compression == Compression::Gzip && bytes.len() >= 1024 {
        return Ok((gzip(&bytes)?, true));
    }
    let _ = compression;
    Ok((bytes, false))
}

#[cfg(feature = "gzip")]
fn gzip(bytes: &[u8]) -> Result<Vec<u8>, Error> {
    use std::io::Write;

    use flate2::Compression as FlateCompression;
    use flate2::write::GzEncoder;

    let mut encoder = GzEncoder::new(Vec::new(), FlateCompression::default());
    encoder.write_all(bytes)?;
    Ok(encoder.finish()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderName, HeaderValue};
    use serde_json::json;

    #[test]
    fn request_ids_prefer_the_typesafe_header() {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-request-id"),
            HeaderValue::from_static("fallback"),
        );
        headers.insert(
            HeaderName::from_static("x-typesafe-request-id"),
            HeaderValue::from_static("typesafe"),
        );
        assert_eq!(request_id(&headers).as_deref(), Some("typesafe"));
        headers.remove("x-typesafe-request-id");
        assert_eq!(request_id(&headers).as_deref(), Some("fallback"));
    }

    #[test]
    fn decodes_validation_failures_with_a_field_path() {
        let headers = HeaderMap::new();
        let bytes = br#"{"model":"jev","answers":{"tone":{"type":"choice","choice":"calm","confidence":"high","probabilities":{}}},"usage":{}}"#;
        let error = decode_response::<crate::answer::SystemOneResponse>(
            "POST",
            "https://api.typesafe.ai/v1/systemone",
            &headers,
            bytes,
        )
        .expect_err("decode");
        match error {
            Error::Decode { field_path, .. } => {
                // Nested answer kinds use a custom deserializer, so the path
                // stops at the answer entry rather than the inner field.
                assert_eq!(field_path.as_deref(), Some("answers.tone"));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn image_header_is_required_for_assets_and_box_questions() {
        use crate::content::State;
        use crate::question::{BoxQuestion, NoulQuestion};

        let mut questions = crate::request::Questions::new();
        questions.insert(
            "billing".to_owned(),
            Question::from(NoulQuestion::new("billing?")),
        );
        let request = SystemOneRequest::new(State::text("hello"), questions);
        assert!(!requires_image_header(&request));

        let mut questions = crate::request::Questions::new();
        questions.insert(
            "where".to_owned(),
            Question::from(BoxQuestion::new("the dog")),
        );
        let request = SystemOneRequest::new(State::text("hello"), questions);
        assert!(requires_image_header(&request));

        let image =
            crate::image::Image::from_bytes(vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A])
                .expect("png");
        let request = SystemOneRequest::new(
            State::text("hello"),
            std::iter::once((
                "billing".to_owned(),
                Question::from(NoulQuestion::new("billing?")),
            ))
            .collect(),
        )
        .inline_image("<photo>", image);
        assert!(requires_image_header(&request));
    }

    #[test]
    fn json_body_encoding_is_uncompressed_by_default() {
        let (bytes, compressed) =
            encode_json_body(&json!({"text": "hello"}), Compression::None).expect("body");
        assert!(!compressed);
        assert_eq!(bytes, br#"{"text":"hello"}"#);
    }
}
