mod support;

use std::time::{Duration, Instant};

use olaya_sdk::{
    AnswerType, Client, Error, Image, NoulQuestion, Questions, RetryPolicy, StatusCode, questions,
};
use support::{MockResponse, MockServer};

const PNG: &[u8] = &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 1, 2, 3];

fn response_body() -> serde_json::Value {
    serde_json::json!({
        "model": "test-model",
        "answers": {
            "billing": {"type": "noul", "noul": 0.98},
            "tone": {
                "type": "choice",
                "choice": "calm",
                "confidence": 0.7,
                "probabilities": {"calm": 0.8, "angry": 0.2}
            },
            "urgency": {
                "type": "score",
                "score": 1.7,
                "confidence": 0.55,
                "legend": {"0": "Can wait", "1": "This week", "2": "Today"},
                "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8}
            }
        },
        "usage": {"input_tokens": 120, "output_tokens": 12}
    })
}

fn fast_retry(max_retries: u32) -> RetryPolicy {
    RetryPolicy::default()
        .max_retries(max_retries)
        .backoff_initial(Duration::from_millis(1))
        .backoff_max(Duration::from_millis(2))
        .backoff_jitter(0.0)
        .deadline(Some(Duration::from_secs(2)))
}

fn client_with(server: &MockServer, retry: RetryPolicy) -> Client {
    Client::builder()
        .api_key("test-key")
        .base_url(server.url())
        .default_model("test-model")
        .retry(retry)
        .build()
        .expect("client")
}

fn client(server: &MockServer) -> Client {
    client_with(server, RetryPolicy::disabled())
}

fn single_question() -> Questions {
    questions! {
        "billing" => NoulQuestion::new("Is this about billing?"),
    }
}

#[tokio::test]
async fn system_one_sends_a_typed_request_and_reads_a_typed_response() {
    let server = MockServer::start(|_request, _index| {
        MockResponse::json(200, response_body()).header("x-typesafe-request-id", "req-123")
    });
    let client = client(&server);

    // Await the call builder directly (IntoFuture).
    let response = client
        .system_one("I was charged twice. Please help.", single_question())
        .await
        .expect("response");

    assert_eq!(response.model, "test-model");
    assert_eq!(response.request_id.as_deref(), Some("req-123"));
    assert_eq!(response.noul("billing").expect("noul").noul, 0.98);
    assert_eq!(response.choice("tone").expect("choice").choice, "calm");
    assert_eq!(response.score("urgency").expect("score").score, 1.7);
    assert_eq!(response.usage.total_tokens(), Some(132));

    let requests = server.requests();
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/systemone");
    assert_eq!(request.header("authorization"), Some("Bearer test-key"));
    assert_eq!(request.header("accept"), Some("application/json"));
    assert!(
        request
            .header("user-agent")
            .expect("user agent")
            .starts_with("olaya-sdk/")
    );
    let body: serde_json::Value = serde_json::from_slice(&request.body).expect("request json");
    assert_eq!(body["model"], "test-model");
    assert_eq!(body["state"], "I was charged twice. Please help.");
    assert_eq!(body["questions"]["billing"]["type"], "noul");
    assert_eq!(
        body["questions"]["billing"]["instructions"],
        "Is this about billing?"
    );
}

#[tokio::test]
async fn a_call_can_override_the_model() {
    let server = MockServer::start(|_request, _index| MockResponse::json(200, response_body()));
    let client = client(&server);
    client
        .system_one("text", single_question())
        .model("jev-preview")
        .send()
        .await
        .expect("response");
    let body: serde_json::Value = serde_json::from_slice(&server.requests()[0].body).expect("json");
    assert_eq!(body["model"], "jev-preview");
}

#[tokio::test]
async fn list_models_uses_the_models_endpoint() {
    let server = MockServer::start(|_request, _index| {
        MockResponse::json(
            200,
            serde_json::json!({
                "models": [
                    {
                        "name": "jev-latest",
                        "description": "General-purpose system one model.",
                        "release_date": "2026-09-15"
                    }
                ]
            }),
        )
        .header("x-typesafe-request-id", "models-1")
    });
    let client = client(&server);
    let models = client
        .models()
        .header("X-Trace", "abc")
        .expect("header")
        .list()
        .await
        .expect("models");
    assert_eq!(models.models.len(), 1);
    assert_eq!(models.models[0].name, "jev-latest");
    assert_eq!(models.request_id.as_deref(), Some("models-1"));
    assert_eq!(
        models.get("jev-latest").expect("model").release_date,
        "2026-09-15"
    );
    assert_eq!(models.names().collect::<Vec<_>>(), vec!["jev-latest"]);

    let requests = server.requests();
    assert_eq!(requests[0].method, "GET");
    assert_eq!(requests[0].path, "/v1/models");
    assert_eq!(requests[0].header("x-trace"), Some("abc"));
}

#[tokio::test]
async fn retries_a_server_error_then_succeeds() {
    let server = MockServer::start(|_request, index| {
        if index == 0 {
            MockResponse::json(500, serde_json::json!({"detail": "temporary failure"}))
        } else {
            MockResponse::json(200, response_body())
        }
    });
    let client = client_with(&server, fast_retry(2));
    let response = client
        .system_one("text", single_question())
        .await
        .expect("response");
    assert_eq!(response.noul("billing").expect("noul").noul, 0.98);
    assert_eq!(server.request_count(), 2);
}

#[tokio::test]
async fn returns_the_last_error_after_retries_are_exhausted() {
    let server = MockServer::start(|_request, _index| {
        MockResponse::json(503, serde_json::json!({"detail": "still down"}))
            .header("x-typesafe-request-id", "req-err")
    });
    let client = client_with(&server, fast_retry(1));
    let error = client
        .system_one("text", single_question())
        .await
        .expect_err("error");
    match &error {
        Error::Api(api) => {
            assert_eq!(api.status, StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(api.attempts, 2);
            assert_eq!(api.request_id.as_deref(), Some("req-err"));
        }
        other => panic!("unexpected error: {other}"),
    }
    assert_eq!(server.request_count(), 2);
}

#[tokio::test]
async fn honors_retry_after_ms() {
    let server = MockServer::start(|_request, index| {
        if index == 0 {
            MockResponse::json(429, serde_json::json!({"detail": "slow down"}))
                .header("retry-after-ms", "10")
        } else {
            MockResponse::json(200, response_body())
        }
    });
    let policy = RetryPolicy::default()
        .max_retries(1)
        .backoff_initial(Duration::from_secs(1))
        .backoff_max(Duration::from_secs(1))
        .backoff_jitter(0.0)
        .deadline(Some(Duration::from_secs(5)));
    let client = client_with(&server, policy);
    let started = Instant::now();
    client
        .system_one("text", single_question())
        .await
        .expect("response");
    assert!(
        started.elapsed() < Duration::from_millis(500),
        "Retry-After was not honored: {:?}",
        started.elapsed()
    );
    assert_eq!(server.request_count(), 2);
}

#[tokio::test]
async fn validation_errors_are_typed_and_not_retried() {
    let server = MockServer::start(|_request, _index| {
        MockResponse::json(
            422,
            serde_json::json!({
                "detail": [{
                    "loc": ["body", "questions", "urgency", "criteria"],
                    "msg": "List should have at least 1 item",
                    "type": "too_short"
                }]
            }),
        )
        .header("x-typesafe-request-id", "req-422")
    });
    let client = client_with(&server, fast_retry(3));
    let error = client
        .system_one("text", single_question())
        .await
        .expect_err("error");
    match &error {
        Error::Api(api) => {
            assert_eq!(api.status, StatusCode::UNPROCESSABLE_ENTITY);
            assert_eq!(api.kind(), olaya_sdk::ApiErrorKind::UnprocessableEntity);
            assert_eq!(api.attempts, 1);
            let errors = api.validation_errors().expect("validation");
            assert_eq!(errors.len(), 1);
            assert_eq!(errors[0].field_path(), "body.questions.urgency.criteria");
        }
        other => panic!("unexpected error: {other}"),
    }
    assert_eq!(server.request_count(), 1);
}

#[tokio::test]
async fn request_timeouts_are_reported_and_retried_when_enabled() {
    let server = MockServer::start(|_request, _index| {
        MockResponse::json(200, response_body()).delay(Duration::from_millis(300))
    });
    let policy = RetryPolicy::default()
        .max_retries(1)
        .backoff_initial(Duration::from_millis(1))
        .backoff_max(Duration::from_millis(1))
        .backoff_jitter(0.0)
        .deadline(Some(Duration::from_secs(2)));
    let client = Client::builder()
        .api_key("test-key")
        .base_url(server.url())
        .default_model("test-model")
        .retry(policy)
        .timeout(Duration::from_millis(100))
        .build()
        .expect("client");
    let error = client
        .system_one("text", single_question())
        .await
        .expect_err("error");
    assert!(error.is_timeout(), "unexpected error: {error}");
    assert_eq!(server.request_count(), 2);
}

#[tokio::test]
async fn multipart_images_are_sent_with_a_part_reference() {
    let server = MockServer::start(|_request, _index| MockResponse::json(200, response_body()));
    let client = client(&server);
    let image = Image::from_bytes(PNG.to_vec()).expect("png");
    client
        .system_one(
            "Inspect this product: <photo>",
            questions! {
                "damaged" => NoulQuestion::new("Is the product damaged?"),
            },
        )
        .image("<photo>", image)
        .await
        .expect("response");

    let request = &server.requests()[0];
    assert!(
        request
            .header("content-type")
            .expect("content type")
            .starts_with("multipart/form-data; boundary=")
    );
    assert_eq!(request.header("typesafe-danger-zone-images"), Some("true"));
    let body = request.body_text();
    assert!(body.contains("name=\"request\""), "missing request part");
    assert!(body.contains("\"model\":\"test-model\""), "missing model");
    assert!(
        body.contains("\"part\":\"image-0\""),
        "missing part reference"
    );
    assert!(body.contains("name=\"image-0\""), "missing image part");
    assert!(
        body.contains("filename=\"image-0.png\""),
        "missing filename"
    );
    assert!(
        body.contains("Content-Type: image/png"),
        "missing image content type"
    );
}

#[tokio::test]
async fn inline_images_are_sent_as_base64_json() {
    let server = MockServer::start(|_request, _index| MockResponse::json(200, response_body()));
    let client = client(&server);
    let image = Image::from_bytes(PNG.to_vec()).expect("png");
    client
        .system_one(
            "Inspect this product: <photo>",
            questions! {
                "damaged" => NoulQuestion::new("Is the product damaged?"),
            },
        )
        .inline_image("<photo>", image)
        .await
        .expect("response");

    let request = &server.requests()[0];
    assert!(
        request
            .header("content-type")
            .expect("content type")
            .starts_with("application/json")
    );
    let body: serde_json::Value = serde_json::from_slice(&request.body).expect("json");
    assert_eq!(body["assets"]["<photo>"]["type"], "image");
    assert_eq!(body["assets"]["<photo>"]["format"], "png");
    assert!(body["assets"]["<photo>"]["data_base64"].is_string());
    assert!(body["assets"]["<photo>"].get("part").is_none());
}

#[tokio::test]
async fn unauthenticated_clients_omit_the_authorization_header() {
    let server = MockServer::start(|_request, _index| MockResponse::json(200, response_body()));
    let client = Client::builder()
        .allow_unauthenticated(true)
        .base_url(server.url())
        .default_model("test-model")
        .retry(RetryPolicy::disabled())
        .build()
        .expect("client");
    client
        .system_one("text", single_question())
        .await
        .expect("response");
    assert!(server.requests()[0].header("authorization").is_none());
}

#[tokio::test]
async fn extra_body_fields_are_merged() {
    let server = MockServer::start(|_request, _index| MockResponse::json(200, response_body()));
    let client = client(&server);
    client
        .system_one("text", single_question())
        .extra_body("beam_width", 4)
        .await
        .expect("response");
    let body: serde_json::Value = serde_json::from_slice(&server.requests()[0].body).expect("json");
    assert_eq!(body["beam_width"], 4);
}

#[tokio::test]
async fn unknown_answer_kinds_are_preserved() {
    let server = MockServer::start(|_request, _index| {
        MockResponse::json(
            200,
            serde_json::json!({
                "model": "test-model",
                "answers": {"rank": {"type": "ranking", "items": ["a", "b"]}},
                "usage": {"input_tokens": 1, "output_tokens": 1}
            }),
        )
    });
    let client = client(&server);
    let response = client
        .system_one("text", single_question())
        .await
        .expect("response");
    let answer = response.answer("rank").expect("answer");
    assert_eq!(answer.answer_type(), AnswerType::Unknown);
    let (name, raw) = response.unknown_answers().next().expect("unknown");
    assert_eq!(name, "rank");
    assert_eq!(raw["items"][1], "b");
}

#[tokio::test]
async fn the_total_deadline_stops_retries() {
    let server = MockServer::start(|_request, _index| {
        MockResponse::json(500, serde_json::json!({"detail": "boom"}))
    });
    let policy = RetryPolicy::default()
        .max_retries(5)
        .backoff_initial(Duration::from_millis(100))
        .backoff_max(Duration::from_millis(100))
        .backoff_jitter(0.0)
        .deadline(Some(Duration::ZERO));
    let client = client_with(&server, policy);
    let error = client
        .system_one("text", single_question())
        .await
        .expect_err("error");
    assert!(matches!(error, Error::Api(_)));
    assert_eq!(server.request_count(), 1);
}

#[cfg(feature = "gzip")]
#[tokio::test]
async fn gzip_compresses_large_json_bodies() {
    use olaya_sdk::Compression;

    let server = MockServer::start(|_request, _index| MockResponse::json(200, response_body()));
    let client = Client::builder()
        .api_key("test-key")
        .base_url(server.url())
        .default_model("test-model")
        .retry(RetryPolicy::disabled())
        .compression(Compression::Gzip)
        .build()
        .expect("client");
    client
        .system_one("x".repeat(4096), single_question())
        .await
        .expect("response");
    let request = &server.requests()[0];
    assert_eq!(request.header("content-encoding"), Some("gzip"));
    assert!(request.body.starts_with(&[0x1F, 0x8B]));
    assert!(
        request.body.len() < 4096,
        "the compressed body should be smaller"
    );
}

#[tokio::test]
async fn local_servers_can_use_a_base_path() {
    let server = MockServer::start(|_request, _index| MockResponse::json(200, response_body()));
    let client = Client::builder()
        .api_key("test-key")
        .base_url(format!("{}/gateway", server.url()))
        .default_model("test-model")
        .retry(RetryPolicy::disabled())
        .build()
        .expect("client");
    client
        .system_one("text", single_question())
        .await
        .expect("response");
    assert_eq!(server.requests()[0].path, "/gateway/v1/systemone");
}
