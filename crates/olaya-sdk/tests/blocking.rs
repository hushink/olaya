#![cfg(feature = "blocking")]

mod support;

use std::time::Duration;

use olaya_sdk::blocking::Client;
use olaya_sdk::{Image, NoulQuestion, RetryPolicy, questions};
use support::{MockResponse, MockServer};

const PNG: &[u8] = &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 1, 2, 3];

fn response_body() -> serde_json::Value {
    serde_json::json!({
        "model": "test-model",
        "answers": {"billing": {"type": "noul", "noul": 0.98}},
        "usage": {"input_tokens": 10, "output_tokens": 2}
    })
}

fn client(server: &MockServer, retry: RetryPolicy) -> Client {
    Client::builder()
        .api_key("test-key")
        .base_url(server.url())
        .default_model("test-model")
        .retry(retry)
        .build()
        .expect("client")
}

#[test]
fn blocking_system_one_round_trips() {
    let server = MockServer::start(|_request, _index| {
        MockResponse::json(200, response_body()).header("x-typesafe-request-id", "req-blocking")
    });
    let client = client(&server, RetryPolicy::disabled());
    let response = client
        .system_one(
            "I was charged twice.",
            questions! {
                "billing" => NoulQuestion::new("Is this about billing?"),
            },
        )
        .send()
        .expect("response");
    assert_eq!(response.request_id.as_deref(), Some("req-blocking"));
    assert_eq!(response.noul("billing").expect("noul").noul, 0.98);

    let request = &server.requests()[0];
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/systemone");
    assert_eq!(request.header("authorization"), Some("Bearer test-key"));
    let body: serde_json::Value = serde_json::from_slice(&request.body).expect("json");
    assert_eq!(body["model"], "test-model");
}

#[test]
fn blocking_client_retries_server_errors() {
    let server = MockServer::start(|_request, index| {
        if index == 0 {
            MockResponse::json(500, serde_json::json!({"detail": "temporary"}))
        } else {
            MockResponse::json(200, response_body())
        }
    });
    let policy = RetryPolicy::default()
        .max_retries(2)
        .backoff_initial(Duration::from_millis(1))
        .backoff_max(Duration::from_millis(2))
        .backoff_jitter(0.0)
        .deadline(Some(Duration::from_secs(2)));
    let client = client(&server, policy);
    client
        .system_one(
            "text",
            questions! {
                "billing" => NoulQuestion::new("Is this about billing?"),
            },
        )
        .send()
        .expect("response");
    assert_eq!(server.request_count(), 2);
}

#[test]
fn blocking_client_lists_models() {
    let server = MockServer::start(|_request, _index| {
        MockResponse::json(
            200,
            serde_json::json!({
                "models": [{
                    "name": "jev-latest",
                    "description": "General-purpose system one model.",
                    "release_date": "2026-09-15"
                }]
            }),
        )
    });
    let client = client(&server, RetryPolicy::disabled());
    let models = client.list_models().expect("models");
    assert_eq!(models.names().collect::<Vec<_>>(), vec!["jev-latest"]);
    assert_eq!(server.requests()[0].path, "/v1/models");
}

#[test]
fn blocking_client_sends_multipart_images() {
    let server = MockServer::start(|_request, _index| MockResponse::json(200, response_body()));
    let client = client(&server, RetryPolicy::disabled());
    let image = Image::from_bytes(PNG.to_vec()).expect("png");
    client
        .system_one(
            "Inspect: <photo>",
            questions! {
                "damaged" => NoulQuestion::new("Is it damaged?"),
            },
        )
        .image("<photo>", image)
        .send()
        .expect("response");

    let request = &server.requests()[0];
    assert!(
        request
            .header("content-type")
            .expect("content type")
            .starts_with("multipart/form-data; boundary=")
    );
    let body = request.body_text();
    assert!(body.contains("name=\"request\""));
    assert!(body.contains("\"part\":\"image-0\""));
    assert!(body.contains("name=\"image-0\""));
    assert!(body.contains("Content-Type: image/png"));
}
