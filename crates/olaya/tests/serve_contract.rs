#![cfg(feature = "serve")]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode, header};
use indexmap::IndexMap;
use olaya::serve::router;
use olaya::{
    Answer, Asset, CalibrationMode, Capabilities, DecisionRuntime, ModelMetadata, NoulAnswer,
    Questions, RepoManifest, RuntimeError, RuntimeMetrics, ServeConfig, State, SystemOneResponse,
    Usage,
};
use serde_json::{Value, json};
use tower::ServiceExt;

#[derive(Default)]
struct MockRuntime {
    loaded: Mutex<Vec<String>>,
    assets: Mutex<Vec<Vec<String>>>,
    batches: Mutex<Vec<usize>>,
    metrics: RuntimeMetrics,
}

impl MockRuntime {
    fn known(name: &str) -> bool {
        name == "mock" || name == "mock:latest"
    }

    fn manifest(name: &str) -> RepoManifest {
        RepoManifest {
            schema_version: 1,
            name: name.to_owned(),
            aliases: vec!["mock:latest".to_owned()],
            family: "mock".to_owned(),
            engine: "mock".to_owned(),
            arch: "mock".to_owned(),
            format: "json".to_owned(),
            layout: "mock-v1".to_owned(),
            subfolder: None,
            files: Vec::new(),
            calibration_mode: CalibrationMode::Clamp,
            capabilities: Capabilities::classification(),
            router: None,
            license: "Apache-2.0".to_owned(),
            description: "mock runtime".to_owned(),
            release_date: Some("2026-01-01".to_owned()),
            source: None,
            revision: None,
        }
    }
}

impl DecisionRuntime for MockRuntime {
    fn predict(
        &self,
        model: &str,
        state: &State,
        _questions: &Questions,
    ) -> Result<SystemOneResponse, RuntimeError> {
        if !Self::known(model) {
            return Err(RuntimeError::ModelNotFound {
                model: model.to_owned(),
            });
        }
        let text = state.as_str().unwrap_or_default();
        if text.contains("truncate") {
            return Err(RuntimeError::StateTruncated {
                model: "mock".to_owned(),
            });
        }
        if text.contains("too-long") {
            return Err(RuntimeError::InputTooLong { tokens: 20_000 });
        }
        if text.contains("invalid") {
            return Err(RuntimeError::invalid_request("invalid state"));
        }
        if text.contains("boom") {
            return Err(RuntimeError::inference("mock inference failed"));
        }
        Ok(SystemOneResponse::new(
            "mock",
            IndexMap::from([("q".to_owned(), Answer::Noul(NoulAnswer { noul: 0.75 }))]),
            Usage::new(7, 0),
        ))
    }

    fn predict_batch(
        &self,
        model: &str,
        requests: &[(&State, &Questions)],
    ) -> Vec<Result<SystemOneResponse, RuntimeError>> {
        self.batches
            .lock()
            .expect("batches lock")
            .push(requests.len());
        requests
            .iter()
            .map(|(state, questions)| self.predict(model, state, questions))
            .collect()
    }

    fn predict_with_assets(
        &self,
        model: &str,
        state: &State,
        questions: &Questions,
        assets: &IndexMap<String, Asset>,
    ) -> Result<SystemOneResponse, RuntimeError> {
        self.assets
            .lock()
            .expect("assets lock")
            .push(assets.keys().cloned().collect());
        self.predict(model, state, questions)
    }

    fn list_models(&self) -> Vec<ModelMetadata> {
        vec![
            ModelMetadata::new("mock", "mock runtime", "2026-01-01"),
            ModelMetadata::new("mock:latest", "mock alias", "2026-01-01"),
        ]
    }

    fn resolve(&self, name: &str) -> Option<RepoManifest> {
        Self::known(name).then(|| Self::manifest("mock"))
    }

    fn load(&self, name: &str) -> Result<(), RuntimeError> {
        if !Self::known(name) {
            return Err(RuntimeError::ModelNotFound {
                model: name.to_owned(),
            });
        }
        let mut loaded = self.loaded.lock().expect("loaded lock");
        if !loaded.iter().any(|entry| entry == "mock") {
            loaded.push("mock".to_owned());
        }
        Ok(())
    }

    fn unload(&self, name: &str) -> bool {
        let mut loaded = self.loaded.lock().expect("loaded lock");
        let before = loaded.len();
        loaded.retain(|entry| entry != "mock" && entry != name);
        before != loaded.len()
    }

    fn loaded_models(&self) -> Vec<String> {
        self.loaded.lock().expect("loaded lock").clone()
    }

    fn pull(&self, reference: &str) -> Result<RepoManifest, RuntimeError> {
        if Self::known(reference) {
            Ok(Self::manifest("mock"))
        } else {
            Err(RuntimeError::Unsupported {
                message: "pull is not supported by this mock".to_owned(),
            })
        }
    }

    fn delete(&self, name: &str) -> Result<bool, RuntimeError> {
        Ok(Self::known(name))
    }

    fn runtime_metrics(&self) -> Option<RuntimeMetrics> {
        Some(self.metrics.clone())
    }
}

fn build_with(config: ServeConfig) -> (Router, Arc<MockRuntime>) {
    let runtime = Arc::new(MockRuntime::default());
    (router(runtime.clone(), config), runtime)
}

fn build(api_key: Option<&str>) -> (Router, Arc<MockRuntime>) {
    build_with(ServeConfig {
        api_key: api_key.map(str::to_owned),
        ..ServeConfig::default()
    })
}

async fn send(
    router: &Router,
    method: &str,
    uri: &str,
    body: Option<Value>,
    headers: &[(&str, &str)],
) -> (StatusCode, Vec<u8>) {
    let mut request = Request::builder().method(method).uri(uri);
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    let request = match body {
        Some(body) => request
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&body).expect("json")))
            .expect("request"),
        None => request.body(Body::empty()).expect("request"),
    };
    let response = router.clone().oneshot(request).await.expect("response");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body")
        .to_vec();
    (status, bytes)
}

fn json(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).expect("json body")
}

fn noul_request(model: &str, state: &str) -> Value {
    json!({
        "model": model,
        "state": state,
        "questions": {"q": {"type": "noul", "instructions": "Is this a test?"}}
    })
}

#[tokio::test]
async fn health_models_and_metrics() {
    let (router, runtime) = build(None);
    let (status, body) = send(&router, "GET", "/health", None, &[]).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, b"ok");

    for path in ["/v1/models", "/api/tags"] {
        let (status, body) = send(&router, "GET", path, None, &[]).await;
        assert_eq!(status, StatusCode::OK, "{path}");
        let body = json(&body);
        assert_eq!(body["models"][0]["name"], "mock");
        assert_eq!(body["models"][1]["name"], "mock:latest");
    }

    let (status, body) = send(&router, "GET", "/metrics", None, &[]).await;
    assert_eq!(status, StatusCode::OK);
    let metrics = String::from_utf8_lossy(&body);
    assert!(metrics.contains("olaya_http_requests_total"));
    assert!(metrics.contains("olaya_requests_total"));
    assert!(metrics.contains("olaya_inference_seconds_bucket"));

    let (status, _) = send(&router, "GET", "/no-such-route", None, &[]).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let snapshot = runtime.runtime_metrics().expect("mock metrics").snapshot();
    assert!(snapshot.http_requests_total >= 4);
    assert_eq!(snapshot.http_inflight, 0);
}

#[tokio::test]
async fn responses_carry_a_request_id() {
    let (router, _) = build(None);
    let request = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .expect("request");
    let response = router.clone().oneshot(request).await.expect("response");
    let generated = response
        .headers()
        .get("x-typesafe-request-id")
        .expect("generated request id")
        .to_str()
        .expect("ascii request id");
    assert!(!generated.is_empty());

    let request = Request::builder()
        .method("GET")
        .uri("/health")
        .header("x-typesafe-request-id", "client-123")
        .body(Body::empty())
        .expect("request");
    let response = router.clone().oneshot(request).await.expect("response");
    assert_eq!(
        response
            .headers()
            .get("x-typesafe-request-id")
            .expect("request id")
            .to_str()
            .expect("ascii request id"),
        "client-123"
    );
}

#[tokio::test]
async fn serve_refuses_a_non_loopback_bind_without_a_key() {
    let (_, runtime) = build(None);
    let runtime: Arc<dyn DecisionRuntime> = runtime;
    let config = ServeConfig {
        host: "0.0.0.0".to_owned(),
        ..ServeConfig::default()
    };
    let error = olaya::serve::serve(runtime, config)
        .await
        .expect_err("non-loopback without a key must be refused");
    assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
}

#[tokio::test]
async fn systemone_returns_the_typed_answer() {
    let (router, _) = build(None);
    let (status, body) = send(
        &router,
        "POST",
        "/v1/systemone",
        Some(noul_request("mock:latest", "hello")),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = json(&body);
    assert_eq!(body["model"], "mock");
    assert_eq!(body["answers"]["q"]["type"], "noul");
    assert_eq!(body["answers"]["q"]["noul"], 0.75);
    assert_eq!(body["usage"]["input_tokens"], 7);
    assert_eq!(body["usage"]["output_tokens"], 0);
}

#[tokio::test]
async fn systemone_maps_runtime_errors_to_the_typesafe_shape() {
    let (router, _) = build(None);

    let (status, body) = send(
        &router,
        "POST",
        "/v1/systemone",
        Some(noul_request("mock", "truncate me")),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let body = json(&body);
    assert_eq!(body["detail"][0]["type"], "state_truncated");
    assert_eq!(body["detail"][0]["ctx"]["model"], "mock");

    let (status, body) = send(
        &router,
        "POST",
        "/v1/systemone",
        Some(noul_request("mock", "too-long")),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json(&body)["detail"][0]["type"], "too_long");

    let (status, body) = send(
        &router,
        "POST",
        "/v1/systemone",
        Some(noul_request("mock", "invalid")),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json(&body)["detail"], "invalid state");

    let (status, body) = send(
        &router,
        "POST",
        "/v1/systemone",
        Some(noul_request("mock", "boom")),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(
        json(&body)["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("inference failed")
    );

    let (status, body) = send(
        &router,
        "POST",
        "/v1/systemone",
        Some(noul_request("missing", "hi")),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(
        json(&body)["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("unknown model")
    );

    let (status, body) = send(&router, "POST", "/v1/systemone", Some(json!({})), &[]).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(
        json(&body)["detail"]
            .as_array()
            .is_some_and(|d| !d.is_empty())
    );
}

#[tokio::test]
async fn bearer_auth_protects_everything_but_health() {
    let (router, _) = build(Some("secret"));
    let (status, _) = send(&router, "GET", "/health", None, &[]).await;
    assert_eq!(status, StatusCode::OK);

    for path in ["/v1/models", "/api/tags"] {
        let (status, _) = send(&router, "GET", path, None, &[]).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{path}");
    }
    let (status, _) = send(
        &router,
        "GET",
        "/v1/models",
        None,
        &[("authorization", "Bearer wrong")],
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let (status, _) = send(
        &router,
        "GET",
        "/v1/models",
        None,
        &[("authorization", "Bearer secret")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = send(
        &router,
        "POST",
        "/v1/systemone",
        Some(noul_request("mock", "hi")),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn model_management_endpoints_track_load_and_unload() {
    let (router, _) = build(None);
    let (status, body) = send(&router, "GET", "/api/ps", None, &[]).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json(&body)["models"], json!([]));

    let (status, body) = send(
        &router,
        "POST",
        "/api/load",
        Some(json!({"model": "mock"})),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json(&body)["loaded"], true);

    let (_, body) = send(&router, "GET", "/api/ps", None, &[]).await;
    assert_eq!(json(&body)["models"], json!(["mock"]));

    let (status, body) = send(
        &router,
        "POST",
        "/api/unload",
        Some(json!({"model": "mock"})),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json(&body)["unloaded"], true);
    let (_, body) = send(
        &router,
        "POST",
        "/api/unload",
        Some(json!({"model": "mock"})),
        &[],
    )
    .await;
    assert_eq!(json(&body)["unloaded"], false);

    let (status, _) = send(
        &router,
        "POST",
        "/api/load",
        Some(json!({"model": "missing"})),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, body) = send(
        &router,
        "POST",
        "/api/show",
        Some(json!({"model": "mock"})),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json(&body)["name"], "mock");

    let (status, body) = send(
        &router,
        "POST",
        "/api/pull",
        Some(json!({"model": "mock"})),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json(&body)["name"], "mock");
    let (status, _) = send(
        &router,
        "POST",
        "/api/pull",
        Some(json!({"model": "missing"})),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);

    let (status, _) = send(
        &router,
        "DELETE",
        "/api/delete",
        Some(json!({"model": "mock"})),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = send(
        &router,
        "DELETE",
        "/api/delete",
        Some(json!({"model": "missing"})),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, _) = send(
        &router,
        "POST",
        "/api/show",
        Some(json!({"model": "missing"})),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn box_and_assets_require_the_danger_zone_header() {
    let (router, runtime) = build(None);
    let box_request = json!({
        "model": "mock",
        "state": "hi",
        "questions": {"q": {"type": "box", "instructions": "find it"}}
    });
    let (status, _) = send(
        &router,
        "POST",
        "/v1/systemone",
        Some(box_request.clone()),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = send(
        &router,
        "POST",
        "/v1/systemone",
        Some(box_request),
        &[("typesafe-danger-zone-images", "true")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let asset_request = json!({
        "model": "mock",
        "state": "hi",
        "questions": {"q": {"type": "noul", "instructions": "ok?"}},
        "assets": {"receipt": {"type": "image", "format": "png", "data_base64": "aGk="}}
    });
    let (status, _) = send(
        &router,
        "POST",
        "/v1/systemone",
        Some(asset_request.clone()),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = send(
        &router,
        "POST",
        "/v1/systemone",
        Some(asset_request),
        &[("typesafe-danger-zone-images", "true")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        runtime.assets.lock().expect("assets").last(),
        Some(&vec!["receipt".to_owned()])
    );

    let boundary = "olaya-test-boundary";
    let request_body = serde_json::to_string(&json!({
        "model": "mock",
        "state": "hi",
        "questions": {"q": {"type": "noul", "instructions": "ok?"}}
    }))
    .expect("request");
    let multipart = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"request\"\r\n\
         Content-Type: application/json\r\n\r\n{request_body}\r\n\
         --{boundary}\r\nContent-Disposition: form-data; name=\"receipt\"; filename=\"r.png\"\r\n\
         Content-Type: image/png\r\n\r\nPNGDATA\r\n--{boundary}--\r\n"
    );
    let content_type = format!("multipart/form-data; boundary={boundary}");
    let request = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header(header::CONTENT_TYPE, &content_type)
        .header("typesafe-danger-zone-images", "true")
        .body(Body::from(multipart.clone()))
        .expect("request");
    let response = router.clone().oneshot(request).await.expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        runtime.assets.lock().expect("assets").last(),
        Some(&vec!["receipt".to_owned()])
    );

    let request = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header(header::CONTENT_TYPE, &content_type)
        .body(Body::from(multipart))
        .expect("request");
    let response = router.clone().oneshot(request).await.expect("response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let no_request_part = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"receipt\"\r\n\
         Content-Type: image/png\r\n\r\nPNGDATA\r\n--{boundary}--\r\n"
    );
    let request = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header(header::CONTENT_TYPE, &content_type)
        .header("typesafe-danger-zone-images", "true")
        .body(Body::from(no_request_part))
        .expect("request");
    let response = router.clone().oneshot(request).await.expect("response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_requests_share_a_batch() {
    let (router, runtime) = build_with(ServeConfig {
        max_batch: 8,
        max_wait: Duration::from_millis(50),
        ..ServeConfig::default()
    });
    let mut handles = Vec::new();
    for _ in 0..8 {
        let router = router.clone();
        handles.push(tokio::spawn(async move {
            send(
                &router,
                "POST",
                "/v1/systemone",
                Some(noul_request("mock", "hello")),
                &[],
            )
            .await
            .0
        }));
    }
    for handle in handles {
        assert_eq!(handle.await.expect("join"), StatusCode::OK);
    }
    let batches = runtime.batches.lock().expect("batches").clone();
    assert!(
        batches.iter().any(|&size| size >= 4),
        "expected a shared batch, saw {batches:?}"
    );
}
