use std::sync::Arc;
use std::time::Instant;

use axum::body::to_bytes;
use axum::extract::{DefaultBodyLimit, FromRequest, Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
#[cfg(feature = "serve-multipart")]
use base64::Engine;
use indexmap::IndexMap;
#[cfg(feature = "serve-multipart")]
use olaya_core::AssetFormat;
use olaya_core::{
    Asset, AssetKind, DANGER_ZONE_IMAGES_HEADER, DANGER_ZONE_LONG_CONTEXT_HEADER, DecisionRuntime,
    ModelMetadataList, Question, Questions, REQUEST_ID_HEADER, RepoManifest, SystemOneRequest,
    SystemOneResponse, new_request_id, parse_request_slice,
};
use serde::Deserialize;
use serde_json::json;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::timeout::TimeoutLayer;

use crate::serve::error::ApiError;
use crate::serve::state::{AppState, ServeConfig};

#[derive(Debug, Deserialize)]
struct ModelRequest {
    model: String,
}

#[derive(Debug, Deserialize)]
struct PullRequest {
    model: String,
    #[serde(default)]
    #[allow(dead_code)]
    stream: Option<bool>,
}

/// Build the HTTP router for a runtime.
pub fn router(runtime: Arc<dyn DecisionRuntime>, config: ServeConfig) -> Router {
    let state = AppState::new(runtime, config);
    let max_body = state.config.max_body_bytes;

    let mut router = Router::new()
        .route("/health", get(health))
        .route("/v1/systemone", post(systemone))
        .route("/v1/models", get(v1_models))
        .route("/api/tags", get(api_tags))
        .route("/api/ps", get(api_ps))
        .route("/api/show", post(api_show))
        .route("/api/load", post(api_load))
        .route("/api/unload", post(api_unload))
        .route("/api/pull", post(api_pull))
        .route("/api/delete", delete(api_delete))
        .route("/metrics", get(metrics))
        .layer(DefaultBodyLimit::max(max_body));
    if !state.config.request_timeout.is_zero() {
        // Bounds slow or stalled clients, including slow body reads.
        router = router.layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            state.config.request_timeout,
        ));
    }
    let router = router
        .layer(middleware::from_fn_with_state(state.clone(), auth))
        .layer(middleware::from_fn_with_state(state.clone(), host_guard));
    // CORS is opt-in. A permissive layer on a loopback server without an API
    // key lets any web page invoke model management and inference endpoints.
    let router = match cors_layer(&state.config.cors_origins) {
        Some(layer) => router.layer(layer),
        None => router,
    };
    router
        .layer(middleware::from_fn_with_state(state.clone(), observe))
        .with_state(state)
}

fn cors_layer(origins: &[String]) -> Option<CorsLayer> {
    if origins.is_empty() {
        return None;
    }
    let origins = origins
        .iter()
        .filter_map(|origin| HeaderValue::from_str(origin.trim()).ok())
        .collect::<Vec<_>>();
    if origins.is_empty() {
        return None;
    }
    Some(
        CorsLayer::new()
            .allow_origin(AllowOrigin::list(origins))
            .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
            .allow_headers([
                header::ACCEPT,
                header::AUTHORIZATION,
                header::CONTENT_TYPE,
                HeaderName::from_static(REQUEST_ID_HEADER),
                HeaderName::from_static(DANGER_ZONE_IMAGES_HEADER),
                HeaderName::from_static(DANGER_ZONE_LONG_CONTEXT_HEADER),
            ])
            .expose_headers([HeaderName::from_static(REQUEST_ID_HEADER)]),
    )
}

/// Bind and serve until the process is stopped.
pub async fn serve(runtime: Arc<dyn DecisionRuntime>, config: ServeConfig) -> std::io::Result<()> {
    config
        .validate()
        .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidInput, message))?;
    if config.api_key.is_none() && !config.allow_unauthenticated && !config.is_loopback() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "refusing to listen on a non-loopback address without OLAYA_API_KEY; \
             pass --allow-unauthenticated to override",
        ));
    }
    let listener = tokio::net::TcpListener::bind((config.host.as_str(), config.port)).await?;
    tracing::info!(
        address = %listener.local_addr().map(|address| address.to_string()).unwrap_or_else(|_| format!("{}:{}", config.host, config.port)),
        max_batch = config.max_batch,
        batch_wait_ms = config.max_wait.as_millis() as u64,
        authenticated = config.api_key.is_some(),
        "olaya HTTP server listening"
    );
    axum::serve(listener, router(runtime, config)).await
}

async fn health() -> &'static str {
    "ok"
}

async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    #[cfg(feature = "serve-metrics")]
    {
        (
            [(
                header::CONTENT_TYPE,
                "text/plain; version=0.0.4; charset=utf-8",
            )],
            state.metrics.render_prometheus(),
        )
            .into_response()
    }
    #[cfg(not(feature = "serve-metrics"))]
    {
        let _ = state;
        StatusCode::NOT_IMPLEMENTED.into_response()
    }
}

async fn v1_models(State(state): State<Arc<AppState>>) -> Json<ModelMetadataList> {
    Json(ModelMetadataList::new(state.runtime.list_models()))
}

async fn api_tags(State(state): State<Arc<AppState>>) -> Json<ModelMetadataList> {
    Json(ModelMetadataList::new(state.runtime.list_models()))
}

async fn api_ps(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(json!({ "models": state.runtime.loaded_models() }))
}

async fn api_show(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ModelRequest>,
) -> Result<Json<RepoManifest>, ApiError> {
    state
        .runtime
        .resolve(&request.model)
        .map(Json)
        .ok_or_else(|| ApiError::not_found(format!("unknown model: {}", request.model)))
}

async fn api_load(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ModelRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let permit = state
        .operations
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ApiError::internal("operation limiter is closed"))?;
    let runtime = state.runtime.clone();
    let model = request.model;
    let loaded = model.clone();
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        runtime.load(&loaded)
    })
    .await
    .map_err(|error| ApiError::internal(format!("load task failed: {error}")))?
    .map_err(ApiError::runtime)?;
    Ok(Json(json!({ "model": model, "loaded": true })))
}

async fn api_unload(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ModelRequest>,
) -> Json<serde_json::Value> {
    let unloaded = state.runtime.unload(&request.model);
    Json(json!({ "model": request.model, "unloaded": unloaded }))
}

async fn api_pull(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PullRequest>,
) -> Result<Json<RepoManifest>, ApiError> {
    let permit = state
        .operations
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ApiError::internal("operation limiter is closed"))?;
    let runtime = state.runtime.clone();
    let model = request.model;
    let manifest = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        runtime.pull(&model)
    })
    .await
    .map_err(|error| ApiError::internal(format!("pull task failed: {error}")))?
    .map_err(ApiError::runtime)?;
    Ok(Json(manifest))
}

async fn api_delete(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ModelRequest>,
) -> Result<StatusCode, ApiError> {
    let runtime = state.runtime.clone();
    let model = request.model;
    let deleted = tokio::task::spawn_blocking(move || runtime.delete(&model))
        .await
        .map_err(|error| ApiError::internal(format!("delete task failed: {error}")))?
        .map_err(ApiError::runtime)?;
    if deleted {
        Ok(StatusCode::OK)
    } else {
        Err(ApiError::not_found("model not found"))
    }
}

async fn systemone(
    State(state): State<Arc<AppState>>,
    request: Request,
) -> Result<Json<SystemOneResponse>, ApiError> {
    let danger = request
        .headers()
        .get("typesafe-danger-zone-images")
        .and_then(|value| value.to_str().ok())
        == Some("true");
    let is_multipart = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("multipart/form-data"));

    let (mut parsed, part_assets) = if is_multipart {
        read_multipart(request, &state).await?
    } else {
        let bytes = to_bytes(request.into_body(), state.config.max_body_bytes)
            .await
            .map_err(|error| ApiError::bad_request(format!("reading request body: {error}")))?;
        (
            parse_request_slice(&bytes).map_err(ApiError::validation)?,
            IndexMap::new(),
        )
    };

    if !part_assets.is_empty() {
        let mut assets = parsed.assets.take().unwrap_or_default();
        for (part_name, part_asset) in part_assets {
            // The JSON `assets` entry refers to a multipart part by its
            // `part` field. Fill that entry with the bytes we received
            // instead of inserting a second asset under the part name.
            let mut referenced = false;
            for asset in assets.values_mut() {
                if asset.part.as_deref() == Some(part_name.as_str()) {
                    asset.format = part_asset.format;
                    asset.data_base64 = part_asset.data_base64.clone();
                    referenced = true;
                }
            }
            if !referenced {
                if assets.contains_key(&part_name) {
                    return Err(ApiError::bad_request(format!(
                        "multipart part {part_name:?} collides with an asset name"
                    )));
                }
                assets.insert(part_name, part_asset);
            }
        }
        parsed.assets = Some(assets);
    }

    let has_assets = parsed.assets.is_some();
    let has_box = parsed
        .questions
        .values()
        .any(|question| matches!(question, Question::Box(_)));
    if (has_assets || has_box) && !danger {
        return Err(ApiError::bad_request(
            "assets and box questions require the TypeSafe-Danger-Zone-Images: true header",
        ));
    }

    let questions = Questions::from(parsed.questions);
    let assets = parsed.assets.take().unwrap_or_default();
    let runtime = state.runtime.clone();
    let batcher = state.batcher.clone();
    let model = parsed.model;
    let state_value = parsed.state;

    let response = if assets.is_empty() {
        if let Some(batcher) = batcher {
            batcher
                .predict(&model, state_value, questions)
                .await
                .map_err(ApiError::runtime)?
        } else {
            tokio::task::spawn_blocking(move || runtime.predict(&model, &state_value, &questions))
                .await
                .map_err(|error| ApiError::internal(format!("decision task failed: {error}")))?
                .map_err(ApiError::runtime)?
        }
    } else {
        tokio::task::spawn_blocking(move || {
            runtime.predict_with_assets(&model, &state_value, &questions, &assets)
        })
        .await
        .map_err(|error| ApiError::internal(format!("decision task failed: {error}")))?
        .map_err(ApiError::runtime)?
    };

    Ok(Json(response))
}

#[cfg(feature = "serve-multipart")]
async fn read_multipart(
    request: Request,
    state: &Arc<AppState>,
) -> Result<(SystemOneRequest, IndexMap<String, Asset>), ApiError> {
    let mut multipart = axum::extract::Multipart::from_request(request, state)
        .await
        .map_err(|error| ApiError::bad_request(format!("multipart: {error}")))?;
    let mut request_body = None;
    let mut assets = IndexMap::new();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| ApiError::bad_request(format!("multipart: {error}")))?
    {
        let name = field.name().unwrap_or_default().to_owned();
        if name.is_empty() {
            return Err(ApiError::bad_request("multipart part has no name"));
        }
        let content_type = field.content_type().unwrap_or_default().to_owned();
        let bytes = field
            .bytes()
            .await
            .map_err(|error| ApiError::bad_request(format!("multipart: {error}")))?;
        if name == "request" {
            if request_body.is_some() {
                return Err(ApiError::bad_request(
                    "multipart request has more than one `request` part",
                ));
            }
            request_body = Some(bytes.to_vec());
        } else {
            if assets.contains_key(&name) {
                return Err(ApiError::bad_request(format!(
                    "multipart has more than one `{name}` part"
                )));
            }
            // Keep the bytes usable by runtimes that implement
            // `predict_with_assets`; the previous implementation dropped them.
            let format = image_format(&content_type, &bytes).ok_or_else(|| {
                ApiError::bad_request(format!(
                    "multipart part `{name}` is not a supported PNG/JPEG image"
                ))
            })?;
            assets.insert(
                name.clone(),
                Asset {
                    kind: AssetKind::Image,
                    format: Some(format),
                    data_base64: Some(base64::engine::general_purpose::STANDARD.encode(&bytes)),
                    part: Some(name),
                },
            );
        }
    }

    let bytes = request_body
        .ok_or_else(|| ApiError::bad_request("multipart request has no `request` part"))?;
    let parsed = parse_request_slice(&bytes).map_err(ApiError::validation)?;
    Ok((parsed, assets))
}

#[cfg(not(feature = "serve-multipart"))]
async fn read_multipart(
    _request: Request,
    _state: &Arc<AppState>,
) -> Result<(SystemOneRequest, IndexMap<String, Asset>), ApiError> {
    Err(ApiError::unsupported(
        "multipart requests are not compiled in; enable the `multipart` feature",
    ))
}

#[cfg(feature = "serve-multipart")]
fn image_format(content_type: &str, bytes: &[u8]) -> Option<AssetFormat> {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Some(AssetFormat::Png);
    }
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some(AssetFormat::Jpeg);
    }
    let media_type = content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    match media_type.as_str() {
        "image/png" => Some(AssetFormat::Png),
        "image/jpeg" | "image/jpg" => Some(AssetFormat::Jpeg),
        _ => None,
    }
}

async fn observe(State(state): State<Arc<AppState>>, request: Request, next: Next) -> Response {
    let started = Instant::now();
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    let request_id = resolve_request_id(request.headers());
    state.metrics.http_request_started();
    let mut response = next.run(request).await;
    let elapsed = started.elapsed().as_secs_f64();
    let status = response.status();
    state.metrics.http_request_finished(
        status.is_client_error() || status.is_server_error(),
        elapsed,
    );
    if let Ok(value) = HeaderValue::from_str(&request_id) {
        response
            .headers_mut()
            .insert(header::HeaderName::from_static(REQUEST_ID_HEADER), value);
    }
    if status.is_server_error() {
        tracing::warn!(
            request_id = %request_id,
            method = %method,
            path = %path,
            status = status.as_u16(),
            elapsed_ms = elapsed * 1000.0,
            "http request failed"
        );
    } else {
        tracing::debug!(
            request_id = %request_id,
            method = %method,
            path = %path,
            status = status.as_u16(),
            elapsed_ms = elapsed * 1000.0,
            "http request"
        );
    }
    response
}

fn resolve_request_id(headers: &HeaderMap) -> String {
    headers
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty() && value.len() <= 128)
        .map(str::to_owned)
        .unwrap_or_else(new_request_id)
}

#[cfg(feature = "serve-auth")]
async fn auth(State(state): State<Arc<AppState>>, request: Request, next: Next) -> Response {
    if request.uri().path() == "/health" {
        return next.run(request).await;
    }
    if let Some(key) = &state.config.api_key
        && !bearer_ok(request.headers(), key)
    {
        return ApiError::unauthorized().into_response();
    }
    next.run(request).await
}

#[cfg(not(feature = "serve-auth"))]
async fn auth(State(_state): State<Arc<AppState>>, request: Request, next: Next) -> Response {
    next.run(request).await
}

/// Reject DNS-rebinding requests. A browser can resolve an attacker-controlled
/// name to 127.0.0.1, making its requests same-origin; checking `Host` blocks
/// that even when the local server has no API key.
async fn host_guard(State(state): State<Arc<AppState>>, request: Request, next: Next) -> Response {
    if let Some(host) = request.headers().get(header::HOST) {
        let host = host.to_str().unwrap_or_default();
        if !host_allowed(&state.config, host) {
            return ApiError::bad_request("unexpected Host header").into_response();
        }
    }
    next.run(request).await
}

fn host_allowed(config: &ServeConfig, host: &str) -> bool {
    let host = host_name(host);
    if !config.allowed_hosts.is_empty() {
        return config.allowed_hosts.iter().any(|allowed| {
            let allowed = allowed.trim();
            allowed == "*" || host_name(allowed).eq_ignore_ascii_case(host)
        });
    }
    if !config.is_loopback() {
        return true;
    }
    let bare = host
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host);
    bare.parse::<std::net::IpAddr>()
        .is_ok_and(|address| address.is_loopback())
        || host.eq_ignore_ascii_case("localhost")
        || host.eq_ignore_ascii_case(host_name(&config.host))
}

fn host_name(host: &str) -> &str {
    let host = host.trim();
    if let Some(rest) = host.strip_prefix('[') {
        if let Some(end) = rest.find(']') {
            return &host[..end + 2];
        }
        return host;
    }
    host.split(':').next().unwrap_or(host)
}

fn bearer_ok(headers: &HeaderMap, key: &str) -> bool {
    let Some(value) = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    else {
        return false;
    };
    let Some((scheme, token)) = value.trim().split_once(' ') else {
        return false;
    };
    let (given, expected) = (token.trim().as_bytes(), key.as_bytes());
    scheme.eq_ignore_ascii_case("bearer")
        && given.len() == expected.len()
        && given
            .iter()
            .zip(expected)
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0
}
