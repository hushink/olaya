use std::sync::Arc;
use std::time::{Duration, Instant};

use olaya_core::{DecisionRuntime, RuntimeMetrics};

use crate::serve::batch::{BatchConfig, Batcher};

/// Server configuration.
#[derive(Clone)]
pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    /// When set, every route except `/health` requires `Authorization: Bearer`.
    pub api_key: Option<String>,
    /// Allow binding a non-loopback address without an API key.
    pub allow_unauthenticated: bool,
    pub max_body_bytes: usize,
    /// Maximum requests waiting in the batch channel before backpressure.
    pub max_queue: usize,
    /// Requests collected into one `predict_batch` call. 1 disables batching.
    pub max_batch: usize,
    /// How long the first request of a batch waits for companions.
    pub max_wait: Duration,
    /// Browser origins allowed to call the API. Empty disables CORS entirely
    /// (the safe default for a local, unauthenticated server).
    pub cors_origins: Vec<String>,
    /// Additional `Host` names accepted by the server. Empty applies the
    /// default policy: a loopback bind only accepts loopback host names,
    /// while a non-loopback bind accepts any Host.
    pub allowed_hosts: Vec<String>,
    /// Maximum wall-clock time for one HTTP request, including body reads.
    /// `Duration::ZERO` disables the timeout.
    pub request_timeout: Duration,
    /// Maximum concurrent model-management operations (`/api/pull`,
    /// `/api/load`).
    pub max_operations: usize,
}

impl std::fmt::Debug for ServeConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ServeConfig")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("allow_unauthenticated", &self.allow_unauthenticated)
            .field("max_body_bytes", &self.max_body_bytes)
            .field("max_queue", &self.max_queue)
            .field("max_batch", &self.max_batch)
            .field("max_wait", &self.max_wait)
            .field("cors_origins", &self.cors_origins)
            .field("allowed_hosts", &self.allowed_hosts)
            .field("request_timeout", &self.request_timeout)
            .field("max_operations", &self.max_operations)
            .finish()
    }
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_owned(),
            port: 11435,
            api_key: None,
            allow_unauthenticated: false,
            max_body_bytes: olaya_core::MAX_BODY_BYTES,
            max_queue: 64,
            max_batch: 32,
            max_wait: Duration::from_millis(2),
            cors_origins: Vec::new(),
            allowed_hosts: Vec::new(),
            request_timeout: Duration::from_secs(600),
            max_operations: 4,
        }
    }
}

impl ServeConfig {
    pub fn is_loopback(&self) -> bool {
        self.host.eq_ignore_ascii_case("localhost")
            || self
                .host
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback())
    }

    /// Validate limits and optional credentials before binding a socket.
    pub fn validate(&self) -> Result<(), String> {
        if self.port == 0 {
            return Err("[serve].port must be between 1 and 65535".to_owned());
        }
        if self.max_batch == 0 {
            return Err("[serve].max_batch must be at least 1".to_owned());
        }
        if self.max_queue == 0 {
            return Err("[serve].max_queue must be at least 1".to_owned());
        }
        if self.max_body_bytes == 0 {
            return Err("[serve].max_body_bytes must be at least 1".to_owned());
        }
        // Upper bounds prevent accidental overflow in `Instant + duration` and
        // absurd allocations from a hand-edited configuration.
        if self.max_batch > 65_536 {
            return Err("[serve].max_batch must be at most 65536".to_owned());
        }
        if self.max_queue > 1_000_000 {
            return Err("[serve].max_queue must be at most 1000000".to_owned());
        }
        if self.max_body_bytes > 1024 * 1024 * 1024 {
            return Err("[serve].max_body_bytes must be at most 1 GiB".to_owned());
        }
        if self.max_wait > Duration::from_secs(3600) {
            return Err("[serve].batch_wait_ms must be at most 3600000".to_owned());
        }
        if let Some(key) = &self.api_key {
            let key = key.trim();
            if key.is_empty() {
                return Err("[serve].api_key must not be empty".to_owned());
            }
            if key
                .chars()
                .any(|character| character.is_whitespace() || character.is_control())
            {
                return Err(
                    "[serve].api_key must not contain whitespace or control characters".to_owned(),
                );
            }
        }
        for origin in &self.cors_origins {
            let origin = origin.trim();
            if origin.is_empty()
                || origin.chars().any(char::is_control)
                || (origin != "*"
                    && !origin.starts_with("http://")
                    && !origin.starts_with("https://"))
            {
                return Err(format!(
                    "[serve].cors_origins contains an invalid origin: {origin:?}"
                ));
            }
        }
        for host in &self.allowed_hosts {
            let host = host.trim();
            if host.is_empty()
                || host.chars().any(char::is_control)
                || host.contains(['/', '?', '#'])
            {
                return Err(format!(
                    "[serve].allowed_hosts contains an invalid host: {host:?}"
                ));
            }
        }
        if self.request_timeout > Duration::from_secs(86_400) {
            return Err("[serve].request_timeout_secs must be at most 86400".to_owned());
        }
        if self.max_operations == 0 || self.max_operations > 1_024 {
            return Err("[serve].max_operations must be between 1 and 1024".to_owned());
        }
        Ok(())
    }
}

/// Shared handler state.
pub struct AppState {
    pub runtime: Arc<dyn DecisionRuntime>,
    pub config: ServeConfig,
    pub batcher: Option<Arc<Batcher>>,
    pub metrics: RuntimeMetrics,
    pub started: Instant,
    pub operations: Arc<tokio::sync::Semaphore>,
}

impl AppState {
    pub fn new(runtime: Arc<dyn DecisionRuntime>, config: ServeConfig) -> Arc<Self> {
        let metrics = runtime.runtime_metrics().unwrap_or_default();
        let batcher = (config.max_batch > 1)
            .then(|| {
                tokio::runtime::Handle::try_current().ok().map(|_| {
                    Batcher::spawn_with_queue(
                        runtime.clone(),
                        BatchConfig {
                            max_batch: config.max_batch,
                            max_wait: config.max_wait,
                        },
                        config.max_queue,
                        metrics.clone(),
                    )
                })
            })
            .flatten();
        let operations = Arc::new(tokio::sync::Semaphore::new(
            config.max_operations.clamp(1, 1_024),
        ));
        Arc::new(Self {
            runtime,
            config,
            batcher,
            metrics,
            started: Instant::now(),
            operations,
        })
    }
}
