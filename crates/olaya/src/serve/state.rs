use std::sync::Arc;
use std::time::{Duration, Instant};

use olaya_core::{DecisionRuntime, RuntimeMetrics};

use crate::serve::batch::{BatchConfig, Batcher};

/// Server configuration.
#[derive(Clone, Debug)]
pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    /// When set, every route except `/health` requires `Authorization: Bearer`.
    pub api_key: Option<String>,
    /// Allow binding a non-loopback address without an API key.
    pub allow_unauthenticated: bool,
    pub max_body_bytes: usize,
    pub max_queue: usize,
    /// Requests collected into one `predict_batch` call. 1 disables batching.
    pub max_batch: usize,
    /// How long the first request of a batch waits for companions.
    pub max_wait: Duration,
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
}

/// Shared handler state.
pub struct AppState {
    pub runtime: Arc<dyn DecisionRuntime>,
    pub config: ServeConfig,
    pub batcher: Option<Arc<Batcher>>,
    pub metrics: RuntimeMetrics,
    pub started: Instant,
}

impl AppState {
    pub fn new(runtime: Arc<dyn DecisionRuntime>, config: ServeConfig) -> Arc<Self> {
        let metrics = runtime.runtime_metrics().unwrap_or_default();
        let batcher = (config.max_batch > 1)
            .then(|| {
                tokio::runtime::Handle::try_current().ok().map(|_| {
                    Batcher::spawn(
                        runtime.clone(),
                        BatchConfig {
                            max_batch: config.max_batch,
                            max_wait: config.max_wait,
                        },
                        metrics.clone(),
                    )
                })
            })
            .flatten();
        Arc::new(Self {
            runtime,
            config,
            batcher,
            metrics,
            started: Instant::now(),
        })
    }
}
