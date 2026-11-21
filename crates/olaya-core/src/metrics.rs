//! Cheap, cloneable runtime metrics for embedded engines and the HTTP server.
//!
//! The handle is shared by `Engine`, the serve `Batcher`, and the HTTP
//! middleware. Every counter is an atomic; latency and batch-size distributions
//! are fixed-bucket histograms so scraping never allocates on the request path.
//! [`RuntimeMetrics::snapshot`] returns a serializable copy and
//! [`RuntimeMetrics::render_prometheus`] renders the standard text format.

use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::Serialize;

/// Inference latency buckets in seconds.
const LATENCY_BOUNDS: [f64; 15] = [
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
];
/// Batch-size buckets.
const BATCH_BOUNDS: [f64; 7] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
/// Model-load latency buckets in seconds.
const LOAD_BOUNDS: [f64; 10] = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0];

/// A fixed-bucket histogram.
struct Histogram {
    bounds: &'static [f64],
    /// One counter per finite bound plus the `+Inf` bucket.
    buckets: Box<[AtomicU64]>,
    count: AtomicU64,
    /// Sum of observations scaled by `sum_scale` (nanoseconds for seconds,
    /// units for batch sizes).
    sum_scaled: AtomicU64,
    sum_scale: f64,
}

impl std::fmt::Debug for Histogram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Histogram")
            .field("bounds", &self.bounds)
            .field("count", &self.count.load(Ordering::Relaxed))
            .finish()
    }
}

impl Histogram {
    fn new(bounds: &'static [f64], sum_scale: f64) -> Self {
        Self {
            bounds,
            buckets: (0..=bounds.len()).map(|_| AtomicU64::new(0)).collect(),
            count: AtomicU64::new(0),
            sum_scaled: AtomicU64::new(0),
            sum_scale,
        }
    }

    fn observe(&self, value: f64) {
        let value = value.max(0.0);
        let index = self
            .bounds
            .iter()
            .position(|bound| value <= *bound)
            .unwrap_or(self.bounds.len());
        self.buckets[index].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        let scaled = (value * self.sum_scale).round();
        let scaled = if scaled >= u64::MAX as f64 {
            u64::MAX
        } else {
            scaled as u64
        };
        self.sum_scaled.fetch_add(scaled, Ordering::Relaxed);
    }

    fn snapshot(&self) -> HistogramSnapshot {
        let mut cumulative = 0u64;
        let buckets = self
            .buckets
            .iter()
            .map(|bucket| {
                cumulative = cumulative.saturating_add(bucket.load(Ordering::Relaxed));
                cumulative
            })
            .collect();
        HistogramSnapshot {
            bounds: self.bounds.to_vec(),
            buckets,
            count: self.count.load(Ordering::Relaxed),
            sum: self.sum_scaled.load(Ordering::Relaxed) as f64 / self.sum_scale,
        }
    }
}

/// A serializable histogram snapshot; `buckets` are cumulative (`le` semantics).
#[derive(Clone, Debug, Default, Serialize)]
pub struct HistogramSnapshot {
    pub bounds: Vec<f64>,
    pub buckets: Vec<u64>,
    pub count: u64,
    pub sum: f64,
}

impl HistogramSnapshot {
    fn render_prometheus(&self, out: &mut String, name: &str, help: &str) {
        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} histogram");
        for (index, bound) in self.bounds.iter().enumerate() {
            let value = self.buckets.get(index).copied().unwrap_or(0);
            let _ = writeln!(
                out,
                "{name}_bucket{{le=\"{}\"}} {value}",
                format_bound(*bound)
            );
        }
        let total = self.buckets.last().copied().unwrap_or(0);
        let _ = writeln!(out, "{name}_bucket{{le=\"+Inf\"}} {total}");
        let _ = writeln!(out, "{name}_sum {}", self.sum);
        let _ = writeln!(out, "{name}_count {}", self.count);
    }
}

fn format_bound(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        format!("{value}")
    }
}

/// A point-in-time copy of [`RuntimeMetrics`].
#[derive(Clone, Debug, Default, Serialize)]
pub struct MetricsSnapshot {
    pub uptime_seconds: f64,
    pub requests_total: u64,
    pub request_errors_total: u64,
    pub input_tokens_total: u64,
    pub output_tokens_total: u64,
    pub model_loads_total: u64,
    pub model_load_errors_total: u64,
    pub model_unloads_total: u64,
    pub loaded_models: u64,
    pub pulls_total: u64,
    pub pull_errors_total: u64,
    pub deletes_total: u64,
    pub batches_total: u64,
    pub batch_requests_total: u64,
    pub batch_queue_depth: u64,
    pub inference_seconds: HistogramSnapshot,
    pub model_load_seconds: HistogramSnapshot,
    pub batch_size: HistogramSnapshot,
    pub http_requests_total: u64,
    pub http_request_errors_total: u64,
    pub http_inflight: u64,
    pub http_request_seconds: HistogramSnapshot,
}

impl MetricsSnapshot {
    /// Render the Prometheus text exposition format (version 0.0.4).
    pub fn render_prometheus(&self) -> String {
        let mut out = String::with_capacity(2048);
        let _ = writeln!(
            out,
            "# HELP olaya_uptime_seconds Seconds since the runtime was created"
        );
        let _ = writeln!(out, "# TYPE olaya_uptime_seconds gauge");
        let _ = writeln!(out, "olaya_uptime_seconds {}", self.uptime_seconds);

        counter(
            &mut out,
            "olaya_requests_total",
            "Completed decision requests",
            self.requests_total,
        );
        counter(
            &mut out,
            "olaya_request_errors_total",
            "Failed decision requests",
            self.request_errors_total,
        );
        counter(
            &mut out,
            "olaya_input_tokens_total",
            "Input tokens consumed by decisions",
            self.input_tokens_total,
        );
        counter(
            &mut out,
            "olaya_output_tokens_total",
            "Output tokens produced by decisions",
            self.output_tokens_total,
        );
        counter(
            &mut out,
            "olaya_model_loads_total",
            "Successful family loads",
            self.model_loads_total,
        );
        counter(
            &mut out,
            "olaya_model_load_errors_total",
            "Failed family loads",
            self.model_load_errors_total,
        );
        counter(
            &mut out,
            "olaya_model_unloads_total",
            "Resident models unloaded",
            self.model_unloads_total,
        );
        gauge(
            &mut out,
            "olaya_loaded_models",
            "Currently resident models",
            self.loaded_models,
        );
        counter(
            &mut out,
            "olaya_pulls_total",
            "Model pull attempts",
            self.pulls_total,
        );
        counter(
            &mut out,
            "olaya_pull_errors_total",
            "Failed model pulls",
            self.pull_errors_total,
        );
        counter(
            &mut out,
            "olaya_deletes_total",
            "Catalog models deleted",
            self.deletes_total,
        );
        counter(
            &mut out,
            "olaya_batches_total",
            "Batched inference calls",
            self.batches_total,
        );
        counter(
            &mut out,
            "olaya_batch_requests_total",
            "Requests submitted through batched inference",
            self.batch_requests_total,
        );
        gauge(
            &mut out,
            "olaya_batch_queue_depth",
            "Requests waiting for the serve batcher",
            self.batch_queue_depth,
        );
        self.inference_seconds.render_prometheus(
            &mut out,
            "olaya_inference_seconds",
            "Decision inference latency in seconds",
        );
        self.model_load_seconds.render_prometheus(
            &mut out,
            "olaya_model_load_seconds",
            "Family load latency in seconds",
        );
        self.batch_size.render_prometheus(
            &mut out,
            "olaya_batch_size",
            "Requests per batched inference call",
        );
        counter(
            &mut out,
            "olaya_http_requests_total",
            "HTTP requests handled",
            self.http_requests_total,
        );
        counter(
            &mut out,
            "olaya_http_request_errors_total",
            "HTTP responses with status >= 400",
            self.http_request_errors_total,
        );
        gauge(
            &mut out,
            "olaya_http_inflight",
            "HTTP requests currently in flight",
            self.http_inflight,
        );
        self.http_request_seconds.render_prometheus(
            &mut out,
            "olaya_http_request_seconds",
            "HTTP request latency in seconds",
        );
        out
    }
}

fn counter(out: &mut String, name: &str, help: &str, value: u64) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} counter");
    let _ = writeln!(out, "{name} {value}");
}

fn gauge(out: &mut String, name: &str, help: &str, value: u64) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} gauge");
    let _ = writeln!(out, "{name} {value}");
}

#[derive(Debug)]
struct MetricsInner {
    started: Instant,
    requests_total: AtomicU64,
    request_errors_total: AtomicU64,
    input_tokens_total: AtomicU64,
    output_tokens_total: AtomicU64,
    model_loads_total: AtomicU64,
    model_load_errors_total: AtomicU64,
    model_unloads_total: AtomicU64,
    loaded_models: AtomicU64,
    pulls_total: AtomicU64,
    pull_errors_total: AtomicU64,
    deletes_total: AtomicU64,
    batches_total: AtomicU64,
    batch_requests_total: AtomicU64,
    batch_queue_depth: AtomicU64,
    inference_seconds: Histogram,
    model_load_seconds: Histogram,
    batch_size: Histogram,
    http_requests_total: AtomicU64,
    http_request_errors_total: AtomicU64,
    http_inflight: AtomicU64,
    http_request_seconds: Histogram,
}

/// A shared metrics handle. Cloning shares the counters.
#[derive(Clone, Debug)]
pub struct RuntimeMetrics {
    inner: Arc<MetricsInner>,
}

impl Default for RuntimeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeMetrics {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(MetricsInner {
                started: Instant::now(),
                requests_total: AtomicU64::new(0),
                request_errors_total: AtomicU64::new(0),
                input_tokens_total: AtomicU64::new(0),
                output_tokens_total: AtomicU64::new(0),
                model_loads_total: AtomicU64::new(0),
                model_load_errors_total: AtomicU64::new(0),
                model_unloads_total: AtomicU64::new(0),
                loaded_models: AtomicU64::new(0),
                pulls_total: AtomicU64::new(0),
                pull_errors_total: AtomicU64::new(0),
                deletes_total: AtomicU64::new(0),
                batches_total: AtomicU64::new(0),
                batch_requests_total: AtomicU64::new(0),
                batch_queue_depth: AtomicU64::new(0),
                inference_seconds: Histogram::new(&LATENCY_BOUNDS, 1e9),
                model_load_seconds: Histogram::new(&LOAD_BOUNDS, 1e9),
                batch_size: Histogram::new(&BATCH_BOUNDS, 1.0),
                http_requests_total: AtomicU64::new(0),
                http_request_errors_total: AtomicU64::new(0),
                http_inflight: AtomicU64::new(0),
                http_request_seconds: Histogram::new(&LATENCY_BOUNDS, 1e9),
            }),
        }
    }

    /// Record one decision request with its token usage and wall time.
    pub fn record_request(&self, input_tokens: u64, output_tokens: u64, seconds: f64) {
        self.inner.requests_total.fetch_add(1, Ordering::Relaxed);
        self.inner
            .input_tokens_total
            .fetch_add(input_tokens, Ordering::Relaxed);
        self.inner
            .output_tokens_total
            .fetch_add(output_tokens, Ordering::Relaxed);
        self.inner.inference_seconds.observe(seconds);
    }

    pub fn record_request_error(&self) {
        self.inner
            .request_errors_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record a family load attempt that got as far as building the family.
    pub fn record_model_load(&self, seconds: f64) {
        self.inner.model_loads_total.fetch_add(1, Ordering::Relaxed);
        self.inner.model_load_seconds.observe(seconds);
    }

    pub fn record_model_load_error(&self) {
        self.inner
            .model_load_errors_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_model_unload(&self, removed: bool) {
        if removed {
            self.inner
                .model_unloads_total
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn set_loaded_models(&self, count: usize) {
        self.inner
            .loaded_models
            .store(count as u64, Ordering::Relaxed);
    }

    pub fn record_pull(&self, success: bool) {
        self.inner.pulls_total.fetch_add(1, Ordering::Relaxed);
        if !success {
            self.inner.pull_errors_total.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_delete(&self, removed: bool) {
        if removed {
            self.inner.deletes_total.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record one batched inference call of `size` requests.
    pub fn record_batch(&self, size: usize) {
        self.inner.batches_total.fetch_add(1, Ordering::Relaxed);
        self.inner
            .batch_requests_total
            .fetch_add(size as u64, Ordering::Relaxed);
        self.inner.batch_size.observe(size as f64);
    }

    pub fn batch_queued(&self) {
        self.inner.batch_queue_depth.fetch_add(1, Ordering::Relaxed);
    }

    pub fn batch_dequeued(&self) {
        decrement(&self.inner.batch_queue_depth);
    }

    pub fn http_request_started(&self) {
        self.inner
            .http_requests_total
            .fetch_add(1, Ordering::Relaxed);
        self.inner.http_inflight.fetch_add(1, Ordering::Relaxed);
    }

    /// `error` should be true for responses with status >= 400.
    pub fn http_request_finished(&self, error: bool, seconds: f64) {
        decrement(&self.inner.http_inflight);
        if error {
            self.inner
                .http_request_errors_total
                .fetch_add(1, Ordering::Relaxed);
        }
        self.inner.http_request_seconds.observe(seconds);
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            uptime_seconds: self.inner.started.elapsed().as_secs_f64(),
            requests_total: load(&self.inner.requests_total),
            request_errors_total: load(&self.inner.request_errors_total),
            input_tokens_total: load(&self.inner.input_tokens_total),
            output_tokens_total: load(&self.inner.output_tokens_total),
            model_loads_total: load(&self.inner.model_loads_total),
            model_load_errors_total: load(&self.inner.model_load_errors_total),
            model_unloads_total: load(&self.inner.model_unloads_total),
            loaded_models: load(&self.inner.loaded_models),
            pulls_total: load(&self.inner.pulls_total),
            pull_errors_total: load(&self.inner.pull_errors_total),
            deletes_total: load(&self.inner.deletes_total),
            batches_total: load(&self.inner.batches_total),
            batch_requests_total: load(&self.inner.batch_requests_total),
            batch_queue_depth: load(&self.inner.batch_queue_depth),
            inference_seconds: self.inner.inference_seconds.snapshot(),
            model_load_seconds: self.inner.model_load_seconds.snapshot(),
            batch_size: self.inner.batch_size.snapshot(),
            http_requests_total: load(&self.inner.http_requests_total),
            http_request_errors_total: load(&self.inner.http_request_errors_total),
            http_inflight: load(&self.inner.http_inflight),
            http_request_seconds: self.inner.http_request_seconds.snapshot(),
        }
    }

    /// Render the Prometheus text exposition format.
    pub fn render_prometheus(&self) -> String {
        self.snapshot().render_prometheus()
    }
}

fn load(counter: &AtomicU64) -> u64 {
    counter.load(Ordering::Relaxed)
}

fn decrement(counter: &AtomicU64) {
    let mut current = counter.load(Ordering::Relaxed);
    while current > 0 {
        match counter.compare_exchange_weak(
            current,
            current - 1,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(next) => current = next,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_are_shared_between_clones() {
        let metrics = RuntimeMetrics::new();
        let clone = metrics.clone();
        metrics.record_request(7, 0, 0.012);
        clone.record_request_error();
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.requests_total, 1);
        assert_eq!(snapshot.request_errors_total, 1);
        assert_eq!(snapshot.input_tokens_total, 7);
        assert_eq!(snapshot.inference_seconds.count, 1);
        assert_eq!(snapshot.inference_seconds.buckets.last().copied(), Some(1));
    }

    #[test]
    fn histograms_use_cumulative_buckets() {
        let metrics = RuntimeMetrics::new();
        for seconds in [0.0005, 0.002, 0.02, 1.5] {
            metrics.record_request(0, 0, seconds);
        }
        let histogram = metrics.snapshot().inference_seconds;
        assert_eq!(histogram.count, 4);
        assert_eq!(histogram.buckets[0], 1); // <= 1ms
        assert_eq!(histogram.buckets[2], 2); // <= 5ms
        assert_eq!(histogram.buckets[4], 3); // <= 25ms
        assert_eq!(histogram.buckets.last().copied(), Some(4)); // +Inf
        assert!((histogram.sum - 1.5225).abs() < 1e-6);
    }

    #[test]
    fn prometheus_contains_counters_gauges_and_histograms() {
        let metrics = RuntimeMetrics::new();
        metrics.record_request(3, 0, 0.01);
        metrics.record_batch(4);
        metrics.http_request_started();
        metrics.http_request_finished(true, 0.02);
        let text = metrics.render_prometheus();
        assert!(text.contains("olaya_requests_total 1"));
        assert!(text.contains("olaya_input_tokens_total 3"));
        assert!(text.contains("olaya_batches_total 1"));
        assert!(text.contains("olaya_batch_requests_total 4"));
        assert!(text.contains("olaya_http_request_errors_total 1"));
        assert!(text.contains("olaya_http_inflight 0"));
        assert!(text.contains("olaya_inference_seconds_bucket{le=\"+Inf\"} 1"));
        assert!(text.contains("olaya_batch_size_bucket{le=\"4\"} 1"));
    }

    #[test]
    fn gauges_never_underflow() {
        let metrics = RuntimeMetrics::new();
        metrics.batch_dequeued();
        metrics.http_request_finished(false, 0.0);
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.batch_queue_depth, 0);
        assert_eq!(snapshot.http_inflight, 0);
    }
}
