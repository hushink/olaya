//! Retry policy, backoff, and `Retry-After` handling.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};

use reqwest::header::HeaderMap;

/// Upper bound on a server-supplied `Retry-After` delay. A hostile or broken
/// server must not be able to stall a client indefinitely.
pub const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);

/// Configuration for retrying failed requests.
///
/// The defaults mirror the official TypeSafe SDK: two retries after the initial
/// attempt, exponential backoff from 500 ms to 5 s with 25% jitter, retries on
/// 408, 429, and 5xx responses, and a 30 s total budget per call.
#[derive(Clone, Debug)]
pub struct RetryPolicy {
    /// Maximum retries after the initial attempt; `0` disables retries.
    pub max_retries: u32,
    /// First backoff delay in seconds, doubled each attempt.
    pub backoff_initial: Duration,
    /// Maximum backoff delay.
    pub backoff_max: Duration,
    /// Fraction of each backoff delay randomly subtracted, between 0 and 1.
    pub backoff_jitter: f64,
    /// HTTP status codes that are retried.
    pub retry_statuses: BTreeSet<u16>,
    /// Whether to honor `Retry-After` and `retry-after-ms` response headers.
    pub respect_retry_after: bool,
    /// Whether connection failures are retried.
    pub retry_connection_errors: bool,
    /// Whether request timeouts are retried.
    pub retry_timeouts: bool,
    /// Total budget per call, including the initial attempt and sleeps.
    pub deadline: Option<Duration>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            backoff_initial: Duration::from_millis(500),
            backoff_max: Duration::from_secs(5),
            backoff_jitter: 0.25,
            retry_statuses: default_retry_statuses(),
            respect_retry_after: true,
            retry_connection_errors: true,
            retry_timeouts: true,
            deadline: Some(Duration::from_secs(30)),
        }
    }
}

impl RetryPolicy {
    /// A policy that never retries.
    pub fn disabled() -> Self {
        Self {
            max_retries: 0,
            retry_statuses: BTreeSet::new(),
            deadline: None,
            ..Self::default()
        }
    }

    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    pub fn backoff_initial(mut self, backoff_initial: Duration) -> Self {
        self.backoff_initial = backoff_initial;
        self
    }

    pub fn backoff_max(mut self, backoff_max: Duration) -> Self {
        self.backoff_max = backoff_max;
        self
    }

    pub fn backoff_jitter(mut self, backoff_jitter: f64) -> Self {
        self.backoff_jitter = backoff_jitter;
        self
    }

    pub fn retry_statuses(mut self, statuses: impl IntoIterator<Item = u16>) -> Self {
        self.retry_statuses = statuses.into_iter().collect();
        self
    }

    pub fn respect_retry_after(mut self, respect: bool) -> Self {
        self.respect_retry_after = respect;
        self
    }

    pub fn retry_connection_errors(mut self, retry: bool) -> Self {
        self.retry_connection_errors = retry;
        self
    }

    pub fn retry_timeouts(mut self, retry: bool) -> Self {
        self.retry_timeouts = retry;
        self
    }

    pub fn deadline(mut self, deadline: Option<Duration>) -> Self {
        self.deadline = deadline;
        self
    }

    /// The backoff delay before retry number `retry_index` (zero-based).
    pub fn backoff(&self, retry_index: u32) -> Duration {
        let factor = 2u32.checked_pow(retry_index.min(20)).unwrap_or(u32::MAX);
        let base = self
            .backoff_initial
            .saturating_mul(factor)
            .min(self.backoff_max);
        let jitter = if self.backoff_jitter.is_finite() {
            self.backoff_jitter.clamp(0.0, 1.0)
        } else {
            0.0
        };
        if jitter == 0.0 {
            return base;
        }
        let subtracted = base.mul_f64(jitter * random_fraction());
        base.saturating_sub(subtracted)
    }

    /// The delay before retry number `retry_index`, preferring the server's
    /// `Retry-After` when allowed.
    pub fn retry_delay(&self, retry_index: u32, retry_after: Option<Duration>) -> Duration {
        if self.respect_retry_after
            && let Some(retry_after) = retry_after
        {
            // Clamp server-controlled delays so `Duration::MAX` (or a huge
            // value) cannot overflow an `Instant` or sleep for years when the
            // caller disabled the deadline.
            return retry_after.min(MAX_RETRY_AFTER);
        }
        self.backoff(retry_index)
    }

    /// Whether sleeping `delay` now would stay inside the total budget.
    pub fn within_deadline(&self, started: Instant, delay: Duration) -> bool {
        match self.deadline {
            None => true,
            Some(budget) => started
                .elapsed()
                .checked_add(delay)
                .is_some_and(|total| total < budget),
        }
    }
}

/// Parse `retry-after-ms` or `Retry-After` from response headers.
pub fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    if let Some(value) = headers
        .get("retry-after-ms")
        .and_then(|value| value.to_str().ok())
        && let Ok(milliseconds) = value.trim().parse::<f64>()
        && milliseconds.is_finite()
        && milliseconds >= 0.0
    {
        return Some(Duration::try_from_secs_f64(milliseconds / 1000.0).unwrap_or(Duration::MAX));
    }
    let value = headers.get("retry-after")?.to_str().ok()?.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    httpdate::parse_http_date(value).ok().map(|date| {
        date.duration_since(SystemTime::now())
            .unwrap_or(Duration::ZERO)
    })
}

fn default_retry_statuses() -> BTreeSet<u16> {
    let mut statuses: BTreeSet<u16> = (500..=599).collect();
    statuses.insert(408);
    statuses.insert(429);
    statuses
}

/// A uniform value in `[0, 1)` without a random-number dependency.
fn random_fraction() -> f64 {
    static STATE: AtomicU64 = AtomicU64::new(0x9E37_79B9_7F4A_7C15);
    let mut value = STATE.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    (value >> 11) as f64 / (1u64 << 53) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderName, HeaderValue};

    fn headers(name: &str, value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_bytes(name.as_bytes()).expect("name"),
            HeaderValue::from_str(value).expect("value"),
        );
        headers
    }

    #[test]
    fn defaults_match_the_official_sdk() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_retries, 2);
        assert_eq!(policy.backoff_initial, Duration::from_millis(500));
        assert_eq!(policy.backoff_max, Duration::from_secs(5));
        assert!(policy.retry_statuses.contains(&408));
        assert!(policy.retry_statuses.contains(&429));
        assert!(policy.retry_statuses.contains(&500));
        assert!(policy.retry_statuses.contains(&529));
        assert!(!policy.retry_statuses.contains(&400));
        assert!(!policy.retry_statuses.contains(&422));
    }

    #[test]
    fn disabled_retries_nothing() {
        let policy = RetryPolicy::disabled();
        assert_eq!(policy.max_retries, 0);
        assert!(policy.retry_statuses.is_empty());
    }

    #[test]
    fn backoff_grows_and_is_bounded() {
        let policy = RetryPolicy {
            backoff_jitter: 0.0,
            ..RetryPolicy::default()
        };
        assert_eq!(policy.backoff(0), Duration::from_millis(500));
        assert_eq!(policy.backoff(1), Duration::from_secs(1));
        assert_eq!(policy.backoff(2), Duration::from_secs(2));
        assert_eq!(policy.backoff(10), Duration::from_secs(5));
    }

    #[test]
    fn retry_after_wins_when_enabled() {
        let policy = RetryPolicy::default().backoff_jitter(0.0);
        assert_eq!(
            policy.retry_delay(0, Some(Duration::from_millis(20))),
            Duration::from_millis(20)
        );
        let policy = policy.respect_retry_after(false);
        assert_eq!(
            policy.retry_delay(0, Some(Duration::from_millis(20))),
            Duration::from_millis(500)
        );
    }

    #[test]
    fn hostile_retry_after_values_are_clamped() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.retry_delay(0, Some(Duration::MAX)), MAX_RETRY_AFTER);
        assert_eq!(
            policy.retry_delay(0, Some(Duration::from_secs(86_400))),
            MAX_RETRY_AFTER
        );
    }

    #[test]
    fn parses_retry_after_seconds_ms_and_date() {
        assert_eq!(
            parse_retry_after(&headers("retry-after", "3")),
            Some(Duration::from_secs(3))
        );
        assert_eq!(
            parse_retry_after(&headers("retry-after-ms", "1500")),
            Some(Duration::from_millis(1500))
        );
        assert_eq!(
            parse_retry_after(&headers("retry-after-ms", "1e300")),
            Some(Duration::MAX)
        );
        let date = SystemTime::now() + Duration::from_secs(4);
        let parsed = parse_retry_after(&headers("retry-after", &httpdate::fmt_http_date(date)))
            .expect("date");
        assert!(parsed >= Duration::from_secs(2));
        assert!(parsed <= Duration::from_secs(5));
    }

    #[test]
    fn deadline_stops_before_an_over_budget_sleep() {
        let started = Instant::now();
        let policy = RetryPolicy::default().deadline(Some(Duration::from_millis(50)));
        assert!(policy.within_deadline(started, Duration::ZERO));
        assert!(!policy.within_deadline(started, Duration::from_secs(1)));
        let policy = policy.deadline(None);
        assert!(policy.within_deadline(started, Duration::from_secs(3600)));
    }
}
