//! Shared client configuration, environment resolution, and headers.

use std::time::Duration;

use reqwest::header::{ACCEPT, AUTHORIZATION, HeaderMap, HeaderName, HeaderValue, USER_AGENT};
use url::Url;

use crate::error::Error;
use crate::retry::RetryPolicy;

/// Default TypeSafe API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";
/// Default model name.
pub const DEFAULT_MODEL: &str = "jev-latest";
/// Environment variable for the API key.
pub const API_KEY_ENV: &str = "TYPESAFE_API_KEY";
/// Environment variable for the API base URL.
pub const BASE_URL_ENV: &str = "TYPESAFE_BASE_URL";
/// Environment variable for the default model.
pub const DEFAULT_MODEL_ENV: &str = "TYPESAFE_DEFAULT_MODEL";
/// Default per-request timeout.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

pub(crate) const USER_AGENT_VALUE: &str = concat!("olaya-sdk/", env!("CARGO_PKG_VERSION"));

/// Request-body compression.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Compression {
    /// Send request bodies uncompressed.
    #[default]
    None,
    /// Gzip JSON request bodies of at least 1 KiB.
    #[cfg(feature = "gzip")]
    Gzip,
}

#[derive(Clone, Default)]
pub(crate) struct ConfigBuilder {
    api_key: Option<String>,
    api_key_disabled: bool,
    base_url: Option<String>,
    default_model: Option<String>,
    retry: Option<RetryPolicy>,
    timeout: Option<Duration>,
    timeout_disabled: bool,
    headers: HeaderMap,
    allow_unauthenticated: bool,
    compression: Compression,
    user_agent: Option<String>,
}

pub(crate) struct Config {
    pub(crate) api_key: Option<String>,
    pub(crate) base_url: Url,
    pub(crate) default_model: String,
    pub(crate) retry: RetryPolicy,
    pub(crate) timeout: Option<Duration>,
    pub(crate) headers: HeaderMap,
    pub(crate) compression: Compression,
    pub(crate) user_agent: String,
}

impl ConfigBuilder {
    pub(crate) fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self.api_key_disabled = false;
        self
    }

    /// Ignore `TYPESAFE_API_KEY` and send no `Authorization` header.
    pub(crate) fn without_api_key(mut self) -> Self {
        self.api_key = None;
        self.api_key_disabled = true;
        self
    }

    pub(crate) fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub(crate) fn default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    pub(crate) fn retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = Some(retry);
        self
    }

    pub(crate) fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self.timeout_disabled = false;
        self
    }

    pub(crate) fn no_timeout(mut self) -> Self {
        self.timeout = None;
        self.timeout_disabled = true;
        self
    }

    pub(crate) fn header(
        mut self,
        name: impl AsRef<str>,
        value: impl AsRef<str>,
    ) -> Result<Self, Error> {
        let (name, value) = parse_header(name.as_ref(), value.as_ref())?;
        self.headers.insert(name, value);
        Ok(self)
    }

    pub(crate) fn allow_unauthenticated(mut self, allow: bool) -> Self {
        self.allow_unauthenticated = allow;
        self
    }

    pub(crate) fn compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    pub(crate) fn user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = Some(user_agent.into());
        self
    }

    pub(crate) fn build(self) -> Result<Config, Error> {
        let api_key = if self.api_key_disabled {
            None
        } else {
            match self.api_key {
                Some(raw) => Some(validate_api_key(&raw)?),
                None => match env_nonempty(API_KEY_ENV) {
                    Some(raw) => Some(validate_api_key(&raw)?),
                    None => None,
                },
            }
        };
        if api_key.is_none() && !self.allow_unauthenticated {
            return Err(Error::MissingApiKey);
        }
        let base_url = resolve_base_url(self.base_url.as_deref())?;
        let default_model = resolve_default_model(self.default_model.as_deref())?;
        let retry = self.retry.unwrap_or_default();
        let timeout = if self.timeout_disabled {
            None
        } else {
            self.timeout.or(Some(DEFAULT_TIMEOUT))
        };
        let user_agent = self
            .user_agent
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| USER_AGENT_VALUE.to_owned());
        // Validate the user agent now so builds fail early.
        HeaderValue::from_str(&user_agent).map_err(|error| Error::Config {
            message: format!("invalid user agent: {error}"),
        })?;
        Ok(Config {
            api_key,
            base_url,
            default_model,
            retry,
            timeout,
            headers: self.headers,
            compression: self.compression,
            user_agent,
        })
    }
}

impl Config {
    pub(crate) fn endpoint(&self, path: &str) -> Result<Url, Error> {
        self.base_url.join(path).map_err(|error| Error::Config {
            message: format!("joining {} with {path}: {error}", self.base_url),
        })
    }

    /// Merge client headers with per-call headers and the managed headers.
    pub(crate) fn request_headers(&self, call_headers: &HeaderMap) -> Result<HeaderMap, Error> {
        let mut headers = self.headers.clone();
        for (name, value) in call_headers {
            if !is_protected(name) {
                headers.insert(name.clone(), value.clone());
            }
        }
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        if !headers.contains_key(USER_AGENT) {
            headers.insert(
                USER_AGENT,
                HeaderValue::from_str(&self.user_agent).map_err(|error| Error::Config {
                    message: format!("invalid user agent: {error}"),
                })?,
            );
        }
        if let Some(api_key) = &self.api_key {
            let value = HeaderValue::from_str(&format!("Bearer {api_key}")).map_err(|error| {
                Error::InvalidApiKey {
                    reason: format!("the key cannot be sent in a header: {error}"),
                }
            })?;
            headers.insert(AUTHORIZATION, value);
        }
        Ok(headers)
    }
}

/// Parse a header, rejecting headers managed by the SDK.
pub(crate) fn parse_header(name: &str, value: &str) -> Result<(HeaderName, HeaderValue), Error> {
    let name = HeaderName::from_bytes(name.as_bytes()).map_err(|error| Error::Config {
        message: format!("invalid header name {name:?}: {error}"),
    })?;
    if is_protected(&name) {
        return Err(Error::Config {
            message: format!("header {name:?} is managed by the SDK"),
        });
    }
    let value = HeaderValue::from_str(value).map_err(|error| Error::Config {
        message: format!("invalid value for header {name:?}: {error}"),
    })?;
    Ok((name, value))
}

fn is_protected(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "authorization"
            | "accept"
            | "content-type"
            | "content-length"
            | "content-encoding"
            | "host"
            | "transfer-encoding"
    )
}

pub(crate) fn validate_api_key(raw: &str) -> Result<String, Error> {
    let key = raw.trim();
    if key.is_empty() {
        return Err(Error::InvalidApiKey {
            reason: "the key is empty".to_owned(),
        });
    }
    if !key.is_ascii() {
        return Err(Error::InvalidApiKey {
            reason: "the key must be ASCII".to_owned(),
        });
    }
    if key
        .chars()
        .any(|character| character.is_ascii_whitespace() || character.is_ascii_control())
    {
        return Err(Error::InvalidApiKey {
            reason: "the key must not contain whitespace or control characters".to_owned(),
        });
    }
    Ok(key.to_owned())
}

fn resolve_base_url(explicit: Option<&str>) -> Result<Url, Error> {
    let raw = match explicit {
        Some(value) => {
            let value = value.trim();
            if value.is_empty() {
                return Err(Error::Config {
                    message: "base_url must not be empty".to_owned(),
                });
            }
            value.to_owned()
        }
        None => env_nonempty(BASE_URL_ENV).unwrap_or_else(|| DEFAULT_BASE_URL.to_owned()),
    };
    normalize_base_url(&raw)
}

fn normalize_base_url(raw: &str) -> Result<Url, Error> {
    let mut url = Url::parse(raw.trim()).map_err(|error| Error::Config {
        message: format!("invalid base_url {raw:?}: {error}"),
    })?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(Error::Config {
            message: format!("base_url must use http or https, not {}", url.scheme()),
        });
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(Error::Config {
            message: "base_url must not contain credentials".to_owned(),
        });
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err(Error::Config {
            message: "base_url must not contain a query or fragment".to_owned(),
        });
    }
    if url.cannot_be_a_base() {
        return Err(Error::Config {
            message: "base_url must be an absolute URL".to_owned(),
        });
    }
    if !url.path().ends_with('/') {
        let path = format!("{}/", url.path());
        url.set_path(&path);
    }
    Ok(url)
}

fn resolve_default_model(explicit: Option<&str>) -> Result<String, Error> {
    let model = match explicit {
        Some(value) => {
            let value = value.trim();
            if value.is_empty() {
                return Err(Error::Config {
                    message: "default_model must not be empty".to_owned(),
                });
            }
            value.to_owned()
        }
        None => env_nonempty(DEFAULT_MODEL_ENV).unwrap_or_else(|| DEFAULT_MODEL.to_owned()),
    };
    Ok(model)
}

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_urls_with_paths_join_correctly() {
        let config = ConfigBuilder::default()
            .api_key("test-key")
            .base_url("https://openrouter.ai/api")
            .build()
            .expect("config");
        assert_eq!(
            config.endpoint("v1/systemone").expect("url").as_str(),
            "https://openrouter.ai/api/v1/systemone"
        );
        assert_eq!(
            config.endpoint("v1/models").expect("url").as_str(),
            "https://openrouter.ai/api/v1/models"
        );
    }

    #[test]
    fn base_urls_reject_queries_credentials_and_bad_schemes() {
        for value in [
            "ftp://example.com",
            "https://user:pass@example.com",
            "https://example.com?x=1",
            "https://example.com#fragment",
            "",
            "   ",
            "not a url",
        ] {
            assert!(
                ConfigBuilder::default()
                    .api_key("test-key")
                    .base_url(value)
                    .build()
                    .is_err(),
                "{value:?} should be rejected"
            );
        }
    }

    #[test]
    fn api_keys_are_trimmed_and_validated() {
        assert_eq!(validate_api_key("  abc-123\n").expect("key"), "abc-123");
        assert!(validate_api_key("").is_err());
        assert!(validate_api_key("abc def").is_err());
        assert!(validate_api_key("abc\tdef").is_err());
        assert!(validate_api_key("abc\u{00e9}").is_err());
        assert!(validate_api_key("abc\n").is_ok());
    }

    #[test]
    fn explicit_empty_options_do_not_fall_back() {
        assert!(
            ConfigBuilder::default()
                .api_key("test-key")
                .default_model("  ")
                .build()
                .is_err()
        );
    }

    #[test]
    fn protected_headers_are_rejected() {
        for name in [
            "authorization",
            "Accept",
            "Content-Type",
            "Content-Length",
            "Content-Encoding",
            "Host",
            "Transfer-Encoding",
        ] {
            assert!(
                ConfigBuilder::default()
                    .api_key("test-key")
                    .header(name, "value")
                    .is_err(),
                "{name:?} should be protected"
            );
        }
        assert!(
            ConfigBuilder::default()
                .api_key("test-key")
                .header("X-Custom", "value")
                .is_ok()
        );
    }

    #[test]
    fn request_headers_include_auth_accept_and_user_agent() {
        let config = ConfigBuilder::default()
            .api_key("test-key")
            .header("X-Custom", "yes")
            .expect("header")
            .build()
            .expect("config");
        let headers = config.request_headers(&HeaderMap::new()).expect("headers");
        assert_eq!(headers.get(AUTHORIZATION).expect("auth"), "Bearer test-key");
        assert_eq!(headers.get(ACCEPT).expect("accept"), "application/json");
        assert_eq!(headers.get("x-custom").expect("custom"), "yes");
        assert!(headers.get(USER_AGENT).is_some());
    }

    #[test]
    fn unauthenticated_clients_omit_authorization() {
        let config = ConfigBuilder::default()
            .allow_unauthenticated(true)
            .build()
            .expect("config");
        let headers = config.request_headers(&HeaderMap::new()).expect("headers");
        assert!(headers.get(AUTHORIZATION).is_none());
    }

    #[test]
    fn without_api_key_ignores_the_environment() {
        let config = ConfigBuilder::default()
            .without_api_key()
            .allow_unauthenticated(true)
            .build()
            .expect("config");
        assert!(config.api_key.is_none());
        let headers = config.request_headers(&HeaderMap::new()).expect("headers");
        assert!(headers.get(AUTHORIZATION).is_none());
    }

    #[test]
    fn call_headers_cannot_override_managed_headers() {
        let config = ConfigBuilder::default()
            .api_key("test-key")
            .build()
            .expect("config");
        let mut call = HeaderMap::new();
        call.insert(AUTHORIZATION, HeaderValue::from_static("Bearer other"));
        call.insert(ACCEPT, HeaderValue::from_static("text/plain"));
        let headers = config.request_headers(&call).expect("headers");
        assert_eq!(headers.get(AUTHORIZATION).expect("auth"), "Bearer test-key");
        assert_eq!(headers.get(ACCEPT).expect("accept"), "application/json");
    }
}
