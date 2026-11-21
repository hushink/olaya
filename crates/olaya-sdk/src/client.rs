//! The asynchronous TypeSafe-compatible client.

use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use reqwest::header::{CONTENT_ENCODING, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};

use crate::answer::{ListModelsResponse, SystemOneResponse};
use crate::config::{Compression, Config, ConfigBuilder, DEFAULT_TIMEOUT};
use crate::content::State;
use crate::error::Error;
use crate::image::Image;
use crate::request::{Asset, Questions, SystemOneRequest};
use crate::retry::RetryPolicy;
use crate::transport::{
    DANGER_ZONE_IMAGES, DANGER_ZONE_LONG_CONTEXT, api_error, decode_response, encode_json_body,
    request_id, requires_image_header,
};

/// An asynchronous client for the TypeSafe API.
///
/// The client is cheap to clone and safe to share between tasks.
///
/// ```no_run
/// use olaya_sdk::{Client, NoulQuestion, questions};
///
/// # async fn run() -> Result<(), olaya_sdk::Error> {
/// let client = Client::from_env()?;
/// let response = client
///     .system_one(
///         "I was charged twice. Please help.",
///         questions! {
///             "billing" => NoulQuestion::new("Is this about billing?"),
///         },
///     )
///     .await?;
/// println!("{}", response.noul("billing").expect("answer").noul);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Client {
    inner: Arc<ClientInner>,
}

struct ClientInner {
    config: Config,
    http: reqwest::Client,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Client")
            .field("base_url", &self.inner.config.base_url.as_str())
            .field("default_model", &self.inner.config.default_model)
            .field("authenticated", &self.inner.config.api_key.is_some())
            .finish()
    }
}

/// Builder for [`Client`].
#[derive(Default)]
pub struct ClientBuilder {
    config: ConfigBuilder,
    http_client: Option<reqwest::Client>,
}

impl ClientBuilder {
    /// Set the API key explicitly. When omitted, `TYPESAFE_API_KEY` is used.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.config = self.config.api_key(api_key);
        self
    }

    /// Set the API root. Defaults to `TYPESAFE_BASE_URL` or
    /// `https://api.typesafe.ai`.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.config = self.config.base_url(base_url);
        self
    }

    /// Set the default model. Defaults to `TYPESAFE_DEFAULT_MODEL` or
    /// `jev-latest`.
    pub fn default_model(mut self, model: impl Into<String>) -> Self {
        self.config = self.config.default_model(model);
        self
    }

    /// Set the retry policy.
    pub fn retry(mut self, retry: RetryPolicy) -> Self {
        self.config = self.config.retry(retry);
        self
    }

    /// Set the per-request timeout. The default is 10 seconds.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config = self.config.timeout(timeout);
        self
    }

    /// Disable the per-request timeout.
    pub fn no_timeout(mut self) -> Self {
        self.config = self.config.no_timeout();
        self
    }

    /// Add a request header. Managed headers such as `Authorization`,
    /// `Accept`, and `Content-Type` are rejected.
    pub fn header(mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Result<Self, Error> {
        self.config = self.config.header(name, value)?;
        Ok(self)
    }

    /// Allow building a client without an API key. A key supplied explicitly
    /// or through `TYPESAFE_API_KEY` is still sent; call
    /// [`ClientBuilder::without_api_key`] to suppress it. Useful for
    /// compatible servers that do not require authentication, such as a local
    /// `olaya serve` instance.
    pub fn allow_unauthenticated(mut self, allow: bool) -> Self {
        self.config = self.config.allow_unauthenticated(allow);
        self
    }

    /// Send no `Authorization` header, even when `TYPESAFE_API_KEY` is set.
    /// Pair this with [`ClientBuilder::allow_unauthenticated`] for local
    /// servers without authentication.
    pub fn without_api_key(mut self) -> Self {
        self.config = self.config.without_api_key();
        self
    }

    /// Set request-body compression. The `gzip` feature enables
    /// `Compression::Gzip`.
    pub fn compression(mut self, compression: Compression) -> Self {
        self.config = self.config.compression(compression);
        self
    }

    /// Override the `User-Agent` header.
    pub fn user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.config = self.config.user_agent(user_agent);
        self
    }

    /// Use a caller-provided `reqwest::Client`.
    pub fn http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = Some(client);
        self
    }

    /// Build the client.
    pub fn build(self) -> Result<Client, Error> {
        let config = self.config.build()?;
        let http = match self.http_client {
            Some(client) => client,
            None => reqwest::Client::builder()
                .pool_idle_timeout(Duration::from_secs(30))
                .pool_max_idle_per_host(100)
                .build()
                .map_err(|error| Error::Config {
                    message: format!("building the HTTP client: {error}"),
                })?,
        };
        Ok(Client {
            inner: Arc::new(ClientInner { config, http }),
        })
    }
}

impl Client {
    /// Start building a client.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    /// Create a client with an explicit API key and defaults from the
    /// environment for the base URL and model.
    pub fn new(api_key: impl Into<String>) -> Result<Self, Error> {
        Self::builder().api_key(api_key).build()
    }

    /// Create a client from `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, and
    /// `TYPESAFE_DEFAULT_MODEL`.
    pub fn from_env() -> Result<Self, Error> {
        Self::builder().build()
    }

    /// The default model name used when a request does not set one.
    pub fn default_model(&self) -> &str {
        &self.inner.config.default_model
    }

    /// The API root, with a trailing slash.
    pub fn base_url(&self) -> &str {
        self.inner.config.base_url.as_str()
    }

    /// The client-level retry policy.
    pub fn retry_policy(&self) -> &RetryPolicy {
        &self.inner.config.retry
    }

    /// The per-request timeout, or `None` when disabled.
    pub fn timeout(&self) -> Option<Duration> {
        self.inner.config.timeout
    }

    /// Whether the client sends an `Authorization` header.
    pub fn is_authenticated(&self) -> bool {
        self.inner.config.api_key.is_some()
    }

    /// Ask questions about `state`. The client's default model is used unless
    /// [`SystemOneCall::model`] is called.
    pub fn system_one(&self, state: impl Into<State>, questions: Questions) -> SystemOneCall<'_> {
        SystemOneCall::new(self, SystemOneRequest::new(state, questions))
    }

    /// Send a prepared [`SystemOneRequest`].
    pub fn system_one_request(&self, request: SystemOneRequest) -> SystemOneCall<'_> {
        SystemOneCall::new(self, request)
    }

    /// Access the models resource.
    pub fn models(&self) -> Models<'_> {
        Models::new(self)
    }

    /// List the models and aliases available to the account.
    pub async fn list_models(&self) -> Result<ListModelsResponse, Error> {
        self.models().list().await
    }

    async fn execute_system_one(
        &self,
        prepared: crate::request::PreparedBody,
        endpoint: String,
        headers: HeaderMap,
        retry: RetryPolicy,
        timeout: Option<Duration>,
    ) -> Result<SystemOneResponse, Error> {
        let started = Instant::now();
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            match self
                .send_system_one_once(&prepared, &endpoint, &headers, timeout)
                .await
            {
                Ok(response) => return Ok(response),
                Err(mut error) => {
                    error.set_attempts(attempt);
                    if !self.should_retry(&error, &retry, attempt) {
                        return Err(error);
                    }
                    let delay = retry.retry_delay(attempt - 1, error.retry_after());
                    if !retry.within_deadline(started, delay) {
                        return Err(error);
                    }
                    tracing::warn!(
                        method = "POST",
                        endpoint = %endpoint,
                        status = ?error.status(),
                        attempt,
                        delay_ms = delay.as_millis(),
                        "retrying TypeSafe request"
                    );
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    async fn send_system_one_once(
        &self,
        prepared: &crate::request::PreparedBody,
        endpoint: &str,
        headers: &HeaderMap,
        timeout: Option<Duration>,
    ) -> Result<SystemOneResponse, Error> {
        let mut request = self.inner.http.post(endpoint).headers(headers.clone());
        if let Some(timeout) = timeout {
            request = request.timeout(timeout);
        }
        if prepared.images.is_empty() {
            let (body, compressed) =
                encode_json_body(&prepared.json, self.inner.config.compression)?;
            request = request.header(CONTENT_TYPE, "application/json").body(body);
            if compressed {
                request = request.header(CONTENT_ENCODING, "gzip");
            }
        } else {
            let json = serde_json::to_string(&prepared.json).map_err(|error| Error::Config {
                message: format!("serializing the multipart request: {error}"),
            })?;
            let mut form = reqwest::multipart::Form::new().part(
                "request",
                reqwest::multipart::Part::text(json)
                    .mime_str("application/json")
                    .map_err(|error| Error::Config {
                        message: format!("building the multipart request: {error}"),
                    })?,
            );
            for image in &prepared.images {
                let part = reqwest::multipart::Part::bytes(image.bytes.to_vec())
                    .file_name(image.filename.clone())
                    .mime_str(image.content_type)
                    .map_err(|error| Error::Config {
                        message: format!("building the multipart image part: {error}"),
                    })?;
                form = form.part(image.part.clone(), part);
            }
            request = request.multipart(form);
        }

        let response = request
            .send()
            .await
            .map_err(|source| self.transport_error("POST", endpoint, timeout, source))?;
        let status = response.status();
        let headers = response.headers().clone();
        let request_id = request_id(&headers);
        let bytes = response
            .bytes()
            .await
            .map_err(|source| self.transport_error("POST", endpoint, timeout, source))?;
        if !status.is_success() {
            return Err(Error::Api(Box::new(api_error(
                status, "POST", endpoint, &headers, &bytes, 1,
            ))));
        }
        tracing::debug!(
            method = "POST",
            status = status.as_u16(),
            request_id = ?request_id,
            "TypeSafe response"
        );
        let mut response: SystemOneResponse = decode_response("POST", endpoint, &headers, &bytes)?;
        response.request_id = request_id;
        Ok(response)
    }

    async fn execute_models(
        &self,
        endpoint: String,
        headers: HeaderMap,
        retry: RetryPolicy,
        timeout: Option<Duration>,
    ) -> Result<ListModelsResponse, Error> {
        tracing::debug!(method = "GET", endpoint = %endpoint, "sending TypeSafe request");
        let started = Instant::now();
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            match self.send_models_once(&endpoint, &headers, timeout).await {
                Ok(response) => return Ok(response),
                Err(mut error) => {
                    error.set_attempts(attempt);
                    if !self.should_retry(&error, &retry, attempt) {
                        return Err(error);
                    }
                    let delay = retry.retry_delay(attempt - 1, error.retry_after());
                    if !retry.within_deadline(started, delay) {
                        return Err(error);
                    }
                    tracing::warn!(
                        method = "GET",
                        endpoint = %endpoint,
                        status = ?error.status(),
                        attempt,
                        delay_ms = delay.as_millis(),
                        "retrying TypeSafe request"
                    );
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    async fn send_models_once(
        &self,
        endpoint: &str,
        headers: &HeaderMap,
        timeout: Option<Duration>,
    ) -> Result<ListModelsResponse, Error> {
        let mut request = self.inner.http.get(endpoint).headers(headers.clone());
        if let Some(timeout) = timeout {
            request = request.timeout(timeout);
        }
        let response = request
            .send()
            .await
            .map_err(|source| self.transport_error("GET", endpoint, timeout, source))?;
        let status = response.status();
        let headers = response.headers().clone();
        let request_id = request_id(&headers);
        let bytes = response
            .bytes()
            .await
            .map_err(|source| self.transport_error("GET", endpoint, timeout, source))?;
        if !status.is_success() {
            return Err(Error::Api(Box::new(api_error(
                status, "GET", endpoint, &headers, &bytes, 1,
            ))));
        }
        tracing::debug!(
            method = "GET",
            status = status.as_u16(),
            request_id = ?request_id,
            "TypeSafe response"
        );
        let mut response: ListModelsResponse = decode_response("GET", endpoint, &headers, &bytes)?;
        response.request_id = request_id;
        Ok(response)
    }

    fn should_retry(&self, error: &Error, retry: &RetryPolicy, attempt: u32) -> bool {
        attempt <= retry.max_retries && error.is_retryable(retry)
    }

    fn transport_error(
        &self,
        method: &'static str,
        endpoint: &str,
        timeout: Option<Duration>,
        source: reqwest::Error,
    ) -> Error {
        if source.is_timeout() {
            Error::Timeout {
                method,
                endpoint: endpoint.to_owned(),
                timeout: timeout.unwrap_or(DEFAULT_TIMEOUT),
                attempts: 1,
                source: Some(Box::new(source)),
            }
        } else {
            Error::Transport {
                method,
                endpoint: endpoint.to_owned(),
                attempts: 1,
                source,
            }
        }
    }
}

/// A prepared System One call. Await it directly or call
/// [`SystemOneCall::send`].
pub struct SystemOneCall<'a> {
    client: &'a Client,
    request: SystemOneRequest,
    retry: Option<RetryPolicy>,
    timeout: Option<Duration>,
    timeout_disabled: bool,
    headers: HeaderMap,
    danger_zone_images: Option<bool>,
    danger_zone_long_context: bool,
}

impl<'a> SystemOneCall<'a> {
    fn new(client: &'a Client, request: SystemOneRequest) -> Self {
        Self {
            client,
            request,
            retry: None,
            timeout: None,
            timeout_disabled: false,
            headers: HeaderMap::new(),
            danger_zone_images: None,
            danger_zone_long_context: false,
        }
    }

    /// Override the model for this call.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.request = self.request.model(model);
        self
    }

    /// Attach an asset keyed by the reference used in the request content.
    pub fn asset(mut self, name: impl Into<String>, asset: impl Into<Asset>) -> Self {
        self.request = self.request.asset(name, asset);
        self
    }

    /// Attach a multipart image.
    pub fn image(mut self, name: impl Into<String>, image: Image) -> Self {
        self.request = self.request.image(name, image);
        self
    }

    /// Attach an inline base64 image.
    pub fn inline_image(mut self, name: impl Into<String>, image: Image) -> Self {
        self.request = self.request.inline_image(name, image);
        self
    }

    /// Add an extra top-level request field.
    pub fn extra_body(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.request = self.request.extra_body(key, value);
        self
    }

    /// Override the retry policy for this call.
    pub fn retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = Some(retry);
        self
    }

    /// Override the timeout for this call.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self.timeout_disabled = false;
        self
    }

    /// Disable the timeout for this call.
    pub fn no_timeout(mut self) -> Self {
        self.timeout = None;
        self.timeout_disabled = true;
        self
    }

    /// Add a request header. Managed headers are rejected.
    pub fn header(mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Result<Self, Error> {
        let (name, value) = crate::config::parse_header(name.as_ref(), value.as_ref())?;
        self.headers.insert(name, value);
        Ok(self)
    }

    /// Force the `TypeSafe-Danger-Zone-Images` header on or off. By default it
    /// is sent automatically when the request contains assets or box
    /// questions.
    pub fn danger_zone_images(mut self, enable: bool) -> Self {
        self.danger_zone_images = Some(enable);
        self
    }

    /// Send `TypeSafe-Danger-Zone-Long-Context: true`.
    pub fn danger_zone_long_context(mut self, enable: bool) -> Self {
        self.danger_zone_long_context = enable;
        self
    }

    /// The request being built.
    pub fn request(&self) -> &SystemOneRequest {
        &self.request
    }

    /// Mutate the request being built.
    pub fn request_mut(&mut self) -> &mut SystemOneRequest {
        &mut self.request
    }

    /// Send the request.
    pub async fn send(self) -> Result<SystemOneResponse, Error> {
        self.execute().await
    }

    async fn execute(self) -> Result<SystemOneResponse, Error> {
        self.request.validate()?;
        let model = self
            .request
            .model
            .clone()
            .unwrap_or_else(|| self.client.inner.config.default_model.clone());
        let prepared = self.request.prepare(&model)?;
        let endpoint = self
            .client
            .inner
            .config
            .endpoint("v1/systemone")?
            .to_string();
        tracing::debug!(
            method = "POST",
            endpoint = %endpoint,
            model = %model,
            images = prepared.images.len(),
            "sending TypeSafe request"
        );
        let needs_images = requires_image_header(&self.request) || !prepared.images.is_empty();
        let mut headers = self.client.inner.config.request_headers(&self.headers)?;
        if self.danger_zone_images.unwrap_or(needs_images) {
            headers.insert(
                HeaderName::from_static(DANGER_ZONE_IMAGES),
                HeaderValue::from_static("true"),
            );
        }
        if self.danger_zone_long_context {
            headers.insert(
                HeaderName::from_static(DANGER_ZONE_LONG_CONTEXT),
                HeaderValue::from_static("true"),
            );
        }
        let retry = self
            .retry
            .unwrap_or_else(|| self.client.inner.config.retry.clone());
        let timeout = if self.timeout_disabled {
            None
        } else {
            self.timeout.or(self.client.inner.config.timeout)
        };
        self.client
            .execute_system_one(prepared, endpoint, headers, retry, timeout)
            .await
    }
}

impl<'a> IntoFuture for SystemOneCall<'a> {
    type Output = Result<SystemOneResponse, Error>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

/// The models resource, reached through [`Client::models`].
pub struct Models<'a> {
    client: &'a Client,
    retry: Option<RetryPolicy>,
    timeout: Option<Duration>,
    timeout_disabled: bool,
    headers: HeaderMap,
}

impl<'a> Models<'a> {
    fn new(client: &'a Client) -> Self {
        Self {
            client,
            retry: None,
            timeout: None,
            timeout_disabled: false,
            headers: HeaderMap::new(),
        }
    }

    /// Override the retry policy for this call.
    pub fn retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = Some(retry);
        self
    }

    /// Override the timeout for this call.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self.timeout_disabled = false;
        self
    }

    /// Disable the timeout for this call.
    pub fn no_timeout(mut self) -> Self {
        self.timeout = None;
        self.timeout_disabled = true;
        self
    }

    /// Add a request header. Managed headers are rejected.
    pub fn header(mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Result<Self, Error> {
        let (name, value) = crate::config::parse_header(name.as_ref(), value.as_ref())?;
        self.headers.insert(name, value);
        Ok(self)
    }

    /// List the models and aliases available to the account.
    pub async fn list(self) -> Result<ListModelsResponse, Error> {
        let endpoint = self.client.inner.config.endpoint("v1/models")?.to_string();
        let headers = self.client.inner.config.request_headers(&self.headers)?;
        let retry = self
            .retry
            .unwrap_or_else(|| self.client.inner.config.retry.clone());
        let timeout = if self.timeout_disabled {
            None
        } else {
            self.timeout.or(self.client.inner.config.timeout)
        };
        self.client
            .execute_models(endpoint, headers, retry, timeout)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_accepts_unauthenticated_clients() {
        let client = Client::builder()
            .allow_unauthenticated(true)
            .base_url("http://127.0.0.1:9")
            .default_model("test-model")
            .build()
            .expect("client");
        assert!(!client.is_authenticated());
        assert_eq!(client.default_model(), "test-model");
        assert_eq!(client.base_url(), "http://127.0.0.1:9/");
    }

    #[test]
    fn invalid_keys_are_rejected_before_any_request() {
        let error = Client::builder()
            .api_key("bad key")
            .build()
            .expect_err("key");
        assert!(matches!(error, Error::InvalidApiKey { .. }));
    }
}
