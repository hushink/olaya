//! Remote catalog registry with a local cache.
//!
//! A fresh install has no filesystem `registry/` directory. The CLI fetches a
//! single `registry-catalog.json` bundle from [`DEFAULT_REGISTRY_URL`] (or
//! `OLAYA_REGISTRY_URL`) on first use and caches it under
//! `<cache_dir>/registry-catalog.json`. The default URL is the
//! `registry-catalog.json` asset of the latest GitHub Release; the release
//! workflow generates and uploads it. Later commands use the cache without
//! touching the network; `Export` and explicit refreshes fetch the current
//! bundle and fall back to the cache when the network is unavailable.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use olaya_core::{CatalogBundle, ManifestRegistry, RegistryError, RepoManifest};
use reqwest::blocking::Client;

use crate::hub::redact_url;

/// Environment variable that overrides the catalog URL.
pub const REGISTRY_URL_ENV: &str = "OLAYA_REGISTRY_URL";
/// Default hosted catalog: the latest GitHub Release's `registry-catalog.json`
/// asset. Override with `OLAYA_REGISTRY_URL` for a versioned asset or an
/// internal mirror.
pub const DEFAULT_REGISTRY_URL: &str =
    "https://github.com/hushink/olaya/releases/latest/download/registry-catalog.json";
/// Cache file name under the hub cache directory.
pub const CATALOG_CACHE_FILE: &str = "registry-catalog.json";
/// Default HTTP timeout for fetching the catalog bundle.
pub const DEFAULT_REGISTRY_TIMEOUT_SECS: u64 = 30;
/// Maximum accepted catalog bundle size. A catalog is a small JSON document;
/// refusing larger bodies prevents a hostile mirror from exhausting memory.
pub const MAX_CATALOG_BYTES: u64 = 16 * 1024 * 1024;

/// Where a catalog load came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CatalogSource {
    Network,
    Cache,
}

impl CatalogSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Network => "network",
            Self::Cache => "cache",
        }
    }
}

/// A [`ManifestRegistry`] backed by a hosted catalog bundle.
pub struct RemoteRegistry {
    url: String,
    cache: Option<PathBuf>,
    client: Client,
}

impl RemoteRegistry {
    /// Build a registry with the default HTTP timeout. `cache` is the catalog
    /// bundle path; pass `None` to disable caching.
    pub fn new(url: impl Into<String>, cache: Option<PathBuf>) -> Result<Self, RegistryError> {
        Self::with_timeout(
            url,
            cache,
            Duration::from_secs(DEFAULT_REGISTRY_TIMEOUT_SECS),
        )
    }

    /// Build a registry with an explicit HTTP timeout.
    pub fn with_timeout(
        url: impl Into<String>,
        cache: Option<PathBuf>,
        timeout: Duration,
    ) -> Result<Self, RegistryError> {
        let url = url.into();
        validate_catalog_url(&url)?;
        let client = Client::builder()
            .timeout(timeout)
            .user_agent(concat!("olaya/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|error| {
                RegistryError::Unavailable(format!("building the HTTP client: {error}"))
            })?;
        Ok(Self { url, cache, client })
    }

    /// Use `OLAYA_REGISTRY_URL` (or [`DEFAULT_REGISTRY_URL`]) and cache under
    /// `cache_dir`.
    pub fn from_env(cache_dir: impl AsRef<Path>) -> Result<Self, RegistryError> {
        let url = std::env::var(REGISTRY_URL_ENV)
            .ok()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_REGISTRY_URL.to_owned());
        Self::new(url, Some(cache_dir.as_ref().join(CATALOG_CACHE_FILE)))
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn cache_path(&self) -> Option<&Path> {
        self.cache.as_deref()
    }

    /// Load the catalog. With `refresh == false`, a readable cache wins and the
    /// network is not touched. With `refresh == true`, the network is tried
    /// first and the cache is the fallback.
    pub fn load(&self, refresh: bool) -> Result<(Vec<RepoManifest>, CatalogSource), RegistryError> {
        if !refresh && let Some(cache) = &self.cache {
            match self.read_cache(cache) {
                Ok(manifests) => return Ok((manifests, CatalogSource::Cache)),
                Err(error) => {
                    tracing::debug!(
                        path = %cache.display(),
                        %error,
                        "catalog cache is unreadable"
                    );
                }
            }
        }
        match self.fetch() {
            Ok(manifests) => {
                if let Some(cache) = &self.cache {
                    self.write_cache(cache, &manifests);
                }
                Ok((manifests, CatalogSource::Network))
            }
            Err(error) => {
                if refresh
                    && let Some(cache) = &self.cache
                    && let Ok(manifests) = self.read_cache(cache)
                {
                    tracing::warn!(
                        url = %redact_url(&self.url),
                        %error,
                        "catalog refresh failed; using the cached catalog"
                    );
                    return Ok((manifests, CatalogSource::Cache));
                }
                Err(error)
            }
        }
    }

    fn fetch(&self) -> Result<Vec<RepoManifest>, RegistryError> {
        let url = redact_url(&self.url);
        tracing::info!(%url, "fetching model catalog");
        let response = self.client.get(&self.url).send().map_err(|error| {
            RegistryError::Unavailable(format!("fetching {url}: {}", error.without_url()))
        })?;
        let status = response.status();
        if !status.is_success() {
            return Err(RegistryError::Unavailable(format!(
                "fetching {url} returned HTTP {status}"
            )));
        }
        if response
            .content_length()
            .is_some_and(|length| length > MAX_CATALOG_BYTES)
        {
            return Err(RegistryError::Unavailable(format!(
                "catalog at {url} is larger than the {MAX_CATALOG_BYTES}-byte limit"
            )));
        }
        let mut bytes = Vec::new();
        response
            .take(MAX_CATALOG_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| RegistryError::Unavailable(format!("reading {url}: {error}")))?;
        if bytes.len() as u64 > MAX_CATALOG_BYTES {
            return Err(RegistryError::Unavailable(format!(
                "catalog at {url} is larger than the {MAX_CATALOG_BYTES}-byte limit"
            )));
        }
        let bundle: CatalogBundle = serde_json::from_slice(&bytes)
            .map_err(|source| RegistryError::Unavailable(format!("parsing {url}: {source}")))?;
        bundle.validate()?;
        let mut manifests = bundle.manifests;
        manifests.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(manifests)
    }

    fn read_cache(&self, path: &Path) -> Result<Vec<RepoManifest>, RegistryError> {
        let metadata = std::fs::metadata(path).map_err(|source| RegistryError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if metadata.len() > MAX_CATALOG_BYTES {
            return Err(RegistryError::Unavailable(format!(
                "cached catalog {} is larger than the {MAX_CATALOG_BYTES}-byte limit",
                path.display()
            )));
        }
        let bytes = std::fs::read(path).map_err(|source| RegistryError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let bundle: CatalogBundle =
            serde_json::from_slice(&bytes).map_err(|source| RegistryError::Json {
                path: path.to_path_buf(),
                source,
            })?;
        bundle.validate()?;
        let mut manifests = bundle.manifests;
        manifests.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(manifests)
    }

    fn write_cache(&self, path: &Path, manifests: &[RepoManifest]) {
        let bundle = CatalogBundle::new(manifests.to_vec());
        let serialized = match serde_json::to_vec_pretty(&bundle) {
            Ok(serialized) => serialized,
            Err(error) => {
                tracing::warn!(path = %path.display(), %error, "serializing the catalog cache");
                return;
            }
        };
        if let Some(parent) = path.parent()
            && let Err(error) = std::fs::create_dir_all(parent)
        {
            tracing::warn!(path = %parent.display(), %error, "creating the catalog cache directory");
            return;
        }
        // Write through a sibling temp file and rename so a crash or a
        // concurrent reader never sees a half-written catalog. The cache may
        // contain a private mirror URL/token, so keep it owner-only on Unix.
        let file_name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "catalog.json".to_owned());
        let temp = path.with_file_name(format!(
            ".{file_name}.tmp-{}-{}",
            std::process::id(),
            nanos()
        ));
        let write = || -> std::io::Result<()> {
            let mut options = std::fs::OpenOptions::new();
            options.create_new(true).write(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut file = options.open(&temp)?;
            file.write_all(&serialized)?;
            file.sync_all()?;
            drop(file);
            std::fs::rename(&temp, path)
        };
        if let Err(error) = write() {
            let _ = std::fs::remove_file(&temp);
            tracing::warn!(path = %path.display(), %error, "writing the catalog cache");
        } else {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
            }
        }
    }
}

fn validate_catalog_url(url: &str) -> Result<(), RegistryError> {
    if url.is_empty() || url.chars().any(char::is_control) {
        return Err(RegistryError::Unavailable(
            "registry URL must not be empty or contain control characters".to_owned(),
        ));
    }
    let Some((scheme, rest)) = url.split_once("://") else {
        return Err(RegistryError::Unavailable(format!(
            "registry URL must be absolute and use http or https: {}",
            redact_url(url)
        )));
    };
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return Err(RegistryError::Unavailable(format!(
            "registry URL must use http or https, not {scheme}"
        )));
    }
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    if authority.is_empty() || authority.contains('@') {
        return Err(RegistryError::Unavailable(
            "registry URL must name a host and must not contain credentials".to_owned(),
        ));
    }
    Ok(())
}

fn nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

impl ManifestRegistry for RemoteRegistry {
    fn manifests(&self) -> Result<Vec<RepoManifest>, RegistryError> {
        self.load(false).map(|(manifests, _)| manifests)
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    use olaya_core::{Capabilities, FileRole, ManifestFile, REPO_MANIFEST_SCHEMA, RepoManifest};

    use super::*;

    fn manifest() -> RepoManifest {
        RepoManifest {
            schema_version: REPO_MANIFEST_SCHEMA,
            name: "mock".to_owned(),
            aliases: vec!["mock:latest".to_owned()],
            family: "mock".to_owned(),
            engine: "candle".to_owned(),
            arch: "mock".to_owned(),
            format: "safetensors".to_owned(),
            layout: "mock-v1".to_owned(),
            subfolder: None,
            files: vec![ManifestFile {
                path: "model.safetensors".to_owned(),
                role: FileRole::Weights,
                sha256: Some("a".repeat(64)),
                size: Some(1),
                url: None,
            }],
            calibration_mode: Default::default(),
            capabilities: Capabilities::default(),
            router: None,
            license: "Apache-2.0".to_owned(),
            description: "mock".to_owned(),
            release_date: None,
            source: None,
            revision: None,
        }
    }

    fn temp_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "olaya-remote-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or_default()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("temp dir");
        path
    }

    fn serve_once(body: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let address = listener.local_addr().expect("address");
        let body = body.to_owned();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut request = [0u8; 2048];
            let _ = stream.read(&mut request);
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream
                .write_all(response.as_bytes())
                .expect("write response");
        });
        format!("http://{address}/registry-catalog.json")
    }

    #[test]
    fn fetches_then_uses_the_cache() {
        let dir = temp_dir();
        let cache = dir.join(CATALOG_CACHE_FILE);
        let bundle = CatalogBundle::new(vec![manifest()]);
        let body = serde_json::to_string(&bundle).expect("serialize");
        let url = serve_once(&body);
        let registry = RemoteRegistry::new(url, Some(cache.clone())).expect("registry");

        let (manifests, source) = registry.load(true).expect("fetch");
        assert_eq!(source, CatalogSource::Network);
        assert_eq!(manifests, vec![manifest()]);
        assert!(cache.is_file());

        let (cached, source) = registry.load(false).expect("cache");
        assert_eq!(source, CatalogSource::Cache);
        assert_eq!(cached, manifests);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn refresh_falls_back_to_the_cache() {
        let dir = temp_dir();
        let cache = dir.join(CATALOG_CACHE_FILE);
        let bundle = CatalogBundle::new(vec![manifest()]);
        std::fs::write(&cache, serde_json::to_vec(&bundle).expect("serialize")).expect("cache");
        let registry = RemoteRegistry::new("http://127.0.0.1:1/registry-catalog.json", Some(cache))
            .expect("registry");
        let (manifests, source) = registry.load(true).expect("cached fallback");
        assert_eq!(source, CatalogSource::Cache);
        assert_eq!(manifests, vec![manifest()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_catalogs_are_rejected() {
        let dir = temp_dir();
        let cache = dir.join(CATALOG_CACHE_FILE);
        let bundle = CatalogBundle {
            schema_version: olaya_core::CATALOG_SCHEMA_VERSION,
            manifests: Vec::new(),
        };
        std::fs::write(&cache, serde_json::to_vec(&bundle).expect("serialize")).expect("cache");
        let registry = RemoteRegistry::new("http://127.0.0.1:1/registry-catalog.json", Some(cache))
            .expect("registry");
        assert!(registry.load(false).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
