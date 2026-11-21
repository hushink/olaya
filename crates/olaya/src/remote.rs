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

use std::path::{Path, PathBuf};
use std::time::Duration;

use olaya_core::{CatalogBundle, ManifestRegistry, RegistryError, RepoManifest};
use reqwest::blocking::Client;

/// Environment variable that overrides the catalog URL.
pub const REGISTRY_URL_ENV: &str = "OLAYA_REGISTRY_URL";
/// Default hosted catalog: the latest GitHub Release's `registry-catalog.json`
/// asset. Override with `OLAYA_REGISTRY_URL` for a versioned asset or an
/// internal mirror.
pub const DEFAULT_REGISTRY_URL: &str =
    "https://github.com/hushink/olaya/releases/latest/download/registry-catalog.json";
/// Cache file name under the hub cache directory.
pub const CATALOG_CACHE_FILE: &str = "registry-catalog.json";

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
    /// Build a registry. `cache` is the catalog bundle path; pass `None` to
    /// disable caching.
    pub fn new(url: impl Into<String>, cache: Option<PathBuf>) -> Result<Self, RegistryError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent(concat!("olaya/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|error| {
                RegistryError::Unavailable(format!("building the HTTP client: {error}"))
            })?;
        Ok(Self {
            url: url.into(),
            cache,
            client,
        })
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
                        url = %self.url,
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
        tracing::info!(url = %self.url, "fetching model catalog");
        let response = self.client.get(&self.url).send().map_err(|error| {
            RegistryError::Unavailable(format!("fetching {}: {error}", self.url))
        })?;
        let status = response.status();
        if !status.is_success() {
            return Err(RegistryError::Unavailable(format!(
                "fetching {} returned HTTP {status}",
                self.url
            )));
        }
        let bytes = response.bytes().map_err(|error| {
            RegistryError::Unavailable(format!("reading {}: {error}", self.url))
        })?;
        let bundle: CatalogBundle = serde_json::from_slice(&bytes).map_err(|source| {
            RegistryError::Unavailable(format!("parsing {}: {source}", self.url))
        })?;
        bundle.validate()?;
        let mut manifests = bundle.manifests;
        manifests.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(manifests)
    }

    fn read_cache(&self, path: &Path) -> Result<Vec<RepoManifest>, RegistryError> {
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
        if let Err(error) = std::fs::write(path, serialized) {
            tracing::warn!(path = %path.display(), %error, "writing the catalog cache");
        }
    }
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
