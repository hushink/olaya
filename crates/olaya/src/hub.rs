#[cfg(feature = "hub")]
use std::collections::HashMap;
#[cfg(feature = "hub")]
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(feature = "hub")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "hub")]
use std::sync::mpsc::RecvTimeoutError;
#[cfg(feature = "hub")]
use std::time::{Duration, Instant};

use olaya_core::{ManifestStore, RepoManifest, RuntimeError, StoreError};
#[cfg(feature = "hub")]
use sha2::{Digest, Sha256};

use crate::store::Store;

/// Default Hugging Face endpoint used for model downloads.
pub const DEFAULT_HF_ENDPOINT: &str = "https://hf-mirror.com";

/// Where models, downloads, and staging live.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HubConfig {
    /// Materialized model directories.
    pub models_dir: PathBuf,
    /// Resume sidecars, Hub caches, and the optional content cache.
    pub cache_dir: PathBuf,
    /// Per-task download staging before verified files move into `models_dir`.
    pub tmp_dir: PathBuf,
    /// Content-addressed OCI manifest/blob store.
    pub store_dir: PathBuf,
    /// Refuse every network access.
    pub offline: bool,
    /// Hugging Face endpoint. Defaults to `https://hf-mirror.com`; set
    /// `HF_ENDPOINT=https://huggingface.co` to use the official endpoint.
    pub hf_endpoint: Option<String>,
    /// ModelScope endpoint override.
    pub modelscope_endpoint: Option<String>,
    /// HTTP/SOCKS proxy for Hub requests.
    pub proxy: Option<String>,
}

impl Default for HubConfig {
    fn default() -> Self {
        let root = default_root();
        Self {
            models_dir: root.join("models"),
            cache_dir: root.join("cache"),
            tmp_dir: root.join("tmp"),
            store_dir: root.join("store"),
            offline: false,
            hf_endpoint: Some(DEFAULT_HF_ENDPOINT.to_owned()),
            modelscope_endpoint: None,
            proxy: None,
        }
    }
}

/// `~/.olaya`, or `$OLAYA_HOME` when set.
pub fn default_root() -> PathBuf {
    if let Some(root) = std::env::var_os("OLAYA_HOME") {
        let root = PathBuf::from(root);
        if !root.as_os_str().is_empty() {
            return root;
        }
    }
    directories::BaseDirs::new()
        .map(|dirs| dirs.home_dir().join(".olaya"))
        .unwrap_or_else(|| PathBuf::from(".olaya"))
}

impl HubConfig {
    /// Read `OLAYA_MODELS_DIR`, `OLAYA_CACHE_DIR`, `OLAYA_TMP_DIR`,
    /// `OLAYA_STORE_DIR`, `OLAYA_OFFLINE`, `HF_ENDPOINT`,
    /// `MODELSCOPE_ENDPOINT`, and `OLAYA_PROXY` over the defaults. Flags and
    /// config files take precedence at the CLI layer.
    pub fn from_env() -> Self {
        let mut config = Self::default();
        config.apply_env();
        config
    }

    /// Overlay the `OLAYA_*` / `HF_ENDPOINT` environment variables onto the
    /// current configuration.
    pub fn apply_env(&mut self) {
        if let Some(path) = non_empty_env("OLAYA_MODELS_DIR") {
            self.models_dir = PathBuf::from(path);
        }
        if let Some(path) = non_empty_env("OLAYA_CACHE_DIR") {
            self.cache_dir = PathBuf::from(path);
        }
        if let Some(path) = non_empty_env("OLAYA_TMP_DIR") {
            self.tmp_dir = PathBuf::from(path);
        }
        if let Some(path) = non_empty_env("OLAYA_STORE_DIR") {
            self.store_dir = PathBuf::from(path);
        }
        if let Some(endpoint) = non_empty_env("HF_ENDPOINT") {
            self.hf_endpoint = Some(endpoint.to_string_lossy().into_owned());
        }
        if let Some(endpoint) = non_empty_env("MODELSCOPE_ENDPOINT") {
            self.modelscope_endpoint = Some(endpoint.to_string_lossy().into_owned());
        }
        if let Some(proxy) = non_empty_env("OLAYA_PROXY") {
            self.proxy = Some(proxy.to_string_lossy().into_owned());
        }
        if let Some(offline) = env_bool("OLAYA_OFFLINE") {
            self.offline = offline;
        }
    }

    /// Reject layouts where a disposable cache/staging directory overlaps a
    /// model or blob directory. Deleting the former must never be able to
    /// remove the latter.
    pub fn validate_layout(&self) -> Result<(), RuntimeError> {
        let models = normalized_absolute(&self.models_dir)?;
        let store = normalized_absolute(&self.store_dir)?;
        let cache = normalized_absolute(&self.cache_dir)?;
        let tmp = normalized_absolute(&self.tmp_dir)?;
        for (label, path) in [("models_dir", &models), ("store_dir", &store)] {
            for (disposable, disposable_path) in [("cache_dir", &cache), ("tmp_dir", &tmp)] {
                if paths_overlap(path, disposable_path) {
                    return Err(RuntimeError::Storage {
                        message: format!(
                            "[hub].{disposable} ({}) must not overlap [hub].{label} ({}); \
                             deleting cache or staging would remove downloaded data",
                            disposable_path.display(),
                            path.display()
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn ensure_dirs(&self) -> Result<(), RuntimeError> {
        self.validate_layout()?;
        for (label, path) in [
            ("models", &self.models_dir),
            ("cache", &self.cache_dir),
            ("tmp", &self.tmp_dir),
            ("store", &self.store_dir),
        ] {
            std::fs::create_dir_all(path).map_err(|error| RuntimeError::Storage {
                message: format!(
                    "cannot create {label} directory {}: {error}",
                    path.display()
                ),
            })?;
        }
        Ok(())
    }

    /// The directory a model's files live in.
    pub fn model_dir(&self, name: &str) -> PathBuf {
        self.models_dir.join(name)
    }

    pub fn store_root(&self) -> &Path {
        &self.store_dir
    }
}

fn non_empty_env(key: &str) -> Option<std::ffi::OsString> {
    std::env::var_os(key).filter(|value| !value.is_empty())
}

fn normalized_absolute(path: &Path) -> Result<PathBuf, RuntimeError> {
    let absolute = std::path::absolute(path).map_err(|error| RuntimeError::Storage {
        message: format!("resolving {}: {error}", path.display()),
    })?;
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            component => normalized.push(component.as_os_str()),
        }
    }
    Ok(normalized)
}

fn paths_overlap(left: &Path, right: &Path) -> bool {
    path_contains(left, right) || path_contains(right, left)
}

fn path_contains(parent: &Path, child: &Path) -> bool {
    let parent = comparable_path(parent);
    let child = comparable_path(child);
    let parent = parent.trim_end_matches('/');
    child == parent || child.starts_with(&format!("{parent}/"))
}

fn comparable_path(path: &Path) -> String {
    let path = path.to_string_lossy().replace('\\', "/");
    if cfg!(windows) {
        path.to_ascii_lowercase()
    } else {
        path
    }
}

fn env_bool(key: &str) -> Option<bool> {
    let value = non_empty_env(key)?;
    let text = value.to_string_lossy();
    match parse_bool(&text) {
        Some(value) => Some(value),
        None => {
            tracing::warn!(variable = key, value = %text, "ignoring invalid boolean environment variable");
            None
        }
    }
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// The download/registry hub.
pub struct Hub {
    config: HubConfig,
    store: Option<Arc<dyn ManifestStore>>,
}

impl Hub {
    /// A hub backed by the default filesystem store in `config.store_dir`.
    pub fn new(config: HubConfig) -> Self {
        Self {
            config,
            store: None,
        }
    }

    /// A hub backed by a caller-supplied store.
    pub fn with_store(config: HubConfig, store: Arc<dyn ManifestStore>) -> Self {
        Self {
            config,
            store: Some(store),
        }
    }

    /// The configured store, or the filesystem store at `store_dir`.
    pub fn store(&self) -> Arc<dyn ManifestStore> {
        self.store
            .clone()
            .unwrap_or_else(|| Arc::new(Store::open(self.config.store_dir.clone())))
    }

    pub fn config(&self) -> &HubConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut HubConfig {
        &mut self.config
    }

    pub fn model_dir(&self, name: &str) -> PathBuf {
        self.config.model_dir(name)
    }

    /// Materialize `reference` into `models_dir/<name>` and return its manifest.
    ///
    /// `hint` is the catalog entry when the reference is already known (for
    /// example `laya`); otherwise the repository must ship
    /// `olaya.manifest.json`.
    pub fn pull(
        &self,
        reference: &str,
        hint: Option<&RepoManifest>,
    ) -> Result<RepoManifest, RuntimeError> {
        if self.config.offline {
            return Err(RuntimeError::Registry {
                message: "offline mode is enabled".to_owned(),
            });
        }
        // A missing cache/staging directory is not an error: recreate it.
        self.config.ensure_dirs()?;
        let manifest = match hint {
            Some(manifest) => manifest.clone(),
            None => self.fetch_manifest(reference)?,
        };
        manifest.validate().map_err(|error| RuntimeError::Storage {
            message: error.to_string(),
        })?;

        let destination = self.config.model_dir(&manifest.name);
        std::fs::create_dir_all(&destination).map_err(storage)?;
        // Also repairs files downloaded by older shaul layouts.
        self.normalize_downloaded(&destination, &manifest)?;
        let store = self.store();
        let stored = store
            .get(&manifest.name)
            .ok()
            .filter(|stored| crate::store::matches_manifest(stored, &manifest));

        if self.files_present(&manifest, &destination) {
            if stored.is_none() {
                import_with_cleanup(store.as_ref(), &manifest, &destination)?;
            }
            tracing::info!(model = %manifest.name, "model is already present");
            return Ok(manifest);
        }

        if let Some(stored) = &stored {
            tracing::info!(model = %manifest.name, "materializing model from the store");
            match store.materialize(stored, &destination) {
                Ok(()) if self.files_present(&manifest, &destination) => return Ok(manifest),
                Ok(()) => tracing::warn!(
                    model = %manifest.name,
                    "store materialization is incomplete; downloading instead"
                ),
                Err(error) => tracing::warn!(
                    model = %manifest.name,
                    %error,
                    "store materialization failed; downloading instead"
                ),
            }
        }

        let source = manifest
            .source
            .clone()
            .unwrap_or_else(|| reference.to_owned());
        tracing::info!(model = %manifest.name, %source, "pulling model");
        self.download(&source, &destination, &manifest)?;

        if !self.files_present(&manifest, &destination) {
            return Err(RuntimeError::Storage {
                message: format!(
                    "model {} is incomplete after pull in {}",
                    manifest.name,
                    destination.display()
                ),
            });
        }
        import_with_cleanup(store.as_ref(), &manifest, &destination)?;
        Ok(manifest)
    }

    fn files_present(&self, manifest: &RepoManifest, dir: &Path) -> bool {
        let base = model_base(&manifest.subfolder, dir);
        manifest.files.iter().all(|file| {
            let path = base.join(&file.path);
            match (std::fs::metadata(&path), file.size) {
                (Ok(metadata), Some(size)) => metadata.is_file() && metadata.len() == size,
                (Ok(metadata), None) => metadata.is_file(),
                _ => false,
            }
        })
    }

    fn download(
        &self,
        source: &str,
        destination: &Path,
        manifest: &RepoManifest,
    ) -> Result<(), RuntimeError> {
        if let Some(path) = local_path(source) {
            self.copy_local(&path, destination, manifest)?;
            return self.download_urls(destination, manifest);
        }
        // Files with an absolute URL (derived layers hosted outside the model
        // repository) are fetched separately; shaul only sees repository files.
        let paths = manifest
            .files
            .iter()
            .filter(|file| file.url.is_none())
            .map(|file| match &manifest.subfolder {
                Some(subfolder) => format!("{subfolder}/{}", file.path),
                None => file.path.clone(),
            })
            .collect::<Vec<_>>();
        if !paths.is_empty() {
            let verify_provider = !manifest.files.iter().all(|file| file.sha256.is_some());
            self.download_paths(
                source,
                destination,
                &paths,
                manifest.revision.as_deref(),
                verify_provider,
            )?;
        }
        self.normalize_downloaded(destination, manifest)?;
        self.download_urls(destination, manifest)
    }

    /// Fetch manifest files that declare an absolute `url` (derived decision or
    /// calibration layers that the model repository does not ship).
    #[cfg(feature = "hub")]
    fn download_urls(
        &self,
        destination: &Path,
        manifest: &RepoManifest,
    ) -> Result<(), RuntimeError> {
        let base = model_base(&manifest.subfolder, destination);
        let missing = manifest
            .files
            .iter()
            .filter(|file| file.url.is_some() && !base.join(&file.path).is_file())
            .collect::<Vec<_>>();
        if missing.is_empty() {
            return Ok(());
        }

        let mut builder =
            reqwest::blocking::Client::builder().timeout(std::time::Duration::from_secs(300));
        if let Some(proxy) = &self.config.proxy {
            let proxy = reqwest::Proxy::all(proxy).map_err(|error| RuntimeError::Registry {
                message: format!("invalid proxy {proxy:?}: {error}"),
            })?;
            builder = builder.proxy(proxy);
        }
        let client = builder.build().map_err(|error| RuntimeError::Registry {
            message: format!("building the HTTP client: {error}"),
        })?;
        let interactive = std::io::stderr().is_terminal();
        let mut progress = TerminalProgress::new(interactive);

        for file in missing {
            let url = resolve_hf_url(
                self.config.hf_endpoint.as_deref(),
                file.url.as_deref().expect("filtered by url"),
            );
            let target = base.join(&file.path);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(storage)?;
            }
            let temp = target.with_extension("part");
            let mut response = client
                .get(&url)
                .send()
                .and_then(reqwest::blocking::Response::error_for_status)
                .map_err(|error| RuntimeError::Registry {
                    message: format!("downloading {url}: {error}"),
                })?;
            let total = response.content_length();
            if interactive {
                progress.update(&progress_line(&file.path, 0, total));
            } else {
                tracing::info!(
                    file = %file.path,
                    total = %total.map(human_bytes).unwrap_or_else(|| "unknown".to_owned()),
                    "downloading file"
                );
            }
            let mut output = std::fs::File::create(&temp).map_err(storage)?;
            let mut hasher = Sha256::new();
            let mut size = 0u64;
            let mut last_update = Instant::now();
            let interval = if interactive {
                Duration::from_millis(100)
            } else {
                Duration::from_secs(1)
            };
            let mut buffer = [0u8; 64 * 1024];
            loop {
                let read = response
                    .read(&mut buffer)
                    .map_err(|error| RuntimeError::Registry {
                        message: format!("reading {url}: {error}"),
                    })?;
                if read == 0 {
                    break;
                }
                hasher.update(&buffer[..read]);
                output.write_all(&buffer[..read]).map_err(storage)?;
                size += read as u64;
                if last_update.elapsed() >= interval {
                    if interactive {
                        progress.update(&progress_line(&file.path, size, total));
                    } else {
                        tracing::info!(
                            file = %file.path,
                            downloaded = %human_bytes(size),
                            total = %total.map(human_bytes).unwrap_or_else(|| "unknown".to_owned()),
                            "download progress"
                        );
                    }
                    last_update = Instant::now();
                }
            }
            drop(output);
            if interactive {
                progress.update(&progress_line(&file.path, size, total.or(Some(size))));
            } else {
                tracing::info!(file = %file.path, size = %human_bytes(size), "downloaded file");
            }
            let digest = hex::encode(hasher.finalize());
            let invalid = file.size.is_some_and(|expected| expected != size)
                || file
                    .sha256
                    .as_deref()
                    .is_some_and(|expected| !expected.eq_ignore_ascii_case(&digest));
            if invalid {
                let _ = std::fs::remove_file(&temp);
                return Err(RuntimeError::Storage {
                    message: format!(
                        "{url} does not match the manifest (size {size}, sha256 {digest})"
                    ),
                });
            }
            std::fs::rename(&temp, &target).map_err(storage)?;
        }
        Ok(())
    }

    #[cfg(not(feature = "hub"))]
    fn download_urls(
        &self,
        _destination: &Path,
        _manifest: &RepoManifest,
    ) -> Result<(), RuntimeError> {
        Ok(())
    }

    /// shaul's Hub backend writes `<destination>/<repo-name>/<path>`; move the
    /// manifest's files to the layout the family expects.
    fn normalize_downloaded(
        &self,
        destination: &Path,
        manifest: &RepoManifest,
    ) -> Result<(), RuntimeError> {
        for file in &manifest.files {
            let relative = match &manifest.subfolder {
                Some(subfolder) => format!("{subfolder}/{}", file.path),
                None => file.path.clone(),
            };
            let target = destination.join(&relative);
            if target.is_file() {
                continue;
            }
            let Some(source) = find_downloaded(destination, &relative) else {
                continue;
            };
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(storage)?;
            }
            if target.exists() {
                std::fs::remove_file(&target).map_err(storage)?;
            }
            if std::fs::rename(&source, &target).is_err() {
                std::fs::copy(&source, &target).map_err(|error| RuntimeError::Storage {
                    message: format!(
                        "moving downloaded {} to {}: {error}",
                        source.display(),
                        target.display()
                    ),
                })?;
                let _ = std::fs::remove_file(&source);
            }
        }
        remove_empty_dirs(destination);
        Ok(())
    }

    fn copy_local(
        &self,
        source: &Path,
        destination: &Path,
        manifest: &RepoManifest,
    ) -> Result<(), RuntimeError> {
        let from_base = model_base(&manifest.subfolder, source);
        let to_base = model_base(&manifest.subfolder, destination);
        for file in &manifest.files {
            let from = from_base.join(&file.path);
            let to = to_base.join(&file.path);
            if from == to {
                continue;
            }
            if file.url.is_some() && !from.is_file() {
                // Fetched from its URL instead.
                continue;
            }
            if let Some(parent) = to.parent() {
                std::fs::create_dir_all(parent).map_err(storage)?;
            }
            std::fs::copy(&from, &to).map_err(|error| RuntimeError::Storage {
                message: format!("copying {} to {}: {error}", from.display(), to.display()),
            })?;
        }
        Ok(())
    }

    #[cfg(feature = "hub")]
    fn download_paths(
        &self,
        source: &str,
        destination: &Path,
        paths: &[String],
        revision: Option<&str>,
        verify_provider: bool,
    ) -> Result<(), RuntimeError> {
        let mut config = shaul::Config::default();
        config.state_dir = Some(self.config.cache_dir.join("state"));
        config.hub.hf_endpoint = self.config.hf_endpoint.clone();
        config.hub.modelscope_endpoint = self.config.modelscope_endpoint.clone();
        config.network.proxy = self.config.proxy.clone();
        config.download.content_cache = false;
        config.download.resume = true;

        let engine = shaul::default_blocking_engine(config).map_err(registry)?;
        let link = shaul::Link::new(source).map_err(registry)?;
        // shaul treats a Hub non-LFS file's git blob id as a plain SHA-1, so
        // its provider verification rejects valid files. Disable it here and
        // re-verify every authoritative sha256 (LFS blobs) after the download
        // instead; manifests that declare all sha256 values are verified by
        // Store::import.
        let verify_after_download = verify_provider;
        if verify_after_download {
            tracing::debug!(
                endpoint = self.config.hf_endpoint.as_deref().unwrap_or("-"),
                "provider git-blob checksums are not enforced; \
                 sha256 checksums are verified after download"
            );
        }
        let options = shaul::TaskOptions {
            include: paths.to_vec(),
            overwrite: true,
            // Verification happens below (authoritative sha256) or in
            // Store::import (manifests that declare every file's sha256).
            verify: Some(shaul::VerifyMode::Disabled),
            revision: revision.map(str::to_owned),
            ..Default::default()
        };

        // shaul reports byte-level progress on a channel; forward it to the
        // log while the blocking download runs.
        let events = engine.subscribe();
        let stop = Arc::new(AtomicBool::new(false));
        let progress_stop = Arc::clone(&stop);
        let progress = std::thread::spawn(move || log_download_progress(events, progress_stop));

        let downloaded = engine
            .download(link, destination.to_path_buf(), options)
            .map_err(registry);
        stop.store(true, Ordering::Relaxed);
        let _ = progress.join();
        let task = downloaded?;
        if verify_after_download {
            verify_task_checksums(destination, &task)?;
        }

        tracing::info!(
            state = ?task.state,
            files = task.files.len(),
            bytes = task.files.iter().map(|file| file.downloaded).sum::<u64>(),
            "download finished"
        );
        Ok(())
    }

    #[cfg(not(feature = "hub"))]
    fn download_paths(
        &self,
        _source: &str,
        _destination: &Path,
        _paths: &[String],
        _revision: Option<&str>,
        _verify_provider: bool,
    ) -> Result<(), RuntimeError> {
        Err(RuntimeError::Unsupported {
            message: "hub support is not compiled in; enable the `hub` feature".to_owned(),
        })
    }

    /// Fetch `olaya.manifest.json` from a repository that ships one.
    fn fetch_manifest(&self, reference: &str) -> Result<RepoManifest, RuntimeError> {
        if local_path(reference).is_some() {
            let path = local_path(reference).expect("checked above");
            let manifest_path = path.join(olaya_core::REPO_MANIFEST_FILE);
            let bytes = std::fs::read(&manifest_path).map_err(|error| RuntimeError::Storage {
                message: format!("reading {}: {error}", manifest_path.display()),
            })?;
            return serde_json::from_slice(&bytes).map_err(|error| RuntimeError::Storage {
                message: format!("parsing {}: {error}", manifest_path.display()),
            });
        }

        let staging = self
            .config
            .tmp_dir
            .join("manifests")
            .join(sanitize_reference(reference));
        std::fs::create_dir_all(&staging).map_err(storage)?;
        self.download_paths(
            reference,
            &staging,
            &[olaya_core::REPO_MANIFEST_FILE.to_owned()],
            None,
            true,
        )?;
        let manifest_path = find_downloaded(&staging, olaya_core::REPO_MANIFEST_FILE).ok_or_else(
            || RuntimeError::Registry {
                message: format!(
                    "{reference} has no {}; publish one in the repository or add a catalog entry",
                    olaya_core::REPO_MANIFEST_FILE
                ),
            },
        )?;
        let bytes = std::fs::read(&manifest_path).map_err(|error| RuntimeError::Registry {
            message: format!("reading {}: {error}", manifest_path.display()),
        })?;
        serde_json::from_slice(&bytes).map_err(|error| RuntimeError::Registry {
            message: format!("parsing {}: {error}", manifest_path.display()),
        })
    }
}

/// Re-verify the checksums shaul knows are authoritative (currently LFS
/// sha256) when its provider verification is bypassed.
#[cfg(feature = "hub")]
fn verify_task_checksums(destination: &Path, task: &shaul::Task) -> Result<(), RuntimeError> {
    for file in &task.files {
        if !file.selected {
            continue;
        }
        let Some(expected) = &file.checksum else {
            continue;
        };
        if expected.algorithm != shaul::ChecksumAlgorithm::Sha256 {
            continue;
        }
        let path =
            find_downloaded(destination, &file.path).ok_or_else(|| RuntimeError::Storage {
                message: format!("downloaded file {} is missing", file.path),
            })?;
        let actual = sha256_file(&path)?;
        if !actual.eq_ignore_ascii_case(&expected.hex) {
            return Err(RuntimeError::Registry {
                message: format!(
                    "checksum mismatch for {}: expected sha256:{}, got sha256:{actual}",
                    file.path, expected.hex
                ),
            });
        }
    }
    Ok(())
}

#[cfg(feature = "hub")]
fn sha256_file(path: &Path) -> Result<String, RuntimeError> {
    let mut file = std::fs::File::open(path).map_err(storage)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(storage)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(feature = "hub")]
fn log_download_progress(
    events: std::sync::mpsc::Receiver<shaul::DownloadEvent>,
    stop: Arc<AtomicBool>,
) {
    let interactive = std::io::stderr().is_terminal();
    let mut progress = TerminalProgress::new(interactive);
    let mut last_logged = HashMap::new();
    let mut last_render = None;
    loop {
        let event = match events.recv_timeout(Duration::from_millis(250)) {
            Ok(event) => event,
            Err(RecvTimeoutError::Timeout) => {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => break,
        };

        match event.kind {
            shaul::EventKind::FileStarted {
                file, total_bytes, ..
            } => {
                if interactive {
                    progress.update(&progress_line(&file, 0, total_bytes));
                    last_render = Some(Instant::now());
                } else {
                    last_logged.insert(file.clone(), Instant::now());
                    tracing::info!(
                        file = %file,
                        total = %total_bytes.map(human_bytes).unwrap_or_else(|| "unknown".to_owned()),
                        "downloading file"
                    );
                }
            }
            shaul::EventKind::Progress {
                file,
                downloaded_bytes,
                total_bytes,
                ..
            } => {
                let now = Instant::now();
                if interactive {
                    let due = last_render.is_none_or(|previous| {
                        now.duration_since(previous) >= Duration::from_millis(100)
                    });
                    if due {
                        progress.update(&progress_line(&file, downloaded_bytes, total_bytes));
                        last_render = Some(now);
                    }
                } else {
                    let due = last_logged.get(&file).is_none_or(|previous| {
                        now.duration_since(*previous) >= Duration::from_secs(1)
                    });
                    if due {
                        last_logged.insert(file.clone(), now);
                        tracing::info!(
                            file = %file,
                            downloaded = %human_bytes(downloaded_bytes),
                            total = %total_bytes.map(human_bytes).unwrap_or_else(|| "unknown".to_owned()),
                            "download progress"
                        );
                    }
                }
            }
            shaul::EventKind::FileCompleted { file, size, .. } => {
                if interactive {
                    progress.update(&progress_line(&file, size, Some(size)));
                } else {
                    last_logged.remove(&file);
                    tracing::info!(file = %file, size = %human_bytes(size), "downloaded file");
                }
            }
            shaul::EventKind::Warning { message, .. } => {
                if interactive {
                    progress.clear();
                }
                tracing::warn!(%message, "download warning");
            }
            _ => {}
        }
    }
    progress.clear();
}

#[cfg(feature = "hub")]
struct TerminalProgress {
    active: bool,
    last_len: usize,
}

#[cfg(feature = "hub")]
impl TerminalProgress {
    fn new(active: bool) -> Self {
        Self {
            active,
            last_len: 0,
        }
    }

    fn update(&mut self, line: &str) {
        if !self.active {
            return;
        }
        let width = line.chars().count();
        let padding = self.last_len.saturating_sub(width);
        let mut stderr = std::io::stderr().lock();
        let _ = write!(stderr, "\r{line}{}", " ".repeat(padding));
        let _ = stderr.flush();
        self.last_len = width;
    }

    fn clear(&mut self) {
        if !self.active || self.last_len == 0 {
            return;
        }
        let mut stderr = std::io::stderr().lock();
        let _ = write!(stderr, "\r{}\r", " ".repeat(self.last_len));
        let _ = stderr.flush();
        self.last_len = 0;
    }
}

#[cfg(feature = "hub")]
impl Drop for TerminalProgress {
    fn drop(&mut self) {
        self.clear();
    }
}

#[cfg(feature = "hub")]
fn progress_line(file: &str, downloaded: u64, total: Option<u64>) -> String {
    let name = if file.chars().count() > 48 {
        let mut name = file.chars().take(45).collect::<String>();
        name.push_str("...");
        name
    } else {
        file.to_owned()
    };
    match total.filter(|total| *total > 0) {
        Some(total) => {
            let percent = downloaded.min(total).saturating_mul(100) / total;
            let filled = (percent as usize * 20 / 100).min(20);
            format!(
                "pulling {name} [{}{}] {percent:>3}% {}/{}",
                "#".repeat(filled),
                "-".repeat(20 - filled),
                human_bytes(downloaded),
                human_bytes(total)
            )
        }
        None => format!("pulling {name} {}", human_bytes(downloaded)),
    }
}

#[cfg(feature = "hub")]
fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(feature = "hub")]
fn resolve_hf_url(endpoint: Option<&str>, url: &str) -> String {
    let Some(endpoint) = endpoint else {
        return url.to_owned();
    };
    const PREFIXES: [&str; 4] = [
        "https://huggingface.co/",
        "http://huggingface.co/",
        "https://www.huggingface.co/",
        "http://www.huggingface.co/",
    ];
    for prefix in PREFIXES {
        if let Some(path) = url.strip_prefix(prefix) {
            return format!("{}/{}", endpoint.trim_end_matches('/'), path);
        }
    }
    url.to_owned()
}

fn import_manifest(
    store: &dyn ManifestStore,
    manifest: &RepoManifest,
    destination: &Path,
) -> Result<(), StoreError> {
    store.put(manifest, destination).map(|_| ())
}

/// Import a model, removing a corrupt source file so that the next pull
/// downloads it again instead of failing on the same bytes forever.
fn import_with_cleanup(
    store: &dyn ManifestStore,
    manifest: &RepoManifest,
    destination: &Path,
) -> Result<(), RuntimeError> {
    import_manifest(store, manifest, destination).map_err(|error| {
        if let StoreError::DigestMismatch { path, .. } | StoreError::SizeMismatch { path, .. } =
            &error
            && path.starts_with(destination)
        {
            let _ = std::fs::remove_file(path);
        }
        RuntimeError::Storage {
            message: format!("storing {}: {error}", manifest.name),
        }
    })
}

/// Find a downloaded file by its repository-relative path. shaul may add a
/// `<repo-name>/` prefix, so accept any suffix match.
fn find_downloaded(root: &Path, relative: &str) -> Option<PathBuf> {
    let relative = relative.replace('\\', "/");
    let direct = root.join(relative.replace('/', std::path::MAIN_SEPARATOR_STR));
    if direct.is_file() {
        return Some(direct);
    }
    let mut matches = Vec::new();
    collect_downloaded(root, root, &relative, &mut matches);
    matches.sort_by_key(|path| path.components().count());
    matches.into_iter().next()
}

fn collect_downloaded(root: &Path, dir: &Path, relative: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_downloaded(root, &path, relative, out);
        } else if path.is_file() {
            let Ok(stripped) = path.strip_prefix(root) else {
                continue;
            };
            let stripped = stripped.to_string_lossy().replace('\\', "/");
            if matches_downloaded(&stripped, relative) {
                out.push(path);
            }
        }
    }
}

/// The expected file, either at the root or under one `<repo-name>/` prefix.
fn matches_downloaded(stripped: &str, relative: &str) -> bool {
    if stripped == relative {
        return true;
    }
    stripped
        .strip_suffix(&format!("/{relative}"))
        .is_some_and(|prefix| !prefix.contains('/'))
}

fn remove_empty_dirs(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            remove_empty_dirs(&path);
            let _ = std::fs::remove_dir(&path);
        }
    }
}

fn model_base(subfolder: &Option<String>, dir: &Path) -> PathBuf {
    match subfolder {
        Some(subfolder) => dir.join(subfolder),
        None => dir.to_path_buf(),
    }
}

fn local_path(reference: &str) -> Option<PathBuf> {
    if let Some(rest) = reference.strip_prefix("local:") {
        return Some(PathBuf::from(rest));
    }
    if let Some(rest) = reference.strip_prefix("file://") {
        return Some(PathBuf::from(rest));
    }
    let path = PathBuf::from(reference);
    path.exists().then_some(path)
}

fn sanitize_reference(reference: &str) -> String {
    reference
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn storage(error: std::io::Error) -> RuntimeError {
    RuntimeError::Storage {
        message: error.to_string(),
    }
}

#[cfg(feature = "hub")]
fn registry(error: shaul::Error) -> RuntimeError {
    RuntimeError::Registry {
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use olaya_core::{CalibrationMode, Capabilities, FileRole, ManifestFile};
    use tempfile::tempdir;

    fn manifest(subfolder: Option<&str>, files: &[(&str, u64)]) -> RepoManifest {
        RepoManifest {
            schema_version: 1,
            name: "tiny".to_owned(),
            aliases: Vec::new(),
            family: "tiny".to_owned(),
            engine: "candle".to_owned(),
            arch: "custom".to_owned(),
            format: "safetensors".to_owned(),
            layout: "tiny-v1".to_owned(),
            subfolder: subfolder.map(str::to_owned),
            files: files
                .iter()
                .map(|(path, size)| ManifestFile {
                    path: (*path).to_owned(),
                    role: FileRole::Other,
                    sha256: None,
                    size: Some(*size),
                    url: None,
                })
                .collect(),
            calibration_mode: CalibrationMode::Raw,
            capabilities: Capabilities::classification(),
            router: None,
            license: "Apache-2.0".to_owned(),
            description: "tiny".to_owned(),
            release_date: None,
            source: None,
            revision: None,
        }
    }

    fn hub() -> Hub {
        Hub::new(HubConfig::default())
    }

    #[test]
    fn rejects_disposable_directories_overlapping_models_or_store() {
        let mut config = HubConfig::default();
        config.models_dir = PathBuf::from("data/models");
        config.cache_dir = config.models_dir.join("cache");
        assert!(config.validate_layout().is_err());

        config.models_dir = PathBuf::from("data/models");
        config.cache_dir = PathBuf::from("data/cache");
        config.store_dir = PathBuf::from("data/cache/store");
        assert!(config.validate_layout().is_err());
    }

    #[test]
    fn boolean_environment_values_accept_common_spellings() {
        assert_eq!(parse_bool("true"), Some(true));
        assert_eq!(parse_bool("YES"), Some(true));
        assert_eq!(parse_bool("0"), Some(false));
        assert_eq!(parse_bool(" off "), Some(false));
        assert_eq!(parse_bool("maybe"), None);
    }

    #[test]
    fn hugging_face_uses_the_mirror_by_default() {
        assert_eq!(
            HubConfig::default().hf_endpoint.as_deref(),
            Some(DEFAULT_HF_ENDPOINT)
        );
    }

    #[test]
    fn progress_line_shows_a_bar_and_percent() {
        let line = progress_line("model.safetensors", 512, Some(1024));
        assert!(line.contains("pulling model.safetensors"));
        assert!(line.contains("50%"));
        assert!(line.contains("512 B/1.0 KiB"));
    }

    #[test]
    fn hugging_face_urls_are_rewritten_to_the_configured_endpoint() {
        assert_eq!(
            resolve_hf_url(
                Some("https://hf-mirror.com/"),
                "https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/resolve/abc/config.json"
            ),
            "https://hf-mirror.com/Qwen/Qwen3.5-0.8B-Base/resolve/abc/config.json"
        );
        assert_eq!(
            resolve_hf_url(
                Some("https://huggingface.co"),
                "https://www.huggingface.co/Qwen/model.bin?download=1"
            ),
            "https://huggingface.co/Qwen/model.bin?download=1"
        );
        assert_eq!(
            resolve_hf_url(Some("https://hf-mirror.com"), "https://ollaya.dev/blobs/x"),
            "https://ollaya.dev/blobs/x"
        );
        assert_eq!(
            resolve_hf_url(None, "https://huggingface.co/Qwen/model.bin"),
            "https://huggingface.co/Qwen/model.bin"
        );
    }

    #[test]
    fn normalizes_repo_prefixed_downloads() {
        let root = tempdir().expect("tempdir");
        let repo = root.path().join("Repo-Name");
        std::fs::create_dir_all(repo.join("tokenizer")).expect("tokenizer dir");
        std::fs::write(repo.join("model.safetensors"), b"weights").expect("weight");
        std::fs::write(repo.join("tokenizer/tokenizer.json"), b"{}").expect("tokenizer");

        let manifest = manifest(
            None,
            &[("model.safetensors", 7), ("tokenizer/tokenizer.json", 2)],
        );
        hub()
            .normalize_downloaded(root.path(), &manifest)
            .expect("normalize");

        assert_eq!(
            std::fs::read(root.path().join("model.safetensors")).expect("weight"),
            b"weights"
        );
        assert!(root.path().join("tokenizer/tokenizer.json").is_file());
        assert!(!repo.exists(), "empty repo prefix should be removed");
    }

    #[test]
    fn normalizes_a_subfolder_download() {
        let root = tempdir().expect("tempdir");
        let repo = root.path().join("Repo-Name");
        std::fs::create_dir_all(repo.join("multilingual")).expect("subfolder");
        std::fs::write(repo.join("multilingual/model.safetensors"), b"multi").expect("weight");

        let manifest = manifest(Some("multilingual"), &[("model.safetensors", 5)]);
        hub()
            .normalize_downloaded(root.path(), &manifest)
            .expect("normalize");

        assert_eq!(
            std::fs::read(root.path().join("multilingual/model.safetensors")).expect("weight"),
            b"multi"
        );
        assert!(!repo.exists(), "empty repo prefix should be removed");
    }

    #[test]
    fn matches_only_one_repo_prefix() {
        assert!(matches_downloaded("config.json", "config.json"));
        assert!(matches_downloaded("Repo/config.json", "config.json"));
        assert!(!matches_downloaded(
            "Repo/encoder/config.json",
            "config.json"
        ));
        assert!(matches_downloaded(
            "Repo/encoder/config.json",
            "encoder/config.json"
        ));
    }

    #[test]
    fn finds_a_nested_repository_manifest() {
        let root = tempdir().expect("tempdir");
        let repo = root.path().join("Repo");
        std::fs::create_dir_all(&repo).expect("repo dir");
        std::fs::write(repo.join(olaya_core::REPO_MANIFEST_FILE), b"{}").expect("manifest");

        let found = find_downloaded(root.path(), olaya_core::REPO_MANIFEST_FILE).expect("found");
        assert_eq!(found, repo.join(olaya_core::REPO_MANIFEST_FILE));
    }
}
