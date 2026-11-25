mod config;
mod registry_seed;

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use config::LoadedConfig;
#[cfg(feature = "serve")]
use olaya::DecisionRuntime;
use olaya::{BuiltinFamilies, Engine, HubConfig, Question, Questions, RepoManifest, State};
use serde_json::json;

#[derive(Debug, Parser)]
#[command(name = "olaya", version, about = "System One decision-model runtime")]
struct Cli {
    #[command(flatten)]
    global: GlobalArgs,
    #[command(subcommand)]
    command: Command,
}

/// Settings shared by every subcommand. Values here follow
/// CLI > environment > file > built-in default.
#[derive(Debug, Args)]
struct GlobalArgs {
    /// Configuration file (default: ./olaya.toml, ./sys1.toml, or ~/.olaya/config.toml).
    #[arg(long, env = "OLAYA_CONFIG", global = true)]
    config: Option<PathBuf>,
    /// Directory containing `olaya.manifest.json` files.
    #[arg(long, env = "OLAYA_REGISTRY_DIR", global = true)]
    registry_dir: Option<PathBuf>,
    /// Use the content store as the manifest registry (no registry directory).
    #[arg(
        long,
        env = "OLAYA_REGISTRY_STORE",
        global = true,
        num_args = 0..=1,
        default_missing_value = "true",
        require_equals = true
    )]
    registry_store: Option<bool>,
    /// Hosted catalog bundle URL.
    #[arg(long, env = "OLAYA_REGISTRY_URL", global = true)]
    registry_url: Option<String>,
    /// HTTP timeout for the hosted catalog bundle, in seconds.
    #[arg(long, env = "OLAYA_REGISTRY_TIMEOUT_SECS", global = true)]
    registry_timeout_secs: Option<u64>,
    /// Materialized model directories.
    #[arg(long, env = "OLAYA_MODELS_DIR", global = true)]
    models_dir: Option<PathBuf>,
    /// Downloads, staging, and catalog cache.
    #[arg(long, env = "OLAYA_CACHE_DIR", global = true)]
    cache_dir: Option<PathBuf>,
    /// Per-task download staging.
    #[arg(long, env = "OLAYA_TMP_DIR", global = true)]
    tmp_dir: Option<PathBuf>,
    /// Content-addressed OCI store root.
    #[arg(long, env = "OLAYA_STORE_DIR", global = true)]
    store_dir: Option<PathBuf>,
    /// Hugging Face endpoint.
    #[arg(long, env = "HF_ENDPOINT", global = true)]
    hf_endpoint: Option<String>,
    /// ModelScope endpoint.
    #[arg(long, env = "MODELSCOPE_ENDPOINT", global = true)]
    modelscope_endpoint: Option<String>,
    /// HTTP/SOCKS proxy for model downloads.
    #[arg(long, env = "OLAYA_PROXY", global = true)]
    proxy: Option<String>,
    /// Comma-separated hosts that bypass the proxy.
    #[arg(long, env = "OLAYA_NO_PROXY", global = true)]
    no_proxy: Option<String>,
    /// Allow manifests to reference loopback/private network endpoints.
    #[arg(
        long,
        env = "OLAYA_ALLOW_PRIVATE_NETWORK",
        global = true,
        num_args = 0..=1,
        default_missing_value = "true",
        require_equals = true
    )]
    allow_private_network: Option<bool>,
    /// Download idle/read timeout in seconds.
    #[arg(long, env = "OLAYA_DOWNLOAD_TIMEOUT_SECS", global = true)]
    download_timeout_secs: Option<u64>,
    /// Download TCP connect timeout in seconds.
    #[arg(long, env = "OLAYA_CONNECT_TIMEOUT_SECS", global = true)]
    connect_timeout_secs: Option<u64>,
    /// Retry attempts for failed HTTP downloads.
    #[arg(long, env = "OLAYA_DOWNLOAD_RETRIES", global = true)]
    download_retries: Option<u32>,
    /// Refuse every network access.
    #[arg(
        long,
        env = "OLAYA_OFFLINE",
        global = true,
        num_args = 0..=1,
        default_missing_value = "true",
        require_equals = true
    )]
    offline: Option<bool>,
    /// Device preference: auto, cpu, cuda, cuda:<index>, or metal.
    #[arg(long, env = "OLAYA_DEVICE", global = true)]
    device: Option<String>,
    /// Weight dtype on CUDA/Metal: f32, f16, or bf16.
    #[arg(long, env = "OLAYA_DTYPE", global = true)]
    dtype: Option<String>,
    /// Allow TF32 tensor-core GEMMs on CUDA.
    #[arg(
        long,
        env = "OLAYA_TF32",
        global = true,
        num_args = 0..=1,
        default_missing_value = "true",
        require_equals = true
    )]
    tf32: Option<bool>,
    /// Model used when a request leaves the model name empty.
    #[arg(long, env = "OLAYA_DEFAULT_MODEL", global = true)]
    default_model: Option<String>,
    /// Maximum number of models kept loaded at once.
    #[arg(long, env = "OLAYA_MAX_LOADED", global = true)]
    max_loaded: Option<usize>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Serve the TypeSafe-compatible HTTP API and the model management API.
    #[cfg(feature = "serve")]
    Serve(ServeArgs),
    /// Run one decision request.
    Ask(AskArgs),
    /// Inspect and manage the local model catalog.
    Models(ModelsArgs),
    /// Print storage paths and environment diagnostics.
    Doctor(DoctorArgs),
}

#[cfg(feature = "serve")]
#[derive(Debug, Args, Default)]
struct ServeArgs {
    /// Bind address.
    #[arg(long, env = "OLAYA_HOST")]
    host: Option<String>,
    /// HTTP port.
    #[arg(long, env = "OLAYA_PORT")]
    port: Option<u16>,
    /// Bearer token for every route except /health.
    #[arg(long, env = "OLAYA_API_KEY")]
    api_key: Option<String>,
    /// Allow binding a non-loopback address without an API key.
    #[arg(
        long,
        env = "OLAYA_ALLOW_UNAUTHENTICATED",
        num_args = 0..=1,
        default_missing_value = "true",
        require_equals = true
    )]
    allow_unauthenticated: Option<bool>,
    /// Requests collected into one batched forward pass (1 disables batching).
    #[arg(long, env = "OLAYA_MAX_BATCH")]
    max_batch: Option<usize>,
    /// Collection window for the first request of a batch.
    #[arg(long, env = "OLAYA_BATCH_WAIT_MS")]
    batch_wait_ms: Option<u64>,
    /// Maximum accepted HTTP request body size.
    #[arg(long, env = "OLAYA_MAX_BODY_BYTES")]
    max_body_bytes: Option<usize>,
    /// Maximum queued requests before backpressure.
    #[arg(long, env = "OLAYA_MAX_QUEUE")]
    max_queue: Option<usize>,
    /// Cancel one HTTP request (including body reads) after this many seconds.
    #[arg(long, env = "OLAYA_REQUEST_TIMEOUT_SECS")]
    request_timeout_secs: Option<u64>,
    /// Maximum concurrent /api/pull and /api/load operations.
    #[arg(long, env = "OLAYA_MAX_OPERATIONS")]
    max_operations: Option<usize>,
    /// Browser origin allowed to call the API (repeatable or comma-separated).
    #[arg(
        long = "cors-origin",
        env = "OLAYA_CORS_ORIGINS",
        value_delimiter = ','
    )]
    cors_origins: Vec<String>,
    /// Additional Host header names accepted by the server.
    #[arg(
        long = "allowed-host",
        env = "OLAYA_ALLOWED_HOSTS",
        value_delimiter = ','
    )]
    allowed_hosts: Vec<String>,
}

#[derive(Debug, Args)]
struct AskArgs {
    /// Model to call; falls back to --default-model / [engine].default_model.
    #[arg(long, env = "OLAYA_MODEL")]
    model: Option<String>,
    /// A JSON string, object or array; `-` reads stdin.
    #[arg(long)]
    state: String,
    /// A JSON file containing the `questions` object.
    #[arg(long)]
    questions: PathBuf,
}

#[derive(Debug, Args)]
struct ModelsArgs {
    #[command(subcommand)]
    command: ModelsCommand,
}

#[derive(Debug, Subcommand)]
enum ModelsCommand {
    /// List manifests in the catalog.
    List {
        /// Include every alias as a separate entry.
        #[arg(long)]
        all: bool,
    },
    /// Show one catalog manifest.
    Show { model: String },
    /// Pull a model into the model directory and the content store.
    Pull { model: String },
    /// Add a manifest to the filesystem registry.
    Add {
        manifest: PathBuf,
        /// Overwrite an existing manifest with different contents.
        #[arg(long)]
        force: bool,
    },
    /// Remove a model's registry manifest, or purge all of its local files.
    Remove {
        model: String,
        /// Also delete materialized weights, store manifests, and the blobs
        /// that no remaining model references.
        #[arg(long)]
        purge: bool,
    },
    /// Export the catalog (remote bundle or store) into a registry directory.
    Export {
        /// Target directory (defaults to the resolved registry directory).
        #[arg(long)]
        dir: Option<PathBuf>,
        /// Overwrite manifests that exist with different contents.
        #[arg(long)]
        force: bool,
    },
    /// List stored model names in the content store.
    Stored,
    /// Print the stored OCI manifest, config, and digest.
    Manifest { model: String },
    /// Re-hash every stored blob for `model`.
    Verify { model: String },
}

#[derive(Debug, Args)]
struct DoctorArgs {}

#[derive(Debug)]
struct Settings {
    config_path: Option<PathBuf>,
    engine: EngineSettings,
    hub: HubConfig,
    registry: RegistrySettings,
}

#[derive(Debug)]
struct EngineSettings {
    device: String,
    dtype: String,
    tf32: bool,
    default_model: Option<String>,
    max_loaded: usize,
}

#[derive(Debug)]
struct RegistrySettings {
    dir: PathBuf,
    store: bool,
    url: String,
    timeout_secs: u64,
    allow_remote: bool,
}

impl Settings {
    fn load(loaded: &LoadedConfig, global: &GlobalArgs) -> Result<Self> {
        let file = &loaded.file;

        let mut hub = HubConfig::default();
        if let Some(path) = &file.hub.models_dir {
            hub.models_dir = config::resolve_path(&loaded.base, path);
        }
        if let Some(path) = &file.hub.cache_dir {
            hub.cache_dir = config::resolve_path(&loaded.base, path);
        }
        if let Some(path) = &file.hub.tmp_dir {
            hub.tmp_dir = config::resolve_path(&loaded.base, path);
        }
        if let Some(path) = &file.hub.store_dir {
            hub.store_dir = config::resolve_path(&loaded.base, path);
        }
        if let Some(offline) = file.hub.offline {
            hub.offline = offline;
        }
        if let Some(endpoint) = clean(file.hub.hf_endpoint.as_deref()) {
            hub.hf_endpoint = Some(endpoint.to_owned());
        }
        if let Some(endpoint) = clean(file.hub.modelscope_endpoint.as_deref()) {
            hub.modelscope_endpoint = Some(endpoint.to_owned());
        }
        if let Some(proxy) = file.hub.proxy.as_deref() {
            // Preserve an explicit empty value: shaul treats it as
            // "bypass ambient proxy environment variables".
            hub.proxy = Some(proxy.trim().to_owned());
        }
        if let Some(no_proxy) = clean(file.hub.no_proxy.as_deref()) {
            hub.no_proxy = Some(no_proxy.to_owned());
        }
        if let Some(allow) = file.hub.allow_private_network {
            hub.allow_private_network = allow;
        }
        if let Some(seconds) = file.hub.download_timeout_secs {
            hub.download_timeout_secs = seconds;
        }
        if let Some(seconds) = file.hub.connect_timeout_secs {
            hub.connect_timeout_secs = seconds;
        }
        if let Some(retries) = file.hub.download_retries {
            hub.download_retries = retries;
        }
        hub.apply_env();

        if let Some(path) = &global.models_dir {
            hub.models_dir = path.clone();
        }
        if let Some(path) = &global.cache_dir {
            hub.cache_dir = path.clone();
        }
        if let Some(path) = &global.tmp_dir {
            hub.tmp_dir = path.clone();
        }
        if let Some(path) = &global.store_dir {
            hub.store_dir = path.clone();
        }
        if let Some(offline) = global.offline {
            hub.offline = offline;
        }
        if let Some(endpoint) = &global.hf_endpoint {
            hub.hf_endpoint = clean(Some(endpoint)).map(str::to_owned);
        }
        if let Some(endpoint) = &global.modelscope_endpoint {
            hub.modelscope_endpoint = clean(Some(endpoint)).map(str::to_owned);
        }
        if let Some(proxy) = &global.proxy {
            hub.proxy = Some(proxy.trim().to_owned());
        }
        if let Some(no_proxy) = &global.no_proxy {
            hub.no_proxy = clean(Some(no_proxy)).map(str::to_owned);
        }
        if let Some(allow) = global.allow_private_network {
            hub.allow_private_network = allow;
        }
        if let Some(seconds) = global.download_timeout_secs {
            hub.download_timeout_secs = seconds;
        }
        if let Some(seconds) = global.connect_timeout_secs {
            hub.connect_timeout_secs = seconds;
        }
        if let Some(retries) = global.download_retries {
            hub.download_retries = retries;
        }

        let engine = EngineSettings {
            device: clean(global.device.as_deref())
                .or_else(|| clean(file.engine.device.as_deref()))
                .unwrap_or(config::DEFAULT_DEVICE)
                .to_owned(),
            dtype: clean(global.dtype.as_deref())
                .or_else(|| clean(file.engine.dtype.as_deref()))
                .unwrap_or(config::DEFAULT_DTYPE)
                .to_owned(),
            tf32: global
                .tf32
                .or(file.engine.tf32)
                .unwrap_or(config::DEFAULT_TF32),
            default_model: Some(
                clean(
                    global
                        .default_model
                        .as_deref()
                        .or(file.engine.default_model.as_deref()),
                )
                .unwrap_or(config::DEFAULT_MODEL)
                .to_owned(),
            ),
            max_loaded: global
                .max_loaded
                .or(file.engine.max_loaded)
                .unwrap_or(config::DEFAULT_MAX_LOADED),
        };
        if engine.max_loaded == 0 {
            anyhow::bail!("[engine].max_loaded must be at least 1");
        }

        let configured_registry = file
            .registry
            .dir
            .as_deref()
            .map(|dir| config::resolve_path(&loaded.base, dir));
        let registry_explicit = global.registry_dir.is_some() || configured_registry.is_some();
        let registry = RegistrySettings {
            dir: global
                .registry_dir
                .clone()
                .or(configured_registry)
                .unwrap_or_else(olaya::default_registry_dir),
            store: global
                .registry_store
                .or(file.registry.store)
                .unwrap_or(config::DEFAULT_REGISTRY_STORE),
            url: clean(
                global
                    .registry_url
                    .as_deref()
                    .or(file.registry.url.as_deref()),
            )
            .unwrap_or(olaya::DEFAULT_REGISTRY_URL)
            .to_owned(),
            timeout_secs: global
                .registry_timeout_secs
                .or(file.registry.timeout_secs)
                .unwrap_or(config::DEFAULT_REGISTRY_TIMEOUT_SECS),
            allow_remote: !registry_explicit,
        };
        if registry.timeout_secs == 0 || registry.timeout_secs > 86_400 {
            anyhow::bail!("[registry].timeout_secs must be between 1 and 86400");
        }
        hub.validate_layout()?;
        validate_disposable_layout(&hub, &registry, loaded.path.as_deref())?;

        Ok(Self {
            config_path: loaded.path.clone(),
            engine,
            hub,
            registry,
        })
    }

    fn engine_builder(&self) -> Result<olaya::EngineBuilder> {
        let device = olaya::olaya_candle::device::resolve(&self.engine.device)
            .with_context(|| format!("resolving device {:?}", self.engine.device))?;
        let mut builder = Engine::builder()
            .builtin_families(BuiltinFamilies::all())
            .device(device)
            .dtype(parse_dtype(&self.engine.dtype)?)
            .tf32(self.engine.tf32)
            .max_loaded(self.engine.max_loaded);
        if let Some(model) = &self.engine.default_model {
            builder = builder.default_model(model.clone());
        }
        Ok(builder)
    }
}

fn clean(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn validate_disposable_layout(
    hub: &HubConfig,
    registry: &RegistrySettings,
    config_path: Option<&Path>,
) -> Result<()> {
    let cache = config::normalized_absolute(&hub.cache_dir)?;
    let tmp = config::normalized_absolute(&hub.tmp_dir)?;
    let registry_dir = config::normalized_absolute(&registry.dir)?;
    let models = config::normalized_absolute(&hub.models_dir)?;
    let store = config::normalized_absolute(&hub.store_dir)?;
    for (label, path) in [("cache_dir", &cache), ("tmp_dir", &tmp)] {
        if overlaps(path, &registry_dir) {
            anyhow::bail!(
                "[registry].dir ({}) must not overlap [hub].{label} ({}); \
                 deleting cache or staging would remove catalog files",
                registry_dir.display(),
                path.display()
            );
        }
    }
    if let Some(config_path) = config_path {
        let config_path = config::normalized_absolute(config_path)?;
        for (label, path) in [
            ("hub.cache_dir", &cache),
            ("hub.tmp_dir", &tmp),
            ("hub.models_dir", &models),
            ("hub.store_dir", &store),
            ("registry.dir", &registry_dir),
        ] {
            if path_contains(path, &config_path) {
                anyhow::bail!(
                    "the configuration file ({}) must not live under [{label}] ({}); \
                     deleting that managed directory would remove the configuration",
                    config_path.display(),
                    path.display()
                );
            }
        }
    }
    Ok(())
}

fn overlaps(left: &Path, right: &Path) -> bool {
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

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let loaded = LoadedConfig::load(cli.global.config.as_deref())?;
    let settings = Settings::load(&loaded, &cli.global)?;
    match cli.command {
        #[cfg(feature = "serve")]
        Command::Serve(args) => {
            // The registry/catalog and model store use blocking clients; build
            // the engine before entering the Tokio runtime.
            let engine = engine_from(&settings)?;
            let runtime = tokio::runtime::Runtime::new().context("creating the Tokio runtime")?;
            runtime.block_on(serve(args, &loaded, engine))
        }
        Command::Ask(args) => ask(args, &settings),
        Command::Models(args) => models(args, &settings),
        Command::Doctor(args) => doctor(args, &loaded, &settings),
    }
}

fn load_remote_catalog(
    settings: &Settings,
    refresh: bool,
) -> Result<(Vec<RepoManifest>, &'static str)> {
    if settings.hub.offline {
        anyhow::bail!(
            "offline mode is enabled; provide a local registry directory or disable \
             [hub].offline / OLAYA_OFFLINE"
        );
    }
    let cache = settings.hub.cache_dir.join(olaya::CATALOG_CACHE_FILE);
    let registry = olaya::RemoteRegistry::with_timeout(
        &settings.registry.url,
        Some(cache),
        std::time::Duration::from_secs(settings.registry.timeout_secs),
    )?;
    let url = olaya::hub::redact_url(registry.url());
    let (manifests, source) = registry.load(refresh).with_context(|| {
        format!(
            "loading the remote catalog from {url}; set [registry].url or OLAYA_REGISTRY_URL, \
             pass --registry-dir, or use --registry-store"
        )
    })?;
    tracing::info!(
        url = %url,
        source = source.as_str(),
        models = manifests.len(),
        "remote catalog loaded"
    );
    Ok((manifests, source.as_str()))
}

/// Load the filesystem registry, falling back to the remote catalog when the
/// default directory is missing or empty and the user did not ask for a
/// specific registry.
///
/// For the implicit `$OLAYA_HOME/registry` directory, the remote catalog is
/// authoritative, but locally added manifests are merged in so
/// `olaya models add` remains visible without `--registry-dir`. Local entries
/// that collide with a remote model are ignored (remote wins).
fn registry_manifests(
    settings: &Settings,
    refresh: bool,
) -> Result<(Vec<RepoManifest>, &'static str)> {
    let dir = &settings.registry.dir;
    let remote_first =
        settings.registry.allow_remote && *dir == olaya::hub::default_root().join("registry");
    if remote_first {
        if settings.hub.offline {
            initialize_registry_seed(dir);
            let manifests = olaya::load_manifests(dir).with_context(|| {
                format!(
                    "offline mode is enabled and the local registry {} is unavailable",
                    dir.display()
                )
            })?;
            return Ok((manifests, "dir-offline"));
        }
        match load_remote_catalog(settings, refresh) {
            Ok((manifests, source)) => {
                // Only writes when the directory is still empty, so a local
                // manifest added before the first remote fetch is preserved.
                initialize_registry_fallback(dir, &manifests);
                let (merged, local_added) = merge_local_catalog(dir, manifests);
                return Ok((merged, if local_added { "network+dir" } else { source }));
            }
            Err(remote_error) => {
                initialize_registry_seed(dir);
                match olaya::load_manifests(dir) {
                    Ok(manifests) => {
                        tracing::warn!(
                            dir = %dir.display(),
                            error = %remote_error,
                            "remote catalog unavailable; using the local registry fallback"
                        );
                        return Ok((manifests, "dir-fallback"));
                    }
                    Err(_) => return Err(remote_error),
                }
            }
        }
    }
    match olaya::load_manifests(dir) {
        Ok(manifests) => Ok((manifests, "dir")),
        Err(error)
            if settings.registry.allow_remote
                && !settings.hub.offline
                && registry_is_empty(dir) =>
        {
            tracing::info!(
                dir = %dir.display(),
                %error,
                "registry directory is unavailable; using the remote catalog"
            );
            load_remote_catalog(settings, refresh)
        }
        Err(error) => Err(error).with_context(|| {
            if settings.hub.offline {
                format!(
                    "loading registry {} in offline mode; pass --registry-dir or \
                     disable [hub].offline / OLAYA_OFFLINE",
                    dir.display()
                )
            } else {
                format!("loading registry {}", dir.display())
            }
        }),
    }
}

/// Seed the implicit registry directory with the manifests compiled into the
/// CLI. This gives a fresh offline install a working catalog before the first
/// remote fetch.
fn initialize_registry_seed(dir: &std::path::Path) {
    if !registry_is_empty(dir) {
        return;
    }
    let mut manifests = Vec::with_capacity(registry_seed::MANIFESTS.len());
    for (path, text) in registry_seed::MANIFESTS {
        match serde_json::from_str::<RepoManifest>(text) {
            Ok(manifest) => manifests.push(manifest),
            Err(error) => {
                tracing::warn!(path, %error, "embedded registry manifest is invalid");
                return;
            }
        }
    }
    match olaya::export_registry(&manifests, dir, true) {
        Ok(report) => tracing::info!(
            dir = %dir.display(),
            written = report.written,
            "seeded the local registry fallback"
        ),
        Err(error) => tracing::warn!(
            dir = %dir.display(),
            %error,
            "could not seed the local registry fallback"
        ),
    }
}

/// Write the loaded catalog into the implicit registry directory when it is
/// still empty. Later runs can use it when the network and catalog cache fail.
fn initialize_registry_fallback(dir: &std::path::Path, manifests: &[RepoManifest]) {
    if !registry_is_empty(dir) {
        return;
    }
    match olaya::export_registry(manifests, dir, true) {
        Ok(report) => tracing::info!(
            dir = %dir.display(),
            written = report.written,
            unchanged = report.unchanged,
            "initialized local registry fallback"
        ),
        Err(error) => tracing::warn!(
            dir = %dir.display(),
            %error,
            "could not initialize the local registry fallback"
        ),
    }
}

/// Merge locally added manifests into a remote catalog. Remote entries win on
/// name or alias conflicts; local-only models are appended.
fn merge_local_catalog(dir: &Path, remote: Vec<RepoManifest>) -> (Vec<RepoManifest>, bool) {
    let local = match olaya::load_manifests(dir) {
        Ok(manifests) => manifests,
        Err(_) => return (remote, false),
    };
    let mut catalog = olaya::Catalog::new();
    for manifest in remote {
        // The remote bundle was validated already.
        let _ = catalog.insert(manifest);
    }
    let mut local_added = false;
    for manifest in local {
        match catalog.insert(manifest) {
            Ok(()) => local_added = true,
            Err(error) => tracing::debug!(
                dir = %dir.display(),
                %error,
                "local registry entry conflicts with the remote catalog; remote wins"
            ),
        }
    }
    let mut manifests = catalog.manifests();
    manifests.sort_by(|left, right| left.name.cmp(&right.name));
    (manifests, local_added)
}

fn warn_remote_first_registry(settings: &Settings) {
    if settings.registry.allow_remote
        && settings.registry.dir == olaya::hub::default_root().join("registry")
    {
        tracing::warn!(
            dir = %settings.registry.dir.display(),
            "the implicit registry merges local-only manifests with the remote catalog; \
             pass --registry-dir to override a remote model or test the local catalog alone"
        );
    }
}

fn registry_is_empty(dir: &std::path::Path) -> bool {
    if !dir.exists() {
        return true;
    }
    if !dir.is_dir() {
        return false;
    }
    std::fs::read_dir(dir)
        .map(|mut entries| entries.next().is_none())
        .unwrap_or(false)
}

fn with_registry(
    builder: olaya::EngineBuilder,
    settings: &Settings,
) -> Result<olaya::EngineBuilder> {
    let builder = builder.hub(settings.hub.clone());
    if settings.registry.store {
        let store = olaya::Hub::new(settings.hub.clone()).store();
        Ok(builder.registry(Arc::new(olaya::StoreRegistry::new(store)))?)
    } else {
        let (manifests, source) = registry_manifests(settings, false)?;
        tracing::debug!(source, models = manifests.len(), "registry loaded");
        Ok(builder.manifests(manifests))
    }
}

#[cfg(feature = "serve")]
fn engine_from(settings: &Settings) -> Result<Engine> {
    with_registry(settings.engine_builder()?, settings)?
        .build()
        .context("building the engine")
}

#[cfg(feature = "serve")]
fn serve_config(args: &ServeArgs, loaded: &LoadedConfig) -> Result<olaya::serve::ServeConfig> {
    let defaults = olaya::serve::ServeConfig::default();
    let file = &loaded.file.serve;
    let config = olaya::serve::ServeConfig {
        host: clean(args.host.as_deref())
            .or_else(|| clean(file.host.as_deref()))
            .unwrap_or(&defaults.host)
            .to_owned(),
        port: args.port.or(file.port).unwrap_or(defaults.port),
        api_key: clean(args.api_key.as_deref())
            .or_else(|| clean(file.api_key.as_deref()))
            .map(str::to_owned),
        allow_unauthenticated: args
            .allow_unauthenticated
            .or(file.allow_unauthenticated)
            .unwrap_or(defaults.allow_unauthenticated),
        max_body_bytes: args
            .max_body_bytes
            .or(file.max_body_bytes)
            .unwrap_or(defaults.max_body_bytes),
        max_queue: args
            .max_queue
            .or(file.max_queue)
            .unwrap_or(defaults.max_queue),
        max_batch: args
            .max_batch
            .or(file.max_batch)
            .unwrap_or(defaults.max_batch),
        max_wait: args
            .batch_wait_ms
            .or(file.batch_wait_ms)
            .map(std::time::Duration::from_millis)
            .unwrap_or(defaults.max_wait),
        request_timeout: std::time::Duration::from_secs(
            args.request_timeout_secs
                .or(file.request_timeout_secs)
                .unwrap_or(config::DEFAULT_REQUEST_TIMEOUT_SECS),
        ),
        max_operations: args
            .max_operations
            .or(file.max_operations)
            .unwrap_or(config::DEFAULT_MAX_OPERATIONS),
        cors_origins: if args.cors_origins.is_empty() {
            normalize_origins(file.cors_origins.clone().unwrap_or_default())
        } else {
            normalize_origins(args.cors_origins.clone())
        },
        allowed_hosts: if args.allowed_hosts.is_empty() {
            normalize_origins(file.allowed_hosts.clone().unwrap_or_default())
        } else {
            normalize_origins(args.allowed_hosts.clone())
        },
    };
    config.validate().map_err(anyhow::Error::msg)?;
    Ok(config)
}

#[cfg(feature = "serve")]
fn normalize_origins(values: impl IntoIterator<Item = String>) -> Vec<String> {
    values
        .into_iter()
        .flat_map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .collect()
}

#[cfg(feature = "serve")]
async fn serve(args: ServeArgs, loaded: &LoadedConfig, engine: Engine) -> Result<()> {
    let runtime: Arc<dyn DecisionRuntime> = Arc::new(engine);
    let config = serve_config(&args, loaded)?;
    let host = config.host.clone();
    let port = config.port;
    olaya::serve::serve(runtime, config).await.with_context(|| {
        format!(
            "serving on {host}:{port}; if the port is already in use, set [serve].port in the \
             config file or pass --port"
        )
    })
}

const MAX_STATE_INPUT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_QUESTIONS_FILE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_MANIFEST_FILE_BYTES: u64 = 8 * 1024 * 1024;

fn read_limited_text(reader: impl Read, limit: u64, label: &str) -> Result<String> {
    let mut bytes = Vec::new();
    reader
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading {label}"))?;
    if bytes.len() as u64 > limit {
        anyhow::bail!("{label} is larger than the {limit}-byte limit");
    }
    String::from_utf8(bytes).with_context(|| format!("{label} is not valid UTF-8"))
}

fn read_limited_file(path: &Path, limit: u64) -> Result<String> {
    let display = path.display().to_string();
    let file = std::fs::File::open(path).with_context(|| format!("opening {display}"))?;
    read_limited_text(file, limit, &display)
}

fn ask(args: AskArgs, settings: &Settings) -> Result<()> {
    let raw_state = if args.state == "-" {
        read_limited_text(
            std::io::stdin().lock(),
            MAX_STATE_INPUT_BYTES,
            "state from stdin",
        )?
    } else {
        args.state
    };
    let state_value: serde_json::Value = match serde_json::from_str(&raw_state) {
        Ok(value) => value,
        Err(_) => serde_json::Value::String(raw_state.clone()),
    };
    let state = State::from_value(state_value).map_err(|error| anyhow::anyhow!("{error}"))?;

    let raw_questions = read_limited_file(&args.questions, MAX_QUESTIONS_FILE_BYTES)?;
    let questions: indexmap::IndexMap<String, Question> =
        serde_json::from_str(&raw_questions).context("parsing questions")?;
    let questions = Questions::from(questions);

    let model = args
        .model
        .or_else(|| settings.engine.default_model.clone())
        .context("model is required; pass --model or set [engine].default_model")?;
    let engine = with_registry(settings.engine_builder()?, settings)?
        .build()
        .context("building the engine")?;
    let answers = engine.ask_typed(&model, &state, &questions)?;
    println!(
        "{}",
        serde_json::to_string_pretty(answers.as_map()).context("serializing answers")?
    );
    Ok(())
}

fn catalog_engine(settings: &Settings) -> Result<Engine> {
    let builder = Engine::builder().builtin_families(BuiltinFamilies::all());
    with_registry(builder, settings)?
        .build()
        .context("building the engine")
}

fn default_store(settings: &Settings) -> Arc<dyn olaya::ManifestStore> {
    olaya::Hub::new(settings.hub.clone()).store()
}

/// Delete a model's materialized files, content-store entry, and filesystem
/// registry manifests, then garbage-collect blobs no remaining model uses.
///
/// The remote catalog is never modified; when it is authoritative the model
/// stays listed and a later pull downloads it again.
fn purge_model(settings: &Settings, reference: &str) -> Result<serde_json::Value> {
    let hub = &settings.hub;
    let store = olaya::Store::open(&hub.store_dir);
    let stored = store.manifest(reference).ok();

    let canonical = match &stored {
        Some(stored) => stored.config.name.clone(),
        None => {
            // No network here: fall back to manifests already on disk.
            let manifests = olaya::load_manifests(&settings.registry.dir).unwrap_or_default();
            manifests
                .iter()
                .find(|manifest| {
                    manifest.name == reference
                        || manifest.aliases.iter().any(|alias| alias == reference)
                })
                .map(|manifest| manifest.name.clone())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "model {reference:?} was not found in {} or {}; nothing to purge",
                        hub.store_dir.display(),
                        settings.registry.dir.display()
                    )
                })?
        }
    };
    if !is_single_path_component(&canonical) {
        anyhow::bail!(
            "refusing to purge model with unsafe name {canonical:?}; the store entry is corrupt"
        );
    }

    let model_dir = hub.models_dir.join(&canonical);
    let (removed_dir, model_files, model_bytes) = remove_model_dir(&model_dir)
        .with_context(|| format!("purging model directory {}", model_dir.display()))?;

    let store_report = match &stored {
        Some(_) => Some(
            store
                .remove(&canonical)
                .with_context(|| format!("removing {canonical} from the content store"))?,
        ),
        None => None,
    };

    let registry_report = if settings.registry.store {
        None
    } else {
        Some(olaya::remove_registry_manifest(
            &settings.registry.dir,
            &canonical,
        )?)
    };
    let registry_removed = registry_report.as_ref().map_or(0, |report| report.removed);
    if store_report.is_none() && !removed_dir && registry_removed == 0 {
        anyhow::bail!(
            "model {canonical:?} has no model directory, store entry, or registry manifest; \
             nothing to purge"
        );
    }

    if settings.registry.allow_remote
        && !settings.registry.store
        && settings.registry.dir == olaya::hub::default_root().join("registry")
    {
        tracing::warn!(
            model = %canonical,
            "the remote catalog still lists this model; purge removed only local files"
        );
    }

    let store_bytes = store_report.as_ref().map_or(0, |report| report.freed_bytes);
    Ok(json!({
        "model": canonical,
        "materialized": {
            "path": model_dir,
            "removed": removed_dir,
            "files": model_files,
            "bytes": model_bytes,
        },
        "store": store_report.as_ref().map(|report| json!({
            "manifests": report.manifests,
            "blobs": report.blobs,
            "bytes": report.freed_bytes,
            "gc_skipped": report.gc_skipped,
        })),
        "registry": registry_report.as_ref().map(|report| json!({
            "dir": report.dir,
            "removed": report.removed,
        })),
        "reclaimed_bytes": model_bytes + store_bytes,
    }))
}

/// Whether `name` is exactly one safe path component, suitable for
/// `models_dir/<name>`.
fn is_single_path_component(name: &str) -> bool {
    let mut components = Path::new(name).components();
    matches!(components.next(), Some(std::path::Component::Normal(_)))
        && components.next().is_none()
        && olaya::is_safe_relative_path(name)
}

/// Remove a real directory without following symlinks.
fn remove_model_dir(dir: &Path) -> Result<(bool, usize, u64)> {
    let metadata = match std::fs::symlink_metadata(dir) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok((false, 0, 0)),
        Err(error) => return Err(error).with_context(|| format!("reading {}", dir.display())),
    };
    if !metadata.is_dir() {
        anyhow::bail!(
            "refusing to purge {}: it is a symlink or not a directory",
            dir.display()
        );
    }
    let (files, bytes) = directory_usage(dir)?;
    std::fs::remove_dir_all(dir).with_context(|| format!("removing {}", dir.display()))?;
    Ok((true, files, bytes))
}

fn directory_usage(dir: &Path) -> Result<(usize, u64)> {
    fn walk(dir: &Path, files: &mut usize, bytes: &mut u64) -> Result<()> {
        for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
            let entry = entry.with_context(|| format!("reading {}", dir.display()))?;
            let path = entry.path();
            let file_type = entry
                .file_type()
                .with_context(|| format!("reading {}", path.display()))?;
            if file_type.is_dir() {
                walk(&path, files, bytes)?;
            } else if file_type.is_file() {
                *files += 1;
                *bytes += entry
                    .metadata()
                    .with_context(|| format!("reading {}", path.display()))?
                    .len();
            }
            // Symlinked entries are not followed and are not counted.
        }
        Ok(())
    }
    let mut files = 0usize;
    let mut bytes = 0u64;
    walk(dir, &mut files, &mut bytes)?;
    Ok((files, bytes))
}

fn models(args: ModelsArgs, settings: &Settings) -> Result<()> {
    let ModelsArgs { command } = args;
    match command {
        ModelsCommand::Stored => {
            let store = default_store(settings);
            println!(
                "{}",
                serde_json::to_string_pretty(&store.list()?)
                    .context("serializing stored models")?
            );
        }
        ModelsCommand::Manifest { model } => {
            let store = default_store(settings);
            let stored = store.get(&model)?;
            let report = json!({
                "model": model,
                "digest": stored.digest,
                "manifest_path": stored.manifest_path.display().to_string(),
                "manifest": stored.manifest,
                "config": stored.config,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&report).context("serializing stored manifest")?
            );
        }
        ModelsCommand::Verify { model } => {
            let store = default_store(settings);
            let stored = store.get(&model)?;
            store.verify(&stored)?;
            println!(
                "{}",
                json!({ "model": model, "digest": stored.digest, "verified": true })
            );
        }
        ModelsCommand::List { all } => {
            let engine = catalog_engine(settings)?;
            let models = if all {
                serde_json::to_value(engine.list_models()).context("serializing models")?
            } else {
                let models = engine
                    .manifests()
                    .into_iter()
                    .map(|manifest| {
                        json!({
                            "name": manifest.name,
                            "aliases": manifest.aliases,
                            "description": manifest.description,
                            "release_date": manifest.release_date.unwrap_or_default(),
                        })
                    })
                    .collect::<Vec<_>>();
                serde_json::to_value(models).context("serializing models")?
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&models).context("serializing models")?
            );
        }
        ModelsCommand::Show { model } => {
            let engine = catalog_engine(settings)?;
            let manifest = engine
                .resolve(&model)
                .ok_or_else(|| anyhow::anyhow!("unknown model: {model}"))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&manifest).context("serializing manifest")?
            );
        }
        ModelsCommand::Pull { model } => {
            let engine = catalog_engine(settings)?;
            let manifest = engine.pull(&model)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&manifest).context("serializing manifest")?
            );
        }
        ModelsCommand::Add { manifest, force } => {
            if settings.registry.store {
                anyhow::bail!(
                    "models add writes a manifest to a filesystem registry; disable \
                     [registry].store or pass --registry-dir"
                );
            }
            let raw = read_limited_file(&manifest, MAX_MANIFEST_FILE_BYTES)?;
            let manifest: RepoManifest = serde_json::from_str(&raw).context("parsing manifest")?;
            let report = olaya::export_registry(&[manifest], &settings.registry.dir, force)?;
            warn_remote_first_registry(settings);
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "dir": report.dir,
                    "written": report.written,
                    "unchanged": report.unchanged,
                }))
                .context("serializing add report")?
            );
        }
        ModelsCommand::Remove { model, purge } => {
            if !purge {
                if settings.registry.store {
                    anyhow::bail!(
                        "models remove edits a filesystem registry; disable [registry].store, \
                         pass --registry-dir, or use --purge to delete store entries and model \
                         files"
                    );
                }
                let report = olaya::remove_registry_manifest(&settings.registry.dir, &model)?;
                warn_remote_first_registry(settings);
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "dir": report.dir,
                        "model": report.model,
                        "removed": report.removed,
                    }))
                    .context("serializing remove report")?
                );
            } else {
                let report = purge_model(settings, &model)?;
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report).context("serializing purge report")?
                );
            }
        }
        ModelsCommand::Export { dir, force } => {
            let dir = dir.unwrap_or_else(|| settings.registry.dir.clone());
            let (manifests, source) = if settings.registry.store {
                let store = default_store(settings);
                let registry = olaya::StoreRegistry::new(store);
                (olaya::ManifestRegistry::manifests(&registry)?, "store")
            } else {
                registry_manifests(settings, true)?
            };
            let report = olaya::export_registry(&manifests, &dir, force)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "dir": report.dir,
                    "written": report.written,
                    "unchanged": report.unchanged,
                    "source": source,
                    "models": manifests.len(),
                }))
                .context("serializing export report")?
            );
        }
    }
    Ok(())
}

fn doctor(_args: DoctorArgs, loaded: &LoadedConfig, settings: &Settings) -> Result<()> {
    let hub = &settings.hub;
    let (registry_models, registry_source, registry_error) = if settings.registry.store {
        let store = olaya::Hub::new(hub.clone()).store();
        let registry = olaya::StoreRegistry::new(store);
        match olaya::ManifestRegistry::manifests(&registry) {
            Ok(manifests) => (manifests.len(), "store", None),
            Err(error) => (0, "store", Some(error.to_string())),
        }
    } else {
        match registry_manifests(settings, false) {
            Ok((manifests, source)) => (manifests.len(), source, None),
            Err(error) => (0, "dir", Some(error.to_string())),
        }
    };
    let stored_models = olaya::Store::open(&hub.store_dir)
        .list()
        .unwrap_or_default();
    let device = olaya::olaya_candle::device::resolve(&settings.engine.device)
        .map(|device| format!("{device:?}"))
        .unwrap_or_else(|error| format!("error: {error}"));
    #[allow(unused_mut)]
    let mut report = json!({
        "config_file": settings.config_path.as_ref().map(|path| path.display().to_string()),
        "config_base": loaded.base.display().to_string(),
        "device_preference": settings.engine.device,
        "device": device,
        "compiled_accelerators": olaya::olaya_candle::device::compiled_accelerators(),
        "dtype": settings.engine.dtype,
        "tf32": settings.engine.tf32,
        "default_model": settings.engine.default_model,
        "max_loaded": settings.engine.max_loaded,
        "registry_dir": settings.registry.dir.display().to_string(),
        "registry_store": settings.registry.store,
        "registry_allow_remote": settings.registry.allow_remote,
        "registry_source": registry_source,
        "registry_models": registry_models,
        "registry_error": registry_error,
        "registry_url": olaya::hub::redact_url(&settings.registry.url),
        "registry_timeout_secs": settings.registry.timeout_secs,
        "registry_cache": hub
            .cache_dir
            .join(olaya::CATALOG_CACHE_FILE)
            .display()
            .to_string(),
        "models_dir": hub.models_dir,
        "cache_dir": hub.cache_dir,
        "tmp_dir": hub.tmp_dir,
        "store_dir": hub.store_dir,
        "models_dir_exists": hub.models_dir.exists(),
        "cache_dir_exists": hub.cache_dir.exists(),
        "tmp_dir_exists": hub.tmp_dir.exists(),
        "store_dir_exists": hub.store_dir.exists(),
        "stored_models": stored_models,
        "offline": hub.offline,
        "hf_endpoint": hub.hf_endpoint.as_deref().map(olaya::hub::redact_url),
        "modelscope_endpoint": hub.modelscope_endpoint.as_deref().map(olaya::hub::redact_url),
        "proxy": hub.proxy.as_deref().map(olaya::hub::redact_url),
        "no_proxy": hub.no_proxy,
        "allow_private_network": hub.allow_private_network,
        "download_timeout_secs": hub.download_timeout_secs,
        "connect_timeout_secs": hub.connect_timeout_secs,
        "download_retries": hub.download_retries,
    });
    #[cfg(feature = "serve")]
    {
        let serve_args = match Cli::try_parse_from(["olaya", "serve"])?.command {
            Command::Serve(args) => args,
            _ => unreachable!("the command was parsed as serve"),
        };
        let serve = serve_config(&serve_args, loaded)?;
        report["serve"] = json!({
            "host": serve.host,
            "port": serve.port,
            "api_key_set": serve.api_key.is_some(),
            "allow_unauthenticated": serve.allow_unauthenticated,
            "max_batch": serve.max_batch,
            "batch_wait_ms": serve.max_wait.as_millis() as u64,
            "max_body_bytes": serve.max_body_bytes,
            "max_queue": serve.max_queue,
            "request_timeout_secs": serve.request_timeout.as_secs(),
            "max_operations": serve.max_operations,
            "cors_origins": serve.cors_origins,
            "allowed_hosts": serve.allowed_hosts,
        });
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn parse_dtype(value: &str) -> Result<olaya::candle_core::DType> {
    match value {
        "f32" => Ok(olaya::candle_core::DType::F32),
        "f16" => Ok(olaya::candle_core::DType::F16),
        "bf16" => Ok(olaya::candle_core::DType::BF16),
        other => anyhow::bail!("unknown dtype {other:?}; expected f32, f16, or bf16"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_global() -> GlobalArgs {
        GlobalArgs {
            config: None,
            registry_dir: None,
            registry_store: None,
            registry_url: None,
            registry_timeout_secs: None,
            models_dir: None,
            cache_dir: None,
            tmp_dir: None,
            store_dir: None,
            hf_endpoint: None,
            modelscope_endpoint: None,
            proxy: None,
            no_proxy: None,
            allow_private_network: None,
            download_timeout_secs: None,
            connect_timeout_secs: None,
            download_retries: None,
            offline: None,
            device: None,
            dtype: None,
            tf32: None,
            default_model: None,
            max_loaded: None,
        }
    }

    #[test]
    fn settings_resolve_cli_over_file_over_default() {
        let loaded = LoadedConfig {
            file: config::FileConfig {
                engine: config::EngineConfig {
                    device: Some("cpu".to_owned()),
                    dtype: Some("f16".to_owned()),
                    tf32: Some(true),
                    default_model: Some("file-model".to_owned()),
                    max_loaded: Some(2),
                },
                registry: config::RegistryConfig {
                    store: Some(true),
                    url: Some("https://file.example/catalog.json".to_owned()),
                    timeout_secs: Some(5),
                    ..Default::default()
                },
                hub: config::HubConfigFile {
                    no_proxy: Some("file.example".to_owned()),
                    allow_private_network: Some(false),
                    download_timeout_secs: Some(10),
                    connect_timeout_secs: Some(5),
                    download_retries: Some(2),
                    ..Default::default()
                },
                ..Default::default()
            },
            path: None,
            base: PathBuf::new(),
        };
        let mut global = empty_global();
        global.device = Some("cuda".to_owned());
        global.tf32 = Some(false);
        global.max_loaded = Some(4);
        global.registry_store = Some(false);
        global.registry_url = Some("https://cli.example/catalog.json".to_owned());
        global.registry_timeout_secs = Some(9);
        global.no_proxy = Some("cli.example".to_owned());
        global.allow_private_network = Some(true);
        global.download_timeout_secs = Some(20);
        global.connect_timeout_secs = Some(7);
        global.download_retries = Some(3);

        let settings = Settings::load(&loaded, &global).expect("settings resolve");
        assert_eq!(settings.engine.device, "cuda");
        assert_eq!(settings.engine.dtype, "f16");
        assert!(!settings.engine.tf32);
        assert_eq!(settings.engine.default_model.as_deref(), Some("file-model"));
        assert_eq!(settings.engine.max_loaded, 4);
        assert!(!settings.registry.store);
        assert_eq!(settings.registry.url, "https://cli.example/catalog.json");
        assert_eq!(settings.registry.timeout_secs, 9);
        assert_eq!(settings.hub.no_proxy.as_deref(), Some("cli.example"));
        assert!(settings.hub.allow_private_network);
        assert_eq!(settings.hub.download_timeout_secs, 20);
        assert_eq!(settings.hub.connect_timeout_secs, 7);
        assert_eq!(settings.hub.download_retries, 3);
    }

    #[cfg(windows)]
    #[test]
    fn overlap_checks_ignore_windows_path_case() {
        assert!(overlaps(
            Path::new(r"C:\Data\models"),
            Path::new(r"c:\data\models\cache")
        ));
        assert!(!overlaps(
            Path::new(r"C:\Data\models"),
            Path::new(r"C:\Data\model-cache")
        ));
    }

    #[test]
    fn rejects_disposable_directories_inside_models_or_store() {
        let loaded = LoadedConfig {
            file: config::FileConfig {
                hub: config::HubConfigFile {
                    models_dir: Some(PathBuf::from("data")),
                    cache_dir: Some(PathBuf::from("data/cache")),
                    ..Default::default()
                },
                ..Default::default()
            },
            path: None,
            base: PathBuf::new(),
        };
        let error = Settings::load(&loaded, &empty_global()).expect_err("overlap is rejected");
        assert!(error.to_string().contains("must not overlap"));
    }

    #[test]
    fn rejects_a_config_file_inside_cache_or_tmp() {
        let loaded = LoadedConfig {
            file: config::FileConfig {
                hub: config::HubConfigFile {
                    cache_dir: Some(PathBuf::from("config-cache")),
                    ..Default::default()
                },
                ..Default::default()
            },
            path: Some(PathBuf::from("config-cache/olaya.toml")),
            base: PathBuf::new(),
        };
        let error =
            Settings::load(&loaded, &empty_global()).expect_err("config overlap is rejected");
        assert!(error.to_string().contains("configuration file"));
    }

    #[test]
    fn purge_path_guard_rejects_traversal_and_multiple_components() {
        assert!(is_single_path_component("laya"));
        assert!(is_single_path_component("laya-multilingual"));
        assert!(!is_single_path_component(""));
        assert!(!is_single_path_component("."));
        assert!(!is_single_path_component(".."));
        assert!(!is_single_path_component("../escape"));
        assert!(!is_single_path_component("a/b"));
        assert!(!is_single_path_component("a\\b"));
        assert!(!is_single_path_component("C:foo"));
        assert!(!is_single_path_component("name."));
    }

    #[test]
    fn tri_state_flags_accept_explicit_false() {
        let cli = Cli::try_parse_from([
            "olaya",
            "--tf32=false",
            "--offline",
            "--registry-store=false",
            "doctor",
        ])
        .expect("cli parses");
        assert_eq!(cli.global.tf32, Some(false));
        assert_eq!(cli.global.offline, Some(true));
        assert_eq!(cli.global.registry_store, Some(false));
    }

    #[test]
    fn offline_mode_uses_the_local_registry_without_network() {
        let dir = std::env::temp_dir().join(format!(
            "olaya-cli-offline-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or_default()
        ));
        initialize_registry_seed(&dir);
        let settings = Settings {
            config_path: None,
            engine: EngineSettings {
                device: config::DEFAULT_DEVICE.to_owned(),
                dtype: config::DEFAULT_DTYPE.to_owned(),
                tf32: config::DEFAULT_TF32,
                default_model: Some(config::DEFAULT_MODEL.to_owned()),
                max_loaded: config::DEFAULT_MAX_LOADED,
            },
            hub: HubConfig {
                offline: true,
                cache_dir: dir.join("cache"),
                ..HubConfig::default()
            },
            registry: RegistrySettings {
                dir: dir.clone(),
                store: false,
                url: "http://127.0.0.1:9/registry-catalog.json".to_owned(),
                timeout_secs: 1,
                allow_remote: true,
            },
        };
        let (manifests, source) =
            registry_manifests(&settings, false).expect("offline local registry loads");
        assert_eq!(source, "dir");
        assert_eq!(manifests.len(), registry_seed::MANIFESTS.len());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(feature = "serve")]
    #[test]
    fn cors_origins_are_normalized_and_cli_overrides_the_file() {
        let loaded = LoadedConfig {
            file: config::FileConfig {
                serve: config::ServeConfigFile {
                    cors_origins: Some(vec!["https://file.example".to_owned()]),
                    ..Default::default()
                },
                ..Default::default()
            },
            path: None,
            base: PathBuf::new(),
        };
        let args = ServeArgs {
            cors_origins: vec![" https://a.example , https://b.example ".to_owned()],
            ..ServeArgs::default()
        };
        let config = serve_config(&args, &loaded).expect("serve config");
        assert_eq!(
            config.cors_origins,
            vec![
                "https://a.example".to_owned(),
                "https://b.example".to_owned()
            ]
        );

        let config = serve_config(&ServeArgs::default(), &loaded).expect("file origins");
        assert_eq!(config.cors_origins, vec!["https://file.example".to_owned()]);

        let cleared = ServeArgs {
            cors_origins: vec![String::new()],
            ..ServeArgs::default()
        };
        let config = serve_config(&cleared, &loaded).expect("cleared origins");
        assert!(config.cors_origins.is_empty());
    }

    #[cfg(feature = "serve")]
    #[test]
    fn serve_flags_override_file_values_including_false() {
        let loaded = LoadedConfig {
            file: config::FileConfig {
                serve: config::ServeConfigFile {
                    port: Some(9000),
                    allow_unauthenticated: Some(true),
                    max_batch: Some(4),
                    ..Default::default()
                },
                ..Default::default()
            },
            path: None,
            base: PathBuf::new(),
        };
        let args = ServeArgs {
            allow_unauthenticated: Some(false),
            max_batch: Some(8),
            ..ServeArgs::default()
        };
        let config = serve_config(&args, &loaded).expect("serve config resolves");
        assert_eq!(config.port, 9000);
        assert!(!config.allow_unauthenticated);
        assert_eq!(config.max_batch, 8);
    }

    #[test]
    fn seeds_a_missing_registry_from_embedded_manifests() {
        let dir = std::env::temp_dir().join(format!(
            "olaya-cli-seed-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or_default()
        ));
        initialize_registry_seed(&dir);
        let manifests = olaya::load_manifests(&dir).expect("embedded seed loads");
        assert_eq!(manifests.len(), registry_seed::MANIFESTS.len());
        assert!(manifests.iter().any(|manifest| manifest.name == "laya"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
