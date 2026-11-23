mod config;
mod registry_seed;

use std::path::PathBuf;
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
    /// Add a manifest to the runtime catalog.
    Add { manifest: PathBuf },
    /// Remove a model from the runtime catalog.
    Remove { model: String },
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

struct Settings {
    config_path: Option<PathBuf>,
    engine: EngineSettings,
    hub: HubConfig,
    registry: RegistrySettings,
}

struct EngineSettings {
    device: String,
    dtype: String,
    tf32: bool,
    default_model: Option<String>,
    max_loaded: usize,
}

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
        if let Some(proxy) = clean(file.hub.proxy.as_deref()) {
            hub.proxy = Some(proxy.to_owned());
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
            hub.proxy = clean(Some(proxy)).map(str::to_owned);
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
        if registry.timeout_secs == 0 {
            anyhow::bail!("[registry].timeout_secs must be at least 1");
        }

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
    value.filter(|value| !value.trim().is_empty())
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
    let (manifests, source) = registry.load(refresh).with_context(|| {
        format!(
            "loading the remote catalog from {}; set [registry].url or OLAYA_REGISTRY_URL, \
             pass --registry-dir, or use --registry-store",
            registry.url()
        )
    })?;
    tracing::info!(
        url = registry.url(),
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
/// For the implicit `$OLAYA_HOME/registry` directory, the remote catalog (or
/// its cache) is authoritative and seeds the local directory on first use, so
/// later offline runs have a fallback.
fn registry_manifests(
    settings: &Settings,
    refresh: bool,
) -> Result<(Vec<RepoManifest>, &'static str)> {
    let dir = &settings.registry.dir;
    let remote_first =
        settings.registry.allow_remote && *dir == olaya::hub::default_root().join("registry");
    if remote_first {
        initialize_registry_seed(dir);
        if settings.hub.offline {
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
                initialize_registry_fallback(dir, &manifests);
                return Ok((manifests, source));
            }
            Err(remote_error) => match olaya::load_manifests(dir) {
                Ok(manifests) => {
                    tracing::warn!(
                        dir = %dir.display(),
                        error = %remote_error,
                        "remote catalog unavailable; using the local registry fallback"
                    );
                    return Ok((manifests, "dir-fallback"));
                }
                Err(_) => return Err(remote_error),
            },
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
    };
    if config.port == 0 {
        anyhow::bail!("[serve].port must be between 1 and 65535");
    }
    if config.max_batch == 0 {
        anyhow::bail!("[serve].max_batch must be at least 1");
    }
    if config.max_queue == 0 {
        anyhow::bail!("[serve].max_queue must be at least 1");
    }
    if config.max_body_bytes == 0 {
        anyhow::bail!("[serve].max_body_bytes must be at least 1");
    }
    Ok(config)
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

fn ask(args: AskArgs, settings: &Settings) -> Result<()> {
    let raw_state = if args.state == "-" {
        std::io::read_to_string(std::io::stdin()).context("reading state from stdin")?
    } else {
        args.state
    };
    let state_value: serde_json::Value = match serde_json::from_str(&raw_state) {
        Ok(value) => value,
        Err(_) => serde_json::Value::String(raw_state.clone()),
    };
    let state = State::from_value(state_value).map_err(|error| anyhow::anyhow!("{error}"))?;

    let raw_questions = std::fs::read_to_string(&args.questions)
        .with_context(|| format!("reading {}", args.questions.display()))?;
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
        ModelsCommand::Add { manifest } => {
            let engine = catalog_engine(settings)?;
            let raw = std::fs::read_to_string(&manifest)
                .with_context(|| format!("reading {}", manifest.display()))?;
            let manifest: RepoManifest = serde_json::from_str(&raw).context("parsing manifest")?;
            engine.add_manifest(manifest)?;
            println!("{}", json!({ "added": true }));
        }
        ModelsCommand::Remove { model } => {
            let engine = catalog_engine(settings)?;
            let removed = engine.delete(&model)?;
            println!("{}", json!({ "model": model, "removed": removed }));
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
        "registry_url": settings.registry.url,
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
        "hf_endpoint": hub.hf_endpoint,
        "modelscope_endpoint": hub.modelscope_endpoint,
        "proxy": hub.proxy,
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

        let settings = Settings::load(&loaded, &global).expect("settings resolve");
        assert_eq!(settings.engine.device, "cuda");
        assert_eq!(settings.engine.dtype, "f16");
        assert!(!settings.engine.tf32);
        assert_eq!(settings.engine.default_model.as_deref(), Some("file-model"));
        assert_eq!(settings.engine.max_loaded, 4);
        assert!(!settings.registry.store);
        assert_eq!(settings.registry.url, "https://cli.example/catalog.json");
        assert_eq!(settings.registry.timeout_secs, 9);
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
