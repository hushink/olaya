mod config;

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
    /// Configuration file (default: ./olaya.toml, ./sys1.toml, or ~/.olaya/config.toml).
    #[arg(long, env = "OLAYA_CONFIG", global = true)]
    config: Option<PathBuf>,
    /// Directory containing `olaya.manifest.json` files.
    #[arg(long, env = "OLAYA_REGISTRY_DIR", global = true)]
    registry_dir: Option<PathBuf>,
    /// Use the content store as the manifest registry (no registry directory).
    #[arg(long, global = true)]
    registry_store: bool,
    #[command(subcommand)]
    command: Command,
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
#[derive(Debug, Args)]
struct ServeArgs {
    #[arg(long)]
    host: Option<String>,
    #[arg(long)]
    port: Option<u16>,
    #[arg(long, env = "OLAYA_API_KEY")]
    api_key: Option<String>,
    #[arg(long)]
    allow_unauthenticated: bool,
    #[arg(long, env = "OLAYA_MODELS_DIR")]
    models_dir: Option<PathBuf>,
    #[arg(long, env = "OLAYA_CACHE_DIR")]
    cache_dir: Option<PathBuf>,
    #[arg(long, env = "OLAYA_TMP_DIR")]
    tmp_dir: Option<PathBuf>,
    #[arg(long, env = "OLAYA_STORE_DIR")]
    store_dir: Option<PathBuf>,
    #[arg(long)]
    device: Option<String>,
    #[arg(long)]
    dtype: Option<String>,
    /// Requests collected into one batched forward pass (1 disables batching).
    #[arg(long)]
    max_batch: Option<usize>,
    /// Collection window for the first request of a batch.
    #[arg(long)]
    batch_wait_ms: Option<u64>,
    /// Allow TF32 tensor-core GEMMs on CUDA.
    #[arg(long)]
    tf32: bool,
}

#[derive(Debug, Args)]
struct AskArgs {
    #[arg(long)]
    model: Option<String>,
    /// A JSON string, object or array; `-` reads stdin.
    #[arg(long)]
    state: String,
    /// A JSON file containing the `questions` object.
    #[arg(long)]
    questions: PathBuf,
    #[arg(long)]
    device: Option<String>,
    /// Weight dtype on CUDA/Metal: f32, f16, or bf16 (CPU always computes in f32).
    #[arg(long)]
    dtype: Option<String>,
    /// Allow TF32 tensor-core GEMMs on CUDA.
    #[arg(long)]
    tf32: bool,
}

#[derive(Debug, Args)]
struct ModelsArgs {
    /// Materialized model directories.
    #[arg(long, env = "OLAYA_MODELS_DIR")]
    models_dir: Option<PathBuf>,
    /// Content-addressed OCI store root.
    #[arg(long, env = "OLAYA_STORE_DIR")]
    store_dir: Option<PathBuf>,
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
        /// Target directory (defaults to OLAYA_REGISTRY_DIR, ./registry, or ~/.olaya/registry).
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
struct DoctorArgs {
    #[arg(long, env = "OLAYA_MODELS_DIR")]
    models_dir: Option<PathBuf>,
    #[arg(long, env = "OLAYA_CACHE_DIR")]
    cache_dir: Option<PathBuf>,
    #[arg(long, env = "OLAYA_TMP_DIR")]
    tmp_dir: Option<PathBuf>,
    #[arg(long, env = "OLAYA_STORE_DIR")]
    store_dir: Option<PathBuf>,
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
    let loaded = LoadedConfig::load(cli.config.as_deref())?;
    let configured_registry = loaded
        .file
        .registry
        .dir
        .as_deref()
        .map(|dir| config::resolve_path(&loaded.base, dir));
    let registry_explicit = cli.registry_dir.is_some() || configured_registry.is_some();
    let registry_dir = cli
        .registry_dir
        .or(configured_registry)
        .unwrap_or_else(olaya::default_registry_dir);
    let registry_store = cli.registry_store || loaded.file.registry.store.unwrap_or(false);
    let allow_remote = !registry_explicit;
    match cli.command {
        #[cfg(feature = "serve")]
        Command::Serve(args) => {
            let runtime = tokio::runtime::Runtime::new().context("creating the Tokio runtime")?;
            runtime.block_on(serve(
                args,
                &loaded,
                &registry_dir,
                registry_store,
                allow_remote,
            ))
        }
        Command::Ask(args) => ask(args, &loaded, &registry_dir, registry_store, allow_remote),
        Command::Models(args) => models(args, &loaded, &registry_dir, registry_store, allow_remote),
        Command::Doctor(args) => doctor(args, &loaded, &registry_dir, registry_store, allow_remote),
    }
}

/// Load the filesystem registry, falling back to the remote catalog when the
/// default directory is missing or empty and the user did not ask for a
/// specific registry.
fn registry_manifests(
    dir: &std::path::Path,
    allow_remote: bool,
    cache_dir: &std::path::Path,
    refresh: bool,
) -> Result<(Vec<RepoManifest>, &'static str)> {
    match olaya::load_manifests(dir) {
        Ok(manifests) => Ok((manifests, "dir")),
        Err(error) if allow_remote && registry_is_empty(dir) => {
            tracing::info!(
                dir = %dir.display(),
                %error,
                "registry directory is unavailable; using the remote catalog"
            );
            let registry = olaya::RemoteRegistry::from_env(cache_dir)?;
            let (manifests, source) = registry.load(refresh).with_context(|| {
                format!(
                    "loading the remote catalog from {}; set OLAYA_REGISTRY_URL, \
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
        Err(error) => Err(error).with_context(|| format!("loading registry {}", dir.display())),
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
    registry_store: bool,
    registry_dir: &std::path::Path,
    hub: HubConfig,
    allow_remote: bool,
) -> Result<olaya::EngineBuilder> {
    let builder = builder.hub(hub.clone());
    if registry_store {
        let store = olaya::Hub::new(hub).store();
        Ok(builder.registry(Arc::new(olaya::StoreRegistry::new(store)))?)
    } else {
        let cache_dir = hub.cache_dir.clone();
        let (manifests, source) =
            registry_manifests(registry_dir, allow_remote, &cache_dir, false)?;
        tracing::debug!(source, models = manifests.len(), "registry loaded");
        Ok(builder.manifests(manifests))
    }
}

fn hub_from(
    loaded: &LoadedConfig,
    models_dir: Option<&Path>,
    cache_dir: Option<&Path>,
    tmp_dir: Option<&Path>,
    store_dir: Option<&Path>,
) -> HubConfig {
    let mut hub = HubConfig::default();
    let file = &loaded.file.hub;
    if let Some(path) = &file.models_dir {
        hub.models_dir = config::resolve_path(&loaded.base, path);
    }
    if let Some(path) = &file.cache_dir {
        hub.cache_dir = config::resolve_path(&loaded.base, path);
    }
    if let Some(path) = &file.tmp_dir {
        hub.tmp_dir = config::resolve_path(&loaded.base, path);
    }
    if let Some(path) = &file.store_dir {
        hub.store_dir = config::resolve_path(&loaded.base, path);
    }
    if let Some(offline) = file.offline {
        hub.offline = offline;
    }
    if let Some(endpoint) = &file.hf_endpoint {
        hub.hf_endpoint = Some(endpoint.clone());
    }
    if let Some(endpoint) = &file.modelscope_endpoint {
        hub.modelscope_endpoint = Some(endpoint.clone());
    }
    if let Some(proxy) = &file.proxy {
        hub.proxy = Some(proxy.clone());
    }
    hub.apply_env();
    if let Some(path) = models_dir {
        hub.models_dir = path.to_path_buf();
    }
    if let Some(path) = cache_dir {
        hub.cache_dir = path.to_path_buf();
    }
    if let Some(path) = tmp_dir {
        hub.tmp_dir = path.to_path_buf();
    }
    if let Some(path) = store_dir {
        hub.store_dir = path.to_path_buf();
    }
    hub
}

fn engine_builder_from(
    loaded: &LoadedConfig,
    device: Option<&str>,
    dtype: Option<&str>,
    tf32: bool,
) -> Result<olaya::EngineBuilder> {
    let device = device
        .or(loaded.file.engine.device.as_deref())
        .unwrap_or("cpu");
    let dtype = dtype
        .or(loaded.file.engine.dtype.as_deref())
        .unwrap_or("f32");
    let mut builder = Engine::builder()
        .builtin_families(BuiltinFamilies::all())
        .device(olaya::olaya_candle::device::resolve(device)?)
        .dtype(parse_dtype(dtype)?)
        .tf32(tf32 || loaded.file.engine.tf32.unwrap_or(false));
    if let Some(model) = &loaded.file.engine.default_model {
        builder = builder.default_model(model.clone());
    }
    Ok(builder)
}

#[cfg(feature = "serve")]
fn engine_from(
    loaded: &LoadedConfig,
    host: &ServeArgs,
    registry_dir: &Path,
    registry_store: bool,
    allow_remote: bool,
) -> Result<Engine> {
    let hub = hub_from(
        loaded,
        host.models_dir.as_deref(),
        host.cache_dir.as_deref(),
        host.tmp_dir.as_deref(),
        host.store_dir.as_deref(),
    );
    let builder = engine_builder_from(
        loaded,
        host.device.as_deref(),
        host.dtype.as_deref(),
        host.tf32,
    )?;
    with_registry(builder, registry_store, registry_dir, hub, allow_remote)?
        .build()
        .context("building the engine")
}

#[cfg(feature = "serve")]
async fn serve(
    args: ServeArgs,
    loaded: &LoadedConfig,
    registry_dir: &Path,
    registry_store: bool,
    allow_remote: bool,
) -> Result<()> {
    let engine = engine_from(loaded, &args, registry_dir, registry_store, allow_remote)?;
    let runtime: Arc<dyn DecisionRuntime> = Arc::new(engine);
    let file = &loaded.file.serve;
    let host = args
        .host
        .or_else(|| file.host.clone())
        .unwrap_or_else(|| "127.0.0.1".to_owned());
    let port = args.port.or(file.port).unwrap_or(11435);
    let api_key = args.api_key.or_else(|| file.api_key.clone());
    let allow_unauthenticated =
        args.allow_unauthenticated || file.allow_unauthenticated.unwrap_or(false);
    let max_batch = args.max_batch.or(file.max_batch).unwrap_or(32);
    let max_wait =
        std::time::Duration::from_millis(args.batch_wait_ms.or(file.batch_wait_ms).unwrap_or(2));
    let config = olaya::serve::ServeConfig {
        host: host.clone(),
        port,
        api_key,
        allow_unauthenticated,
        max_batch,
        max_wait,
        ..olaya::serve::ServeConfig::default()
    };
    olaya::serve::serve(runtime, config).await.with_context(|| {
        format!(
            "serving on {host}:{port}; if the port is already in use, set [serve].port in the \
             config file or pass --port"
        )
    })
}

fn ask(
    args: AskArgs,
    loaded: &LoadedConfig,
    registry_dir: &Path,
    registry_store: bool,
    allow_remote: bool,
) -> Result<()> {
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
        .or_else(|| loaded.file.engine.default_model.clone())
        .context(
            "model is required; pass --model or set [engine].default_model in the config file",
        )?;
    let builder = engine_builder_from(
        loaded,
        args.device.as_deref(),
        args.dtype.as_deref(),
        args.tf32,
    )?;
    let hub = hub_from(loaded, None, None, None, None);
    let engine = with_registry(builder, registry_store, registry_dir, hub, allow_remote)?
        .build()
        .context("building the engine")?;
    let answers = engine.ask_typed(&model, &state, &questions)?;
    println!(
        "{}",
        serde_json::to_string_pretty(answers.as_map()).context("serializing answers")?
    );
    Ok(())
}

fn catalog_engine(
    loaded: &LoadedConfig,
    registry_dir: &Path,
    registry_store: bool,
    models_dir: Option<&Path>,
    store_dir: Option<&Path>,
    allow_remote: bool,
) -> Result<Engine> {
    let hub = hub_from(loaded, models_dir, None, None, store_dir);
    let builder = Engine::builder().builtin_families(BuiltinFamilies::all());
    with_registry(builder, registry_store, registry_dir, hub, allow_remote)?
        .build()
        .context("building the engine")
}

fn default_store(loaded: &LoadedConfig, store_dir: Option<&Path>) -> Arc<dyn olaya::ManifestStore> {
    let hub = hub_from(loaded, None, None, None, store_dir);
    olaya::Hub::new(hub).store()
}

fn models(
    args: ModelsArgs,
    loaded: &LoadedConfig,
    registry_dir: &Path,
    registry_store: bool,
    allow_remote: bool,
) -> Result<()> {
    let ModelsArgs {
        models_dir,
        store_dir,
        command,
    } = args;
    match command {
        ModelsCommand::Stored => {
            let store = default_store(loaded, store_dir.as_deref());
            println!(
                "{}",
                serde_json::to_string_pretty(&store.list()?)
                    .context("serializing stored models")?
            );
        }
        ModelsCommand::Manifest { model } => {
            let store = default_store(loaded, store_dir.as_deref());
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
            let store = default_store(loaded, store_dir.as_deref());
            let stored = store.get(&model)?;
            store.verify(&stored)?;
            println!(
                "{}",
                json!({ "model": model, "digest": stored.digest, "verified": true })
            );
        }
        ModelsCommand::List { all } => {
            let engine = catalog_engine(
                loaded,
                registry_dir,
                registry_store,
                models_dir.as_deref(),
                store_dir.as_deref(),
                allow_remote,
            )?;
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
            let engine = catalog_engine(
                loaded,
                registry_dir,
                registry_store,
                models_dir.as_deref(),
                store_dir.as_deref(),
                allow_remote,
            )?;
            let manifest = engine
                .resolve(&model)
                .ok_or_else(|| anyhow::anyhow!("unknown model: {model}"))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&manifest).context("serializing manifest")?
            );
        }
        ModelsCommand::Pull { model } => {
            let engine = catalog_engine(
                loaded,
                registry_dir,
                registry_store,
                models_dir.as_deref(),
                store_dir.as_deref(),
                allow_remote,
            )?;
            let manifest = engine.pull(&model)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&manifest).context("serializing manifest")?
            );
        }
        ModelsCommand::Add { manifest } => {
            let engine = catalog_engine(
                loaded,
                registry_dir,
                registry_store,
                models_dir.as_deref(),
                store_dir.as_deref(),
                allow_remote,
            )?;
            let raw = std::fs::read_to_string(&manifest)
                .with_context(|| format!("reading {}", manifest.display()))?;
            let manifest: RepoManifest = serde_json::from_str(&raw).context("parsing manifest")?;
            engine.add_manifest(manifest)?;
            println!("{}", json!({ "added": true }));
        }
        ModelsCommand::Remove { model } => {
            let engine = catalog_engine(
                loaded,
                registry_dir,
                registry_store,
                models_dir.as_deref(),
                store_dir.as_deref(),
                allow_remote,
            )?;
            let removed = engine.delete(&model)?;
            println!("{}", json!({ "model": model, "removed": removed }));
        }
        ModelsCommand::Export { dir, force } => {
            let dir = dir.unwrap_or_else(olaya::default_registry_dir);
            let (manifests, source) = if registry_store {
                let store = default_store(loaded, store_dir.as_deref());
                let registry = olaya::StoreRegistry::new(store);
                (olaya::ManifestRegistry::manifests(&registry)?, "store")
            } else {
                let hub = hub_from(loaded, None, None, None, None);
                registry_manifests(registry_dir, allow_remote, &hub.cache_dir, true)?
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

fn doctor(
    args: DoctorArgs,
    loaded: &LoadedConfig,
    registry_dir: &Path,
    registry_store: bool,
    allow_remote: bool,
) -> Result<()> {
    let hub = hub_from(
        loaded,
        args.models_dir.as_deref(),
        args.cache_dir.as_deref(),
        args.tmp_dir.as_deref(),
        args.store_dir.as_deref(),
    );
    let (registry_models, registry_source, registry_error) = if registry_store {
        let store = olaya::Hub::new(hub.clone()).store();
        let registry = olaya::StoreRegistry::new(store);
        match olaya::ManifestRegistry::manifests(&registry) {
            Ok(manifests) => (manifests.len(), "store", None),
            Err(error) => (0, "store", Some(error.to_string())),
        }
    } else {
        match registry_manifests(registry_dir, allow_remote, &hub.cache_dir, false) {
            Ok((manifests, source)) => (manifests.len(), source, None),
            Err(error) => (0, "dir", Some(error.to_string())),
        }
    };
    let store_dir_exists = hub.store_dir.exists();
    let stored_models = olaya::Store::open(&hub.store_dir)
        .list()
        .unwrap_or_default();
    let report = json!({
        "config_file": loaded.path.as_ref().map(|path| path.display().to_string()),
        "registry_dir": registry_dir.display().to_string(),
        "registry_source": registry_source,
        "registry_models": registry_models,
        "registry_error": registry_error,
        "registry_url": std::env::var(olaya::REGISTRY_URL_ENV)
            .ok()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| olaya::DEFAULT_REGISTRY_URL.to_owned()),
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
        "store_dir_exists": store_dir_exists,
        "stored_models": stored_models,
        "offline": hub.offline,
        "hf_endpoint": hub.hf_endpoint,
    });
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
