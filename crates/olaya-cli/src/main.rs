use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
#[cfg(feature = "serve")]
use olaya::DecisionRuntime;
use olaya::{BuiltinFamilies, Engine, HubConfig, Question, Questions, RepoManifest, State};
use serde_json::json;

#[derive(Debug, Parser)]
#[command(name = "olaya", version, about = "System One decision-model runtime")]
struct Cli {
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
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 11435)]
    port: u16,
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
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value = "f32")]
    dtype: String,
    /// Requests collected into one batched forward pass (1 disables batching).
    #[arg(long, default_value_t = 32)]
    max_batch: usize,
    /// Collection window for the first request of a batch.
    #[arg(long, default_value_t = 2)]
    batch_wait_ms: u64,
    /// Allow TF32 tensor-core GEMMs on CUDA.
    #[arg(long)]
    tf32: bool,
}

#[derive(Debug, Args)]
struct AskArgs {
    #[arg(long)]
    model: String,
    /// A JSON string, object or array; `-` reads stdin.
    #[arg(long)]
    state: String,
    /// A JSON file containing the `questions` object.
    #[arg(long)]
    questions: PathBuf,
    #[arg(long, default_value = "cpu")]
    device: String,
    /// Weight dtype on CUDA/Metal: f32, f16, or bf16 (CPU always computes in f32).
    #[arg(long, default_value = "f32")]
    dtype: String,
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
    List,
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
    let registry_explicit = cli.registry_dir.is_some()
        || std::env::var_os(olaya::registry::REGISTRY_ENV).is_some_and(|value| !value.is_empty());
    let registry_dir = cli.registry_dir.unwrap_or_else(olaya::default_registry_dir);
    let registry_store = cli.registry_store;
    let allow_remote = !registry_explicit;
    match cli.command {
        #[cfg(feature = "serve")]
        Command::Serve(args) => {
            let runtime = tokio::runtime::Runtime::new().context("creating the Tokio runtime")?;
            runtime.block_on(serve(args, &registry_dir, registry_store, allow_remote))
        }
        Command::Ask(args) => ask(args, &registry_dir, registry_store, allow_remote),
        Command::Models(args) => models(args, &registry_dir, registry_store, allow_remote),
        Command::Doctor(args) => doctor(args, &registry_dir, registry_store, allow_remote),
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

#[cfg(feature = "serve")]
fn engine_from(
    host: &ServeArgs,
    registry_dir: &std::path::Path,
    registry_store: bool,
    allow_remote: bool,
) -> Result<Engine> {
    let mut hub = HubConfig::from_env();
    if let Some(dir) = &host.models_dir {
        hub.models_dir = dir.clone();
    }
    if let Some(dir) = &host.cache_dir {
        hub.cache_dir = dir.clone();
    }
    if let Some(dir) = &host.tmp_dir {
        hub.tmp_dir = dir.clone();
    }
    if let Some(dir) = &host.store_dir {
        hub.store_dir = dir.clone();
    }
    let device = olaya::olaya_candle::device::resolve(&host.device)?;
    let dtype = parse_dtype(&host.dtype)?;
    let builder = Engine::builder()
        .builtin_families(BuiltinFamilies::all())
        .device(device)
        .dtype(dtype)
        .tf32(host.tf32);
    with_registry(builder, registry_store, registry_dir, hub, allow_remote)?
        .build()
        .context("building the engine")
}

#[cfg(feature = "serve")]
async fn serve(
    args: ServeArgs,
    registry_dir: &std::path::Path,
    registry_store: bool,
    allow_remote: bool,
) -> Result<()> {
    let engine = engine_from(&args, registry_dir, registry_store, allow_remote)?;
    let runtime: Arc<dyn DecisionRuntime> = Arc::new(engine);
    let config = olaya::serve::ServeConfig {
        host: args.host,
        port: args.port,
        api_key: args.api_key,
        allow_unauthenticated: args.allow_unauthenticated,
        max_batch: args.max_batch,
        max_wait: std::time::Duration::from_millis(args.batch_wait_ms),
        ..olaya::serve::ServeConfig::default()
    };
    olaya::serve::serve(runtime, config)
        .await
        .context("serving")
}

fn ask(
    args: AskArgs,
    registry_dir: &std::path::Path,
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

    let device = olaya::olaya_candle::device::resolve(&args.device)?;
    let dtype = parse_dtype(&args.dtype)?;
    let builder = Engine::builder()
        .builtin_families(BuiltinFamilies::all())
        .device(device)
        .dtype(dtype)
        .tf32(args.tf32);
    let engine = with_registry(
        builder,
        registry_store,
        registry_dir,
        HubConfig::from_env(),
        allow_remote,
    )?
    .build()
    .context("building the engine")?;
    let answers = engine.ask_typed(&args.model, &state, &questions)?;
    println!(
        "{}",
        serde_json::to_string_pretty(answers.as_map()).context("serializing answers")?
    );
    Ok(())
}

fn catalog_engine(
    registry_dir: &std::path::Path,
    registry_store: bool,
    models_dir: Option<&std::path::Path>,
    store_dir: Option<&std::path::Path>,
    allow_remote: bool,
) -> Result<Engine> {
    let mut hub = HubConfig::from_env();
    if let Some(dir) = models_dir {
        hub.models_dir = dir.to_path_buf();
    }
    if let Some(dir) = store_dir {
        hub.store_dir = dir.to_path_buf();
    }
    let builder = Engine::builder().builtin_families(BuiltinFamilies::all());
    with_registry(builder, registry_store, registry_dir, hub, allow_remote)?
        .build()
        .context("building the engine")
}

fn default_store(store_dir: Option<&std::path::Path>) -> Arc<dyn olaya::ManifestStore> {
    let mut hub = HubConfig::from_env();
    if let Some(dir) = store_dir {
        hub.store_dir = dir.to_path_buf();
    }
    olaya::Hub::new(hub).store()
}

fn models(
    args: ModelsArgs,
    registry_dir: &std::path::Path,
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
            let store = default_store(store_dir.as_deref());
            println!(
                "{}",
                serde_json::to_string_pretty(&store.list()?)
                    .context("serializing stored models")?
            );
        }
        ModelsCommand::Manifest { model } => {
            let store = default_store(store_dir.as_deref());
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
            let store = default_store(store_dir.as_deref());
            let stored = store.get(&model)?;
            store.verify(&stored)?;
            println!(
                "{}",
                json!({ "model": model, "digest": stored.digest, "verified": true })
            );
        }
        ModelsCommand::List => {
            let engine = catalog_engine(
                registry_dir,
                registry_store,
                models_dir.as_deref(),
                store_dir.as_deref(),
                allow_remote,
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&engine.list_models())
                    .context("serializing models")?
            );
        }
        ModelsCommand::Show { model } => {
            let engine = catalog_engine(
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
                let store = default_store(store_dir.as_deref());
                let registry = olaya::StoreRegistry::new(store);
                (olaya::ManifestRegistry::manifests(&registry)?, "store")
            } else {
                let hub = HubConfig::from_env();
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
    registry_dir: &std::path::Path,
    registry_store: bool,
    allow_remote: bool,
) -> Result<()> {
    let mut hub = HubConfig::from_env();
    if let Some(dir) = args.models_dir {
        hub.models_dir = dir;
    }
    if let Some(dir) = args.cache_dir {
        hub.cache_dir = dir;
    }
    if let Some(dir) = args.tmp_dir {
        hub.tmp_dir = dir;
    }
    if let Some(dir) = args.store_dir {
        hub.store_dir = dir;
    }
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
