//! TOML configuration shared by `serve`, `ask`, and the model commands.
//!
//! Precedence is CLI flag > environment variable > file value > built-in
//! default. File paths are resolved against the configuration file; CLI/env
//! paths are resolved against the current working directory.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

/// Built-in engine defaults. Settings are resolved as
/// CLI/env > file > these defaults.
pub const DEFAULT_DEVICE: &str = "auto";
pub const DEFAULT_DTYPE: &str = "f32";
pub const DEFAULT_TF32: bool = false;
pub const DEFAULT_MODEL: &str = "laya:latest";
pub const DEFAULT_MAX_LOADED: usize = 3;

/// Built-in server defaults, mirrored by `olaya::serve::ServeConfig::default`.
pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_PORT: u16 = 11435;
pub const DEFAULT_ALLOW_UNAUTHENTICATED: bool = false;
pub const DEFAULT_MAX_BATCH: usize = 32;
pub const DEFAULT_BATCH_WAIT_MS: u64 = 2;
pub const DEFAULT_MAX_BODY_BYTES: usize = olaya::MAX_BODY_BYTES;
pub const DEFAULT_MAX_QUEUE: usize = 64;

/// Built-in catalog defaults.
pub const DEFAULT_REGISTRY_STORE: bool = false;
pub const DEFAULT_REGISTRY_TIMEOUT_SECS: u64 = olaya::DEFAULT_REGISTRY_TIMEOUT_SECS;

/// Template written to `$OLAYA_HOME/config.toml` on first use.
///
/// The device default is `auto` so a CUDA/Metal build uses its accelerator
/// without baking a machine-specific value into the generated file.
pub fn default_config_text() -> String {
    format!(
        r#"# olaya shared configuration.
#
# Precedence: CLI flag > environment variable > this file > built-in default.
# Relative paths are resolved against this file's directory.

[engine]
# `auto` uses CUDA when this build supports it and a GPU is usable, then Metal,
# and falls back to CPU. Accelerators compiled into this binary: {accelerators}.
device = "{device}"
# Weight dtype on CUDA/Metal: f32, f16, or bf16 (CPU always computes in f32).
dtype = "{dtype}"
tf32 = {tf32}
default_model = "{default_model}"
# Maximum number of models kept loaded at once.
max_loaded = {max_loaded}

[serve]
host = "{host}"
port = {port}
# Set a bearer token to protect every route except /health.
# api_key = "..."
allow_unauthenticated = {allow_unauthenticated}
max_batch = {max_batch}
batch_wait_ms = {batch_wait_ms}
max_body_bytes = {max_body_bytes}
max_queue = {max_queue}

[hub]
# All paths default under $OLAYA_HOME; relative paths resolve against this file.
# cache_dir/tmp_dir are disposable and must not overlap models_dir/store_dir.
# models_dir = "models"
# cache_dir = "cache"
# tmp_dir = "tmp"
# store_dir = "store"
# offline = false
# Hugging Face downloads use the mirror by default.
# hf_endpoint = "https://hf-mirror.com"
# modelscope_endpoint = "https://www.modelscope.cn"
# proxy = "http://127.0.0.1:7890"

[registry]
# Use a local manifest directory instead of the release catalog.
# dir = "./registry"
# store = false
# url = "{registry_url}"
timeout_secs = {registry_timeout_secs}
"#,
        accelerators = olaya::olaya_candle::device::compiled_accelerators(),
        device = DEFAULT_DEVICE,
        dtype = DEFAULT_DTYPE,
        tf32 = DEFAULT_TF32,
        default_model = DEFAULT_MODEL,
        max_loaded = DEFAULT_MAX_LOADED,
        host = DEFAULT_HOST,
        port = DEFAULT_PORT,
        allow_unauthenticated = DEFAULT_ALLOW_UNAUTHENTICATED,
        max_batch = DEFAULT_MAX_BATCH,
        batch_wait_ms = DEFAULT_BATCH_WAIT_MS,
        max_body_bytes = DEFAULT_MAX_BODY_BYTES,
        max_queue = DEFAULT_MAX_QUEUE,
        registry_url = olaya::DEFAULT_REGISTRY_URL,
        registry_timeout_secs = DEFAULT_REGISTRY_TIMEOUT_SECS
    )
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileConfig {
    #[serde(default)]
    pub registry: RegistryConfig,
    #[serde(default)]
    pub hub: HubConfigFile,
    #[serde(default)]
    pub engine: EngineConfig,
    #[serde(default)]
    #[cfg_attr(not(feature = "serve"), allow(dead_code))]
    pub serve: ServeConfigFile,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryConfig {
    pub dir: Option<PathBuf>,
    pub store: Option<bool>,
    pub url: Option<String>,
    pub timeout_secs: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HubConfigFile {
    pub models_dir: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
    pub tmp_dir: Option<PathBuf>,
    pub store_dir: Option<PathBuf>,
    pub offline: Option<bool>,
    pub hf_endpoint: Option<String>,
    pub modelscope_endpoint: Option<String>,
    pub proxy: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineConfig {
    pub device: Option<String>,
    pub dtype: Option<String>,
    pub tf32: Option<bool>,
    pub default_model: Option<String>,
    pub max_loaded: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(not(feature = "serve"), allow(dead_code))]
pub struct ServeConfigFile {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub api_key: Option<String>,
    pub allow_unauthenticated: Option<bool>,
    pub max_batch: Option<usize>,
    pub batch_wait_ms: Option<u64>,
    pub max_body_bytes: Option<usize>,
    pub max_queue: Option<usize>,
}

#[derive(Debug)]
pub struct LoadedConfig {
    pub file: FileConfig,
    pub path: Option<PathBuf>,
    pub base: PathBuf,
}

impl LoadedConfig {
    /// Load an explicit `--config` / `OLAYA_CONFIG` path, otherwise search
    /// `./olaya.toml`, `./sys1.toml`, and `$OLAYA_HOME/config.toml`.
    ///
    /// If none exists, write a feature-aware template to
    /// `$OLAYA_HOME/config.toml` and load it. A missing home directory or a
    /// read-only filesystem is not fatal: the command continues with defaults.
    pub fn load(explicit: Option<&Path>) -> Result<Self> {
        if let Some(path) = explicit {
            return Self::read(path);
        }
        let paths = default_paths();
        for path in &paths {
            if path.is_file() {
                return Self::read(path);
            }
        }
        let path = paths
            .last()
            .expect("default_paths always contains the user configuration");
        match Self::write_default(path) {
            Ok(()) => {
                tracing::info!(path = %path.display(), "created default configuration");
                Self::read(path)
            }
            Err(error) => {
                tracing::warn!(
                    path = %path.display(),
                    %error,
                    "could not create the default configuration; using built-in defaults"
                );
                Ok(Self {
                    file: FileConfig::default(),
                    path: None,
                    base: std::env::current_dir().context("reading the current directory")?,
                })
            }
        }
    }

    fn write_default(path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
        {
            Ok(mut file) => file
                .write_all(default_config_text().as_bytes())
                .with_context(|| format!("writing configuration {}", path.display())),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
            Err(error) => {
                Err(error).with_context(|| format!("creating configuration {}", path.display()))
            }
        }
    }

    fn read(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading configuration {}", path.display()))?;
        let file: FileConfig = toml::from_str(&text)
            .with_context(|| format!("parsing configuration {}", path.display()))?;
        let base = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Ok(Self {
            file,
            path: Some(path.to_path_buf()),
            base,
        })
    }
}

pub fn resolve_path(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

/// Make a path absolute and remove `.`/`..` components for overlap checks.
pub fn normalized_absolute(path: &Path) -> Result<PathBuf> {
    let absolute =
        std::path::absolute(path).with_context(|| format!("resolving {}", path.display()))?;
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

fn default_paths() -> Vec<PathBuf> {
    vec![
        PathBuf::from("olaya.toml"),
        PathBuf::from("sys1.toml"),
        olaya::hub::default_root().join("config.toml"),
    ]
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "serve")]
    use std::time::Duration;

    use super::*;

    #[test]
    fn parses_the_shared_sections() {
        let config: FileConfig = toml::from_str(
            r#"
[registry]
dir = "catalog"
store = false
url = "https://example.com/catalog.json"
timeout_secs = 12

[hub]
models_dir = "models"
offline = true

[engine]
device = "cuda"
dtype = "f16"
tf32 = true
default_model = "laya:latest"
max_loaded = 2

[serve]
host = "0.0.0.0"
port = 11436
max_batch = 8
max_body_bytes = 4096
max_queue = 16
"#,
        )
        .expect("config parses");
        assert_eq!(config.registry.dir, Some(PathBuf::from("catalog")));
        assert_eq!(config.registry.timeout_secs, Some(12));
        assert_eq!(config.engine.device.as_deref(), Some("cuda"));
        assert_eq!(config.engine.dtype.as_deref(), Some("f16"));
        assert_eq!(config.engine.max_loaded, Some(2));
        assert_eq!(config.serve.port, Some(11436));
        assert_eq!(config.serve.max_queue, Some(16));
        assert_eq!(config.hub.offline, Some(true));
    }

    #[test]
    fn resolves_relative_paths_against_the_config_directory() {
        let base = Path::new("config-dir");
        assert_eq!(resolve_path(base, Path::new("models")), base.join("models"));
        let absolute = std::env::current_dir()
            .expect("current directory")
            .join("models");
        assert_eq!(resolve_path(base, &absolute), absolute);
    }

    #[test]
    fn normalized_absolute_removes_parent_components() {
        let normalized = normalized_absolute(Path::new("cache/../models")).expect("normalizes");
        assert!(normalized.is_absolute());
        assert!(normalized.ends_with("models"));
        assert!(!normalized.to_string_lossy().contains("cache"));
    }

    #[test]
    fn generated_template_is_parseable_and_feature_aware() {
        let text = default_config_text();
        let config: FileConfig = toml::from_str(&text).expect("generated template parses");
        assert_eq!(config.engine.device.as_deref(), Some(DEFAULT_DEVICE));
        assert_eq!(config.engine.dtype.as_deref(), Some(DEFAULT_DTYPE));
        assert_eq!(config.engine.tf32, Some(DEFAULT_TF32));
        assert_eq!(config.engine.default_model.as_deref(), Some(DEFAULT_MODEL));
        assert_eq!(config.engine.max_loaded, Some(DEFAULT_MAX_LOADED));
        assert_eq!(config.serve.host.as_deref(), Some(DEFAULT_HOST));
        assert_eq!(config.serve.port, Some(DEFAULT_PORT));
        assert_eq!(config.serve.max_batch, Some(DEFAULT_MAX_BATCH));
        assert_eq!(config.serve.batch_wait_ms, Some(DEFAULT_BATCH_WAIT_MS));
        assert_eq!(config.serve.max_body_bytes, Some(DEFAULT_MAX_BODY_BYTES));
        assert_eq!(config.serve.max_queue, Some(DEFAULT_MAX_QUEUE));
        assert_eq!(
            config.registry.timeout_secs,
            Some(DEFAULT_REGISTRY_TIMEOUT_SECS)
        );
        assert!(!text.contains("workspace"));
        assert!(!text.contains("RustRover"));
        assert!(text.contains(olaya::olaya_candle::device::compiled_accelerators()));
    }

    #[cfg(feature = "serve")]
    #[test]
    fn cli_defaults_match_serve_config_defaults() {
        let serve = olaya::serve::ServeConfig::default();
        assert_eq!(DEFAULT_HOST, serve.host);
        assert_eq!(DEFAULT_PORT, serve.port);
        assert_eq!(DEFAULT_ALLOW_UNAUTHENTICATED, serve.allow_unauthenticated);
        assert_eq!(DEFAULT_MAX_BATCH, serve.max_batch);
        assert_eq!(DEFAULT_MAX_BODY_BYTES, serve.max_body_bytes);
        assert_eq!(DEFAULT_MAX_QUEUE, serve.max_queue);
        assert_eq!(Duration::from_millis(DEFAULT_BATCH_WAIT_MS), serve.max_wait);
    }
}
