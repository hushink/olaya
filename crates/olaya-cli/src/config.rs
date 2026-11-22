//! TOML configuration shared by `serve`, `ask`, and the model commands.
//!
//! Precedence is CLI flag > environment variable > file value > built-in
//! default. Relative paths are resolved against the configuration file.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

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
device = "auto"
# Weight dtype on CUDA/Metal: f32, f16, or bf16 (CPU always computes in f32).
dtype = "f32"
tf32 = false
default_model = "laya:latest"

[serve]
host = "127.0.0.1"
port = 11435
max_batch = 32
batch_wait_ms = 2

[hub]
# Hugging Face downloads use the mirror by default.
# hf_endpoint = "https://hf-mirror.com"

[registry]
# Use a local manifest directory instead of the release catalog.
# dir = "./registry"
# store = false
"#,
        accelerators = olaya::olaya_candle::device::compiled_accelerators()
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
    pub serve: ServeConfigFile,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryConfig {
    pub dir: Option<PathBuf>,
    pub store: Option<bool>,
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
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServeConfigFile {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub api_key: Option<String>,
    pub allow_unauthenticated: Option<bool>,
    pub max_batch: Option<usize>,
    pub batch_wait_ms: Option<u64>,
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

fn default_paths() -> Vec<PathBuf> {
    vec![
        PathBuf::from("olaya.toml"),
        PathBuf::from("sys1.toml"),
        olaya::hub::default_root().join("config.toml"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_shared_sections() {
        let config: FileConfig = toml::from_str(
            r#"
[registry]
dir = "catalog"
store = false

[hub]
models_dir = "models"
offline = true

[engine]
device = "cuda"
dtype = "f16"
tf32 = true
default_model = "laya:latest"

[serve]
host = "0.0.0.0"
port = 11436
max_batch = 8
"#,
        )
        .expect("config parses");
        assert_eq!(config.registry.dir, Some(PathBuf::from("catalog")));
        assert_eq!(config.engine.device.as_deref(), Some("cuda"));
        assert_eq!(config.engine.dtype.as_deref(), Some("f16"));
        assert_eq!(config.serve.port, Some(11436));
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
    fn generated_template_is_parseable_and_feature_aware() {
        let text = default_config_text();
        let config: FileConfig = toml::from_str(&text).expect("generated template parses");
        assert_eq!(config.engine.device.as_deref(), Some("auto"));
        assert_eq!(config.engine.default_model.as_deref(), Some("laya:latest"));
        assert!(!text.contains("workspace"));
        assert!(!text.contains("RustRover"));
        assert!(text.contains(olaya::olaya_candle::device::compiled_accelerators()));
    }
}
