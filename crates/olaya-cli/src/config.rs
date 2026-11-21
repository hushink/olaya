//! TOML configuration shared by `serve`, `ask`, and the model commands.
//!
//! Precedence is CLI flag > environment variable > file value > built-in
//! default. Relative paths are resolved against the configuration file.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

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
    pub fn load(explicit: Option<&Path>) -> Result<Self> {
        if let Some(path) = explicit {
            return Self::read(path);
        }
        for path in default_paths() {
            if path.is_file() {
                return Self::read(&path);
            }
        }
        Ok(Self {
            file: FileConfig::default(),
            path: None,
            base: std::env::current_dir().context("reading the current directory")?,
        })
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
}
