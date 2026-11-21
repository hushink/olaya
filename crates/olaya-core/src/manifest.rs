//! Model manifests.
//!
//! A model author writes [`RepoManifest`] as `olaya.manifest.json`; the hub
//! converts it into an OCI distribution manifest ([`OciManifest`]) stored in
//! the content-addressed model store.

use std::collections::{BTreeMap, HashSet};
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::decision::{CalibrationMode, Capabilities};
use crate::model::{ModelMetadata, is_valid_release_date};

pub const REPO_MANIFEST_FILE: &str = "olaya.manifest.json";
pub const REPO_MANIFEST_SCHEMA: u32 = 1;

pub const MEDIA_TYPE_MANIFEST: &str = "application/vnd.docker.distribution.manifest.v2+json";
pub const MEDIA_CONFIG: &str = "application/vnd.olaya.config.v1+json";
pub const MEDIA_SAFETENSORS: &str = "application/vnd.olaya.candle.safetensors";
pub const MEDIA_PICKLE: &str = "application/vnd.olaya.candle.pickle";
pub const MEDIA_TOKENIZER: &str = "application/vnd.olaya.tokenizer";
pub const MEDIA_CALIBRATION: &str = "application/vnd.olaya.calibration";
pub const MEDIA_DECISION: &str = "application/vnd.olaya.decision";
pub const MEDIA_LICENSE: &str = "application/vnd.olaya.license";
pub const MEDIA_README: &str = "application/vnd.olaya.readme";

/// Descriptor annotation: the repository-relative file path.
pub const ANNOTATION_PATH: &str = "org.olaya.path";
/// Descriptor annotation: the `ManifestFile.role` name.
pub const ANNOTATION_ROLE: &str = "org.olaya.role";

/// The manifest a model repository ships.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepoManifest {
    pub schema_version: u32,
    pub name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    pub family: String,
    /// `candle` or `remote`.
    #[serde(default = "default_engine")]
    pub engine: String,
    /// Backbone architecture: `modernbert`, `qwen3`, `qwen35`, or a custom id.
    pub arch: String,
    /// `safetensors`, `pytorch_pickle`, `remote`, or a family-specific format.
    #[serde(default = "default_format")]
    pub format: String,
    /// Sequence layout id, e.g. `laya-markers-v1`.
    pub layout: String,
    /// Repository subdirectory that holds the files, e.g. `multilingual`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subfolder: Option<String>,
    #[serde(default)]
    pub files: Vec<ManifestFile>,
    #[serde(default)]
    pub calibration_mode: CalibrationMode,
    #[serde(default)]
    pub capabilities: Capabilities,
    /// Set when this manifest is a router: requests dispatch to another model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub router: Option<RouterConfig>,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub release_date: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Hub revision (branch, tag, or commit) to pull from `source`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}

/// A router manifest: this model dispatches each request to another model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Routing strategy; `script` routes on the state's script and language.
    pub strategy: String,
    /// Route name -> model name, e.g. `english` -> `laya`.
    #[serde(default)]
    pub routes: BTreeMap<String, String>,
    /// Target when no route matches.
    pub default: String,
}

fn default_engine() -> String {
    "candle".to_owned()
}

fn default_format() -> String {
    "safetensors".to_owned()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ManifestFile {
    /// Repository-relative, forward-slash path.
    pub path: String,
    pub role: FileRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    /// Absolute URL, or empty to resolve against the repository.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileRole {
    Weights,
    Tokenizer,
    Calibration,
    Decision,
    Config,
    Pickle,
    Other,
}

impl FileRole {
    pub fn name(self) -> &'static str {
        match self {
            Self::Weights => "weights",
            Self::Tokenizer => "tokenizer",
            Self::Calibration => "calibration",
            Self::Decision => "decision",
            Self::Config => "config",
            Self::Pickle => "pickle",
            Self::Other => "other",
        }
    }

    /// Parse the value stored in the `org.olaya.role` annotation.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "weights" => Some(Self::Weights),
            "tokenizer" => Some(Self::Tokenizer),
            "calibration" => Some(Self::Calibration),
            "decision" => Some(Self::Decision),
            "config" => Some(Self::Config),
            "pickle" => Some(Self::Pickle),
            "other" => Some(Self::Other),
            _ => None,
        }
    }

    pub fn media_type(self) -> &'static str {
        match self {
            Self::Weights => MEDIA_SAFETENSORS,
            Self::Tokenizer => MEDIA_TOKENIZER,
            Self::Calibration => MEDIA_CALIBRATION,
            Self::Decision => MEDIA_DECISION,
            Self::Config => MEDIA_CONFIG,
            Self::Pickle => MEDIA_PICKLE,
            Self::Other => "application/octet-stream",
        }
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ManifestError {
    #[error("unsupported schema_version: {0}")]
    SchemaVersion(u32),
    #[error("field `{0}` must not be empty")]
    Empty(&'static str),
    #[error("unsupported engine: {0}")]
    Engine(String),
    #[error("engine `router` requires a router config")]
    MissingRouter,
    #[error("a router config requires engine `router`, got `{0}`")]
    RouterEngine(String),
    #[error("router manifests must define at least one route")]
    NoRoutes,
    #[error("router route `{0}` must name a model")]
    EmptyRoute(String),
    #[error("manifest must contain at least one file")]
    NoFiles,
    #[error("duplicate file path: {0}")]
    DuplicatePath(String),
    #[error("file path must be relative, use forward slashes, and contain no `..`: {0}")]
    InvalidPath(String),
    #[error("invalid sha256 for `{path}`: {value}")]
    InvalidSha256 { path: String, value: String },
    #[error("file size must be greater than zero: {0}")]
    ZeroSize(String),
    #[error("duplicate alias: {0}")]
    DuplicateAlias(String),
    #[error("alias must not equal its model name: {0}")]
    AliasEqualsName(String),
    #[error("invalid release_date: {0}")]
    ReleaseDate(String),
}

impl RepoManifest {
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.schema_version != REPO_MANIFEST_SCHEMA {
            return Err(ManifestError::SchemaVersion(self.schema_version));
        }
        for (field, value) in [
            ("name", self.name.as_str()),
            ("family", self.family.as_str()),
            ("arch", self.arch.as_str()),
            ("layout", self.layout.as_str()),
            ("format", self.format.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(ManifestError::Empty(field));
            }
        }
        if self.engine != "candle" && self.engine != "remote" && self.engine != "router" {
            return Err(ManifestError::Engine(self.engine.clone()));
        }
        if self.engine == "router" && self.router.is_none() {
            return Err(ManifestError::MissingRouter);
        }
        if self.router.is_some() && self.engine != "router" {
            return Err(ManifestError::RouterEngine(self.engine.clone()));
        }
        if self.engine == "candle" && self.files.is_empty() {
            return Err(ManifestError::NoFiles);
        }
        if let Some(router) = &self.router {
            if router.strategy.trim().is_empty() {
                return Err(ManifestError::Empty("router.strategy"));
            }
            if router.routes.is_empty() {
                return Err(ManifestError::NoRoutes);
            }
            if router.default.trim().is_empty() {
                return Err(ManifestError::Empty("router.default"));
            }
            for (route, target) in &router.routes {
                if route.trim().is_empty() {
                    return Err(ManifestError::Empty("router.routes"));
                }
                if target.trim().is_empty() {
                    return Err(ManifestError::EmptyRoute(route.clone()));
                }
            }
        }
        if let Some(date) = &self.release_date
            && !is_valid_release_date(date)
        {
            return Err(ManifestError::ReleaseDate(date.clone()));
        }
        if let Some(subfolder) = &self.subfolder {
            if subfolder.trim().is_empty() {
                return Err(ManifestError::Empty("subfolder"));
            }
            if !is_valid_relative_path(subfolder) {
                return Err(ManifestError::InvalidPath(subfolder.clone()));
            }
        }

        let mut paths = HashSet::new();
        for file in &self.files {
            if !is_valid_relative_path(&file.path) {
                return Err(ManifestError::InvalidPath(file.path.clone()));
            }
            if !paths.insert(file.path.clone()) {
                return Err(ManifestError::DuplicatePath(file.path.clone()));
            }
            if file.size == Some(0) {
                return Err(ManifestError::ZeroSize(file.path.clone()));
            }
            if let Some(value) = &file.sha256
                && !is_hex(value, 64)
            {
                return Err(ManifestError::InvalidSha256 {
                    path: file.path.clone(),
                    value: value.clone(),
                });
            }
        }

        let mut aliases = HashSet::new();
        for alias in &self.aliases {
            if alias.trim().is_empty() {
                return Err(ManifestError::Empty("aliases"));
            }
            if alias == &self.name {
                return Err(ManifestError::AliasEqualsName(alias.clone()));
            }
            if !aliases.insert(alias) {
                return Err(ManifestError::DuplicateAlias(alias.clone()));
            }
        }
        Ok(())
    }

    pub fn file(&self, role: FileRole) -> Option<&ManifestFile> {
        self.files.iter().find(|file| file.role == role)
    }

    pub fn metadata(&self) -> ModelMetadata {
        ModelMetadata::new(
            self.name.clone(),
            self.description.clone(),
            self.release_date.clone().unwrap_or_default(),
        )
    }

    pub fn alias_metadata(&self) -> Vec<ModelMetadata> {
        self.aliases
            .iter()
            .map(|alias| {
                ModelMetadata::new(
                    alias.clone(),
                    self.description.clone(),
                    self.release_date.clone().unwrap_or_default(),
                )
            })
            .collect()
    }
}

fn is_valid_relative_path(value: &str) -> bool {
    if value.is_empty() || value.contains('\\') || value.contains('\0') {
        return false;
    }
    let path = Path::new(value);
    if path.is_absolute() {
        return false;
    }
    path.components()
        .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
}

fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

// ---------------------------------------------------------------------------
// OCI distribution types and the model config blob.

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Descriptor {
    pub media_type: String,
    pub digest: String,
    pub size: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub urls: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub annotations: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OciManifest {
    pub schema_version: u32,
    pub media_type: String,
    pub config: Descriptor,
    pub layers: Vec<Descriptor>,
}

impl OciManifest {
    pub fn layer(&self, media_type: &str) -> Option<&Descriptor> {
        self.layers
            .iter()
            .find(|layer| layer.media_type == media_type)
    }

    pub fn blobs(&self) -> impl Iterator<Item = &Descriptor> {
        std::iter::once(&self.config).chain(self.layers.iter())
    }

    pub fn total_size(&self) -> u64 {
        self.blobs().map(|descriptor| descriptor.size).sum()
    }
}

/// The config blob: model metadata stored alongside the OCI layers.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    /// `safetensors`, `pytorch_pickle`, `remote`, or a family-specific format.
    pub model_format: String,
    pub family: String,
    #[serde(default)]
    pub arch: String,
    #[serde(default)]
    pub parameter_size: String,
    #[serde(default)]
    pub context_length: u32,
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default)]
    pub license: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub release_date: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<String>,
    #[serde(default)]
    pub calibration_mode: CalibrationMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subfolder: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Capabilities>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub router: Option<RouterConfig>,
}
