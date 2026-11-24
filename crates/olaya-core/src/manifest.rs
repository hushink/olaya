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
    /// Absolute `http://`/`https://` URL. When absent, the file is resolved
    /// against the model repository.
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
    #[error("file path must be a safe relative path using forward slashes: {0}")]
    InvalidPath(String),
    #[error("invalid model name: {0}")]
    InvalidName(String),
    #[error("invalid family name: {0}")]
    InvalidFamily(String),
    #[error("invalid alias: {0}")]
    InvalidAlias(String),
    #[error("invalid URL for `{path}` (expected an absolute http or https URL)")]
    InvalidUrl { path: String, url: String },
    #[error("invalid sha256 for `{path}`: {value}")]
    InvalidSha256 { path: String, value: String },
    #[error("file size must be greater than zero: {0}")]
    ZeroSize(String),
    #[error("duplicate alias: {0}")]
    DuplicateAlias(String),
    #[error("duplicate model name: {0}")]
    DuplicateName(String),
    #[error("model name conflicts with an existing alias: {0}")]
    NameConflictsWithAlias(String),
    #[error("alias collides with another store entry after sanitization: {0}")]
    StoreNameCollision(String),
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
        // `name` becomes a directory under `models_dir` and `family` a path
        // component in an exported registry. Neither may escape those roots.
        if !is_safe_component(&self.name) {
            return Err(ManifestError::InvalidName(self.name.clone()));
        }
        if !is_safe_component(&self.family) {
            return Err(ManifestError::InvalidFamily(self.family.clone()));
        }
        if let Some(source) = &self.source
            && (source.is_empty() || source.contains('\0') || source.chars().any(char::is_control))
        {
            return Err(ManifestError::Empty("source"));
        }
        if let Some(revision) = &self.revision
            && (revision.is_empty()
                || revision.contains('\0')
                || revision.chars().any(char::is_control))
        {
            return Err(ManifestError::Empty("revision"));
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
            if !is_safe_relative_path(subfolder) {
                return Err(ManifestError::InvalidPath(subfolder.clone()));
            }
        }

        let mut paths = HashSet::new();
        for file in &self.files {
            if !is_safe_relative_path(&file.path) {
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
            if let Some(url) = &file.url
                && !is_valid_http_url(url)
            {
                return Err(ManifestError::InvalidUrl {
                    path: file.path.clone(),
                    url: url.clone(),
                });
            }
        }

        let mut aliases = HashSet::new();
        let mut store_components = HashSet::new();
        store_components.insert(store_component(&self.name));
        for alias in &self.aliases {
            if alias.trim().is_empty() {
                return Err(ManifestError::Empty("aliases"));
            }
            if !is_safe_alias(alias) {
                return Err(ManifestError::InvalidAlias(alias.clone()));
            }
            if alias == &self.name {
                return Err(ManifestError::AliasEqualsName(alias.clone()));
            }
            if !aliases.insert(alias) {
                return Err(ManifestError::DuplicateAlias(alias.clone()));
            }
            if !store_components.insert(store_component(alias)) {
                return Err(ManifestError::StoreNameCollision(alias.clone()));
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

/// The directory/file component used by the content store for a model name or
/// alias. Colons and other non-portable characters are mapped to `_`, so
/// validation must reject aliases that collide after this mapping.
pub fn store_component(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|character| {
            if character.is_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "_".to_owned()
    } else {
        sanitized
    }
}

/// Whether a manifest-style relative path is safe to join under a managed
/// directory: no absolute/Windows prefix, no `..`, no backslashes, no NUL or
/// control characters, no Windows-invalid characters, and no glob
/// metacharacters (which shaul would interpret in `include` patterns).
pub fn is_safe_relative_path(value: &str) -> bool {
    if value.is_empty() || value.len() > 4096 || value.ends_with('/') {
        return false;
    }
    // Reject characters that are unsafe on Windows, can hide an extension, or
    // have meaning to the glob matcher shaul uses for `include` patterns.
    if value.chars().any(|character| {
        character.is_control()
            || is_unsafe_format(character)
            || matches!(
                character,
                '\\' | '\0' | ':' | '*' | '?' | '[' | ']' | '{' | '}' | '<' | '>' | '"' | '|'
            )
    }) {
        return false;
    }
    let path = Path::new(value);
    if path.is_absolute() {
        return false;
    }
    let mut has_normal = false;
    for component in path.components() {
        match component {
            Component::Normal(name) => {
                let name = name.to_string_lossy();
                if name.is_empty()
                    || name.ends_with('.')
                    || name.ends_with(' ')
                    || is_windows_reserved(&name)
                {
                    return false;
                }
                has_normal = true;
            }
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return false,
        }
    }
    has_normal
}

/// Whether `value` can safely be used as one directory/file component under a
/// managed root. Used for model names and family directories.
fn is_safe_component(value: &str) -> bool {
    if value.is_empty() || value.len() > 256 || value.trim() != value {
        return false;
    }
    if matches!(value, "." | "..") || value.ends_with('.') || value.ends_with(' ') {
        return false;
    }
    if value.chars().any(|character| {
        character.is_control()
            || is_unsafe_format(character)
            || matches!(
                character,
                '/' | '\\' | '\0' | ':' | '*' | '?' | '[' | ']' | '{' | '}' | '<' | '>' | '"' | '|'
            )
    }) {
        return false;
    }
    !is_windows_reserved(value)
}

/// Aliases are catalog keys, not paths; they may contain `:` (for example
/// `laya:latest`) but must not be able to traverse out of a store directory.
fn is_safe_alias(value: &str) -> bool {
    if value.is_empty() || value.len() > 512 || value.trim() != value {
        return false;
    }
    if matches!(value, "." | "..") {
        return false;
    }
    !value.chars().any(|character| {
        character.is_control()
            || is_unsafe_format(character)
            || matches!(character, '/' | '\\' | '\0')
    })
}

fn is_valid_http_url(value: &str) -> bool {
    if value.is_empty() || value.chars().any(char::is_control) {
        return false;
    }
    match value.split_once("://") {
        Some((scheme, rest)) => {
            !rest.is_empty()
                && (scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https"))
        }
        None => false,
    }
}

/// Unicode format characters that can hide or spoof a file name (bidi
/// overrides, zero-width characters, and similar). Mirrors shaul's path
/// validation so a manifest accepted here cannot be rejected later by the
/// download engine.
fn is_unsafe_format(character: char) -> bool {
    matches!(
        character,
        '\u{00AD}'
            | '\u{061C}'
            | '\u{180E}'
            | '\u{200B}'..='\u{200F}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2060}'..='\u{2064}'
            | '\u{2066}'..='\u{206F}'
            | '\u{FEFF}'
            | '\u{FFF9}'..='\u{FFFB}'
            | '\u{E0000}'..='\u{E007F}'
    )
}

fn is_windows_reserved(value: &str) -> bool {
    let stem = value
        .split('.')
        .next()
        .unwrap_or(value)
        .trim_end_matches(' ')
        .to_ascii_uppercase();
    matches!(
        stem.as_str(),
        "CON"
            | "PRN"
            | "AUX"
            | "NUL"
            | "COM1"
            | "COM2"
            | "COM3"
            | "COM4"
            | "COM5"
            | "COM6"
            | "COM7"
            | "COM8"
            | "COM9"
            | "LPT1"
            | "LPT2"
            | "LPT3"
            | "LPT4"
            | "LPT5"
            | "LPT6"
            | "LPT7"
            | "LPT8"
            | "LPT9"
    )
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

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> RepoManifest {
        RepoManifest {
            schema_version: REPO_MANIFEST_SCHEMA,
            name: "tiny".to_owned(),
            aliases: vec!["tiny:latest".to_owned()],
            family: "tiny".to_owned(),
            engine: "candle".to_owned(),
            arch: "custom".to_owned(),
            format: "safetensors".to_owned(),
            layout: "tiny-v1".to_owned(),
            subfolder: None,
            files: vec![ManifestFile {
                path: "model.safetensors".to_owned(),
                role: FileRole::Weights,
                sha256: None,
                size: Some(1),
                url: None,
            }],
            calibration_mode: Default::default(),
            capabilities: Default::default(),
            router: None,
            license: "Apache-2.0".to_owned(),
            description: "tiny".to_owned(),
            release_date: None,
            source: None,
            revision: None,
        }
    }

    #[test]
    fn rejects_names_that_escape_the_model_directory() {
        for name in ["../escape", "a/b", "a\\b", "a:b", "CON", ".", "..", " x"] {
            let mut manifest = manifest();
            manifest.name = name.to_owned();
            assert!(
                manifest.validate().is_err(),
                "unsafe model name accepted: {name:?}"
            );
        }
        let mut manifest = manifest();
        manifest.family = "../escape".to_owned();
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn rejects_unsafe_aliases_but_allows_version_tags() {
        let mut manifest = manifest();
        manifest.aliases = vec!["../escape".to_owned()];
        assert!(manifest.validate().is_err());
        manifest.aliases = vec!["laya:latest".to_owned(), "laya-typed".to_owned()];
        manifest.validate().expect("version aliases are valid");
    }

    #[test]
    fn rejects_aliases_that_collide_in_the_store() {
        let mut manifest = manifest();
        manifest.name = "a_b".to_owned();
        manifest.aliases = vec!["a:b".to_owned()];
        assert!(manifest.validate().is_err());

        manifest.name = "laya".to_owned();
        manifest.aliases = vec!["laya:latest".to_owned()];
        manifest.validate().expect("distinct store components");
    }

    #[test]
    fn rejects_unsafe_file_paths_and_globs() {
        for path in [
            "../escape.bin",
            "/absolute.bin",
            "a\\b.bin",
            "a:b.bin",
            "a*b.bin",
            "a?b.bin",
            "a[b].bin",
            "a{b}.bin",
            "a\u{202E}b.bin",
            "C:/Windows/System32/x.dll",
        ] {
            let mut manifest = manifest();
            manifest.files[0].path = path.to_owned();
            assert!(
                manifest.validate().is_err(),
                "unsafe file path accepted: {path:?}"
            );
        }
        let mut manifest = manifest();
        manifest.files[0].path = "tokenizer/tokenizer.json".to_owned();
        manifest.validate().expect("nested path is valid");
    }

    #[test]
    fn only_http_urls_are_accepted_for_layers() {
        for url in [
            "file:///etc/passwd",
            "data:text/plain,hello",
            "",
            "ftp://example.com/x",
        ] {
            let mut manifest = manifest();
            manifest.files[0].url = Some(url.to_owned());
            assert!(
                manifest.validate().is_err(),
                "unsafe layer URL accepted: {url:?}"
            );
        }
        let mut manifest = manifest();
        manifest.files[0].url = Some("https://example.com/model.bin?token=abc".to_owned());
        manifest.validate().expect("https URL is valid");
    }
}
