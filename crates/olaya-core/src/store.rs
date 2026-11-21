//! The pluggable manifest store boundary.
//!
//! Engine-agnostic by design: a store implementation only needs the manifest
//! types and the standard library, so it can live in its own crate without
//! depending on candle or HTTP.

use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::manifest::{
    ANNOTATION_PATH, ANNOTATION_ROLE, FileRole, ManifestError, ManifestFile, ModelConfig,
    OciManifest, REPO_MANIFEST_SCHEMA, RepoManifest,
};

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("store I/O at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("store JSON at {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid manifest: {0}")]
    Manifest(#[from] ManifestError),
    #[error("invalid digest: {0}")]
    Digest(String),
    #[error("digest mismatch for {path}: expected {expected}, got {actual}")]
    DigestMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("size mismatch for {path}: expected {expected}, got {actual}")]
    SizeMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    #[error("missing blob: {0}")]
    MissingBlob(String),
    #[error("invalid model reference: {0}")]
    Reference(String),
    #[error("layer {0} has no `org.olaya.path` annotation")]
    MissingPath(String),
    #[error("invalid relative path: {0}")]
    InvalidPath(String),
    #[error("manifest store unavailable: {0}")]
    Unavailable(String),
}

/// A stored model: its OCI manifest, config blob, digest, and manifest path.
#[derive(Clone, Debug)]
pub struct StoredModel {
    pub manifest: OciManifest,
    pub config: ModelConfig,
    /// `sha256:<hex>` of the manifest bytes.
    pub digest: String,
    pub manifest_path: PathBuf,
}

impl StoredModel {
    /// Reconstruct the repository manifest from the OCI manifest and config.
    ///
    /// This makes a stored model self-describing: a registry consumer can take
    /// the OCI manifest + blobs and recover the same `RepoManifest` the model
    /// author published.
    pub fn to_repo_manifest(&self) -> RepoManifest {
        let name = if self.config.name.is_empty() {
            self.manifest_path
                .parent()
                .and_then(|path| path.parent())
                .and_then(|path| path.file_name())
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default()
        } else {
            self.config.name.clone()
        };
        let files = self
            .manifest
            .layers
            .iter()
            .filter_map(|layer| {
                let path = layer.annotations.get(ANNOTATION_PATH)?.clone();
                let role = layer
                    .annotations
                    .get(ANNOTATION_ROLE)
                    .and_then(|name| FileRole::from_name(name))
                    .unwrap_or(FileRole::Other);
                let sha256 = layer.digest.strip_prefix("sha256:").map(str::to_owned);
                Some(ManifestFile {
                    path,
                    role,
                    sha256,
                    size: Some(layer.size),
                    url: layer.urls.first().cloned(),
                })
            })
            .collect();
        RepoManifest {
            schema_version: REPO_MANIFEST_SCHEMA,
            name,
            aliases: self.config.aliases.clone(),
            family: self.config.family.clone(),
            engine: self
                .config
                .engine
                .clone()
                .unwrap_or_else(|| "candle".to_owned()),
            arch: self.config.arch.clone(),
            format: self.config.model_format.clone(),
            layout: self.config.layout.clone().unwrap_or_default(),
            subfolder: self.config.subfolder.clone(),
            files,
            calibration_mode: self.config.calibration_mode,
            capabilities: self.config.capabilities.clone().unwrap_or_default(),
            router: self.config.router.clone(),
            license: self.config.license.clone(),
            description: self.config.description.clone(),
            release_date: self.config.release_date.clone(),
            source: Some(self.config.source.clone()).filter(|source| !source.is_empty()),
            revision: self.config.revision.clone(),
        }
    }
}

/// A pluggable manifest/blob store.
///
/// The default filesystem implementation lives in the `olaya` facade. Custom
/// implementations (database, object storage, remote OCI registry) depend only
/// on this crate and are injected through the engine builder.
pub trait ManifestStore: Send + Sync {
    /// List the stored models. Implementations may return canonical names,
    /// aliases, or both; callers deduplicate by manifest name.
    fn list(&self) -> Result<Vec<String>, StoreError>;
    /// Resolve `name`, an alias, or `name@sha256:<hex>`.
    fn get(&self, reference: &str) -> Result<StoredModel, StoreError>;
    /// Store a repository manifest and its materialized files.
    fn put(&self, manifest: &RepoManifest, model_dir: &Path) -> Result<StoredModel, StoreError>;
    /// Copy the stored files into `destination`.
    fn materialize(&self, model: &StoredModel, destination: &Path) -> Result<(), StoreError>;
    /// Verify the stored manifest's blobs.
    fn verify(&self, model: &StoredModel) -> Result<(), StoreError>;
}
