//! The pluggable manifest-registry boundary.
//!
//! A registry answers "which manifests exist?"; it does not own blobs. The
//! default filesystem registry lives in the `olaya` facade.

use std::path::PathBuf;

use thiserror::Error;

use crate::manifest::{ManifestError, RepoManifest};

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("reading registry directory {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parsing {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid manifest {path}: {error}")]
    Manifest {
        path: PathBuf,
        #[source]
        error: ManifestError,
    },
    #[error("no model manifest (*.json) found under {dir}")]
    Empty { dir: PathBuf },
    #[error(
        "manifest {path} is named {stem:?}, but the manifest's name is {name:?} and no alias matches"
    )]
    NameMismatch {
        path: PathBuf,
        stem: String,
        name: String,
    },
    #[error("manifest registry unavailable: {0}")]
    Unavailable(String),
}

/// A pluggable source of model manifests.
pub trait ManifestRegistry: Send + Sync {
    fn manifests(&self) -> Result<Vec<RepoManifest>, RegistryError>;
}
