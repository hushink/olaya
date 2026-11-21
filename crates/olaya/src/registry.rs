//! Filesystem registry: load `*.json` model manifests from a directory tree.
//!
//! The repository root's `registry/` directory is the development catalog. A
//! compiled binary can point at any directory with `--registry-dir` or
//! `OLAYA_REGISTRY_DIR`.
//!
//! A manifest file is named after the model name or one of its aliases:
//! `registry/v1/library/laya/manifests/laya.json`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use olaya_core::{ManifestRegistry, ManifestStore, RegistryError, RepoManifest};

/// Environment variable that overrides the registry directory.
pub const REGISTRY_ENV: &str = "OLAYA_REGISTRY_DIR";

/// The default filesystem registry.
///
/// [`ManifestRegistry`] is the extension point; this type only walks a
/// directory tree.
#[derive(Clone, Debug)]
pub struct FsRegistry {
    dir: PathBuf,
}

impl FsRegistry {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

impl ManifestRegistry for FsRegistry {
    fn manifests(&self) -> Result<Vec<RepoManifest>, RegistryError> {
        load_manifests(&self.dir)
    }
}

/// A registry view over a [`ManifestStore`].
///
/// Every stored OCI manifest is converted back into a [`RepoManifest`] via
/// [`olaya_core::StoredModel::to_repo_manifest`]. This lets an engine run
/// entirely from a distributed content store, with no filesystem registry.
pub struct StoreRegistry {
    store: Arc<dyn ManifestStore>,
}

impl StoreRegistry {
    pub fn new(store: Arc<dyn ManifestStore>) -> Self {
        Self { store }
    }

    pub fn store(&self) -> &Arc<dyn ManifestStore> {
        &self.store
    }
}

impl ManifestRegistry for StoreRegistry {
    fn manifests(&self) -> Result<Vec<RepoManifest>, RegistryError> {
        let names = self
            .store
            .list()
            .map_err(|error| RegistryError::Unavailable(error.to_string()))?;
        let mut manifests: Vec<RepoManifest> = Vec::new();
        for name in names {
            let stored = match self.store.get(&name) {
                Ok(stored) => stored,
                Err(error) => {
                    tracing::warn!(%name, %error, "skipping unreadable stored manifest");
                    continue;
                }
            };
            let manifest = stored.to_repo_manifest();
            if let Err(error) = manifest.validate() {
                tracing::warn!(%name, %error, "skipping invalid stored manifest");
                continue;
            }
            if manifests
                .iter()
                .any(|existing| existing.name == manifest.name)
            {
                continue;
            }
            manifests.push(manifest);
        }
        manifests.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(manifests)
    }
}

/// `$OLAYA_REGISTRY_DIR`, else `./registry` when it exists, else
/// `~/.olaya/registry`.
pub fn default_registry_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os(REGISTRY_ENV).filter(|value| !value.is_empty()) {
        return PathBuf::from(dir);
    }
    let local = PathBuf::from("registry");
    if local.is_dir() {
        return local;
    }
    crate::hub::default_root().join("registry")
}

/// Load every `*.json` manifest under `dir`, recursively.
///
/// Each file's stem must equal the manifest's `name` or one of its `aliases`.
/// Manifests are validated; the result is sorted by canonical model name.
pub fn load_manifests(dir: impl AsRef<Path>) -> Result<Vec<RepoManifest>, RegistryError> {
    let dir = dir.as_ref();
    let mut manifests = Vec::new();
    walk(dir, &mut manifests)?;
    if manifests.is_empty() {
        return Err(RegistryError::Empty {
            dir: dir.to_path_buf(),
        });
    }
    manifests.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(manifests)
}

fn walk(dir: &Path, out: &mut Vec<RepoManifest>) -> Result<(), RegistryError> {
    let entries = std::fs::read_dir(dir).map_err(|source| RegistryError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    let mut entries =
        entries
            .collect::<Result<Vec<_>, _>>()
            .map_err(|source| RegistryError::Io {
                path: dir.to_path_buf(),
                source,
            })?;
    entries.sort_by_key(|entry| entry.file_name());

    for entry in entries {
        let path = entry.path();
        let file_type = entry.file_type().map_err(|source| RegistryError::Io {
            path: path.clone(),
            source,
        })?;
        if file_type.is_dir() {
            walk(&path, out)?;
        } else if path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
        {
            out.push(read_manifest(&path)?);
        }
    }
    Ok(())
}

fn read_manifest(path: &Path) -> Result<RepoManifest, RegistryError> {
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default()
        .to_owned();
    let bytes = std::fs::read(path).map_err(|source| RegistryError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let manifest: RepoManifest =
        serde_json::from_slice(&bytes).map_err(|source| RegistryError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    manifest
        .validate()
        .map_err(|error| RegistryError::Manifest {
            path: path.to_path_buf(),
            error,
        })?;
    if manifest.name != stem && !manifest.aliases.iter().any(|alias| alias == &stem) {
        return Err(RegistryError::NameMismatch {
            path: path.to_path_buf(),
            stem,
            name: manifest.name,
        });
    }
    Ok(manifest)
}
