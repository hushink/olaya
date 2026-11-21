//! Materialize a catalog into the filesystem registry layout.

use std::path::{Path, PathBuf};

use olaya_core::{RegistryError, RepoManifest};
use serde::Serialize;

/// Result of [`export_registry`].
#[derive(Clone, Debug, Serialize)]
pub struct ExportReport {
    /// Directory that received the manifests.
    pub dir: PathBuf,
    /// Files created or overwritten.
    pub written: usize,
    /// Existing files that already described the same manifest.
    pub unchanged: usize,
}

/// Write `manifests` under `dir` as
/// `v1/library/<family>/manifests/<name>.json`.
///
/// Existing files are compared by parsed manifest, not by formatting. A file
/// that describes a different manifest is an error unless `force` is set;
/// identical files are left untouched so repeated exports are idempotent.
pub fn export_registry(
    manifests: &[RepoManifest],
    dir: impl AsRef<Path>,
    force: bool,
) -> Result<ExportReport, RegistryError> {
    let dir = dir.as_ref();
    let mut pending: Vec<(PathBuf, Vec<u8>)> = Vec::new();
    let mut conflicts = Vec::new();
    let mut unchanged = 0usize;
    for manifest in manifests {
        let target = dir.join(manifest_path(manifest)?);
        match std::fs::read(&target) {
            Ok(existing) => match serde_json::from_slice::<RepoManifest>(&existing) {
                Ok(existing) if existing == *manifest => unchanged += 1,
                _ if force => pending.push((target, serialized(manifest)?)),
                _ => conflicts.push(target),
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                pending.push((target, serialized(manifest)?));
            }
            Err(source) => {
                return Err(RegistryError::Io {
                    path: target,
                    source,
                });
            }
        }
    }
    if !force && !conflicts.is_empty() {
        let first = conflicts
            .first()
            .map(|path| path.display().to_string())
            .unwrap_or_default();
        return Err(RegistryError::Unavailable(format!(
            "{} manifest(s) already exist with different contents under {} \
             (first: {first}); pass --force to overwrite",
            conflicts.len(),
            dir.display()
        )));
    }
    for (target, bytes) in &pending {
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|source| RegistryError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        std::fs::write(target, bytes).map_err(|source| RegistryError::Io {
            path: target.clone(),
            source,
        })?;
    }
    Ok(ExportReport {
        dir: dir.to_path_buf(),
        written: pending.len(),
        unchanged,
    })
}

fn serialized(manifest: &RepoManifest) -> Result<Vec<u8>, RegistryError> {
    manifest
        .validate()
        .map_err(|error| RegistryError::Manifest {
            path: PathBuf::from(&manifest.name),
            error,
        })?;
    let mut text = serde_json::to_string_pretty(manifest).map_err(|error| {
        RegistryError::Unavailable(format!("serializing {}: {error}", manifest.name))
    })?;
    text.push('\n');
    Ok(text.into_bytes())
}

fn manifest_path(manifest: &RepoManifest) -> Result<PathBuf, RegistryError> {
    if !safe_component(&manifest.family) || !safe_component(&manifest.name) {
        return Err(RegistryError::Unavailable(format!(
            "manifest name/family is not a safe path component: {:?}/{:?}",
            manifest.family, manifest.name
        )));
    }
    Ok(Path::new("v1")
        .join("library")
        .join(&manifest.family)
        .join("manifests")
        .join(format!("{}.json", manifest.name)))
}

fn safe_component(value: &str) -> bool {
    !value.is_empty() && value != "." && value != ".." && !value.contains(['/', '\\', '\0'])
}

#[cfg(test)]
mod tests {
    use super::*;
    use olaya_core::{Capabilities, FileRole, ManifestFile, REPO_MANIFEST_SCHEMA};

    fn manifest() -> RepoManifest {
        RepoManifest {
            schema_version: REPO_MANIFEST_SCHEMA,
            name: "mock".to_owned(),
            aliases: vec!["mock:latest".to_owned()],
            family: "mock".to_owned(),
            engine: "candle".to_owned(),
            arch: "mock".to_owned(),
            format: "safetensors".to_owned(),
            layout: "mock-v1".to_owned(),
            subfolder: None,
            files: vec![ManifestFile {
                path: "model.safetensors".to_owned(),
                role: FileRole::Weights,
                sha256: Some("a".repeat(64)),
                size: Some(1),
                url: None,
            }],
            calibration_mode: Default::default(),
            capabilities: Capabilities::default(),
            router: None,
            license: "Apache-2.0".to_owned(),
            description: "mock".to_owned(),
            release_date: None,
            source: None,
            revision: None,
        }
    }

    fn temp_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "olaya-export-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or_default()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("temp dir");
        path
    }

    #[test]
    fn export_is_idempotent_and_detects_conflicts() {
        let dir = temp_dir();
        let manifests = vec![manifest()];
        let first = export_registry(&manifests, &dir, false).expect("export");
        assert_eq!(first.written, 1);
        let path = dir.join("v1/library/mock/manifests/mock.json");
        assert!(path.is_file());

        let second = export_registry(&manifests, &dir, false).expect("second export");
        assert_eq!(second.written, 0);
        assert_eq!(second.unchanged, 1);

        let mut changed = manifest();
        changed.description = "changed".to_owned();
        assert!(export_registry(&[changed.clone()], &dir, false).is_err());
        let forced = export_registry(&[changed], &dir, true).expect("force export");
        assert_eq!(forced.written, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unsafe_components_are_rejected() {
        let mut bad = manifest();
        bad.name = "../escape".to_owned();
        assert!(export_registry(&[bad], temp_dir(), false).is_err());
    }
}
