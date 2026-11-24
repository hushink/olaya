//! The distributable catalog bundle fetched by [`crate::ManifestRegistry`]
//! implementations such as the remote registry.

use std::collections::HashSet;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::manifest::RepoManifest;
use crate::registry::RegistryError;

/// Current `registry-catalog.json` schema.
pub const CATALOG_SCHEMA_VERSION: u32 = 1;

/// A catalog snapshot: every manifest the registry knows, in one JSON file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CatalogBundle {
    pub schema_version: u32,
    pub manifests: Vec<RepoManifest>,
}

impl CatalogBundle {
    pub fn new(mut manifests: Vec<RepoManifest>) -> Self {
        manifests.sort_by(|left, right| left.name.cmp(&right.name));
        Self {
            schema_version: CATALOG_SCHEMA_VERSION,
            manifests,
        }
    }

    pub fn validate(&self) -> Result<(), RegistryError> {
        if self.schema_version != CATALOG_SCHEMA_VERSION {
            return Err(RegistryError::Unavailable(format!(
                "unsupported catalog schema_version {} (expected {CATALOG_SCHEMA_VERSION})",
                self.schema_version
            )));
        }
        if self.manifests.is_empty() {
            return Err(RegistryError::Unavailable(
                "catalog contains no manifests".to_owned(),
            ));
        }
        let mut names = HashSet::with_capacity(self.manifests.len());
        for manifest in &self.manifests {
            manifest
                .validate()
                .map_err(|error| RegistryError::Manifest {
                    path: PathBuf::from(&manifest.name),
                    error,
                })?;
            if !names.insert(manifest.name.as_str()) {
                return Err(RegistryError::Unavailable(format!(
                    "duplicate catalog model name {:?}",
                    manifest.name
                )));
            }
        }
        // Check aliases against every canonical name, not only the names seen
        // so far: an alias must never shadow a model defined later in the
        // bundle.
        let mut aliases = HashSet::new();
        for manifest in &self.manifests {
            for alias in &manifest.aliases {
                if names.contains(alias.as_str()) || !aliases.insert(alias.as_str()) {
                    return Err(RegistryError::Unavailable(format!(
                        "duplicate catalog alias {alias:?}"
                    )));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_bundles_are_rejected() {
        let bundle = CatalogBundle {
            schema_version: CATALOG_SCHEMA_VERSION,
            manifests: Vec::new(),
        };
        assert!(bundle.validate().is_err());
    }

    #[test]
    fn unsupported_schema_is_rejected() {
        let bundle = CatalogBundle {
            schema_version: CATALOG_SCHEMA_VERSION + 1,
            manifests: Vec::new(),
        };
        assert!(bundle.validate().is_err());
    }

    fn manifest(name: &str, aliases: &[&str]) -> RepoManifest {
        RepoManifest {
            schema_version: crate::manifest::REPO_MANIFEST_SCHEMA,
            name: name.to_owned(),
            aliases: aliases.iter().map(|alias| (*alias).to_owned()).collect(),
            family: "tiny".to_owned(),
            engine: "candle".to_owned(),
            arch: "custom".to_owned(),
            format: "safetensors".to_owned(),
            layout: "tiny-v1".to_owned(),
            subfolder: None,
            files: vec![crate::manifest::ManifestFile {
                path: "model.safetensors".to_owned(),
                role: crate::manifest::FileRole::Weights,
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
    fn aliases_must_not_shadow_a_later_model_name() {
        let bundle = CatalogBundle {
            schema_version: CATALOG_SCHEMA_VERSION,
            manifests: vec![manifest("first", &["second"]), manifest("second", &[])],
        };
        assert!(bundle.validate().is_err());
    }
}
