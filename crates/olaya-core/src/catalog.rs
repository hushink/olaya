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
        let mut aliases = HashSet::new();
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
}
