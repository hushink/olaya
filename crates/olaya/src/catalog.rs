use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use olaya_core::{ManifestError, ModelMetadata, RepoManifest};

/// The runtime model catalog: canonical manifests plus aliases.
#[derive(Debug, Default)]
pub struct Catalog {
    manifests: IndexMap<String, RepoManifest>,
    aliases: HashMap<String, String>,
}

impl Catalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, manifest: RepoManifest) -> Result<(), ManifestError> {
        manifest.validate()?;
        if self.manifests.contains_key(&manifest.name) {
            return Err(ManifestError::DuplicateAlias(manifest.name.clone()));
        }
        for alias in &manifest.aliases {
            if self.manifests.contains_key(alias) {
                return Err(ManifestError::DuplicateAlias(alias.clone()));
            }
            if let Some(existing) = self.aliases.get(alias) {
                return Err(ManifestError::DuplicateAlias(format!(
                    "{alias} (already points at {existing})"
                )));
            }
        }
        for alias in &manifest.aliases {
            self.aliases.insert(alias.clone(), manifest.name.clone());
        }
        self.manifests.insert(manifest.name.clone(), manifest);
        Ok(())
    }

    /// Insert or replace an entry with the same canonical name.
    pub fn upsert(&mut self, manifest: RepoManifest) -> Result<(), ManifestError> {
        let name = manifest.name.clone();
        self.remove(&name);
        self.insert(manifest)
    }

    pub fn resolve(&self, name: &str) -> Option<(String, RepoManifest)> {
        if let Some(manifest) = self.manifests.get(name) {
            return Some((manifest.name.clone(), manifest.clone()));
        }
        let canonical = self.aliases.get(name)?;
        let manifest = self.manifests.get(canonical)?;
        Some((manifest.name.clone(), manifest.clone()))
    }

    pub fn remove(&mut self, name: &str) -> Option<String> {
        let (canonical, _) = self.resolve(name)?;
        self.manifests.shift_remove(&canonical);
        self.aliases.retain(|_, target| target != &canonical);
        Some(canonical)
    }

    pub fn manifests(&self) -> Vec<RepoManifest> {
        self.manifests.values().cloned().collect()
    }

    pub fn list(&self) -> Vec<ModelMetadata> {
        let mut models = Vec::new();
        for manifest in self.manifests.values() {
            models.push(manifest.metadata());
            models.extend(manifest.alias_metadata());
        }
        models
    }

    pub fn is_empty(&self) -> bool {
        self.manifests.is_empty()
    }

    pub fn len(&self) -> usize {
        self.manifests.len()
    }

    pub fn aliases(&self) -> &HashMap<String, String> {
        &self.aliases
    }

    /// Validate a batch and insert it, rolling back on the first failure.
    pub fn insert_all(
        &mut self,
        manifests: impl IntoIterator<Item = RepoManifest>,
    ) -> Result<(), ManifestError> {
        let mut seen_names = HashSet::new();
        let mut seen_aliases = HashSet::new();
        for manifest in manifests {
            manifest.validate()?;
            if !seen_names.insert(manifest.name.clone()) {
                return Err(ManifestError::DuplicateAlias(manifest.name));
            }
            for alias in &manifest.aliases {
                if !seen_aliases.insert(alias.clone()) {
                    return Err(ManifestError::DuplicateAlias(alias.clone()));
                }
            }
        }
        Ok(())
    }
}
