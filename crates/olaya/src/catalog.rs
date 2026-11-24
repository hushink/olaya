use std::collections::HashMap;

use indexmap::IndexMap;
use olaya_core::{ManifestError, ModelMetadata, RepoManifest};

/// The runtime model catalog: canonical manifests plus aliases.
#[derive(Clone, Debug, Default)]
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
            return Err(ManifestError::DuplicateName(manifest.name.clone()));
        }
        // A canonical name must not shadow an existing alias, and an alias
        // must not shadow an existing canonical name. Otherwise `resolve`
        // silently makes one of them unreachable.
        if self.aliases.contains_key(&manifest.name) {
            return Err(ManifestError::NameConflictsWithAlias(manifest.name.clone()));
        }
        for alias in &manifest.aliases {
            if self.manifests.contains_key(alias) {
                return Err(ManifestError::NameConflictsWithAlias(alias.clone()));
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
    ///
    /// Validation and insertion are applied to a copy first, so a failed
    /// replace leaves the catalog unchanged.
    pub fn upsert(&mut self, manifest: RepoManifest) -> Result<(), ManifestError> {
        let name = manifest.name.clone();
        let mut next = self.clone();
        next.remove(&name);
        next.insert(manifest)?;
        *self = next;
        Ok(())
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

    /// Validate a batch and insert it, leaving the catalog unchanged on the
    /// first failure.
    pub fn insert_all(
        &mut self,
        manifests: impl IntoIterator<Item = RepoManifest>,
    ) -> Result<(), ManifestError> {
        let mut next = self.clone();
        for manifest in manifests {
            next.insert(manifest)?;
        }
        *self = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use olaya_core::{CalibrationMode, Capabilities, FileRole, ManifestFile, REPO_MANIFEST_SCHEMA};

    fn manifest(name: &str, aliases: &[&str]) -> RepoManifest {
        RepoManifest {
            schema_version: REPO_MANIFEST_SCHEMA,
            name: name.to_owned(),
            aliases: aliases.iter().map(|alias| (*alias).to_owned()).collect(),
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
            calibration_mode: CalibrationMode::Raw,
            capabilities: Capabilities::default(),
            router: None,
            license: "Apache-2.0".to_owned(),
            description: "tiny".to_owned(),
            release_date: None,
            source: None,
            revision: None,
        }
    }

    #[test]
    fn names_and_aliases_cannot_shadow_each_other() {
        let mut catalog = Catalog::new();
        catalog
            .insert(manifest("first", &["second"]))
            .expect("first");
        assert!(catalog.insert(manifest("second", &[])).is_err());

        let mut catalog = Catalog::new();
        catalog.insert(manifest("second", &[])).expect("second");
        assert!(catalog.insert(manifest("first", &["second"])).is_err());
    }

    #[test]
    fn batch_insert_is_atomic() {
        let mut catalog = Catalog::new();
        catalog.insert(manifest("existing", &[])).expect("existing");
        let batch = vec![manifest("a", &[]), manifest("existing", &[])];
        assert!(catalog.insert_all(batch).is_err());
        assert_eq!(catalog.len(), 1);
        assert!(catalog.resolve("a").is_none());
    }
}
