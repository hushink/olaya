//! Content-addressed OCI store.
//!
//! Layout:
//!
//! ```text
//! <root>/v1/library/<model>/manifests/<name>.json
//! <root>/v1/library/<model>/manifests/sha256-<hex>.json
//! <root>/blobs/sha256-<hex>
//! ```
//!
//! `Store::import` converts a [`RepoManifest`] + materialized files into an
//! [`OciManifest`] with one descriptor per file. `Store::materialize` restores
//! the files from the blobs. `Store::manifest("name@sha256:<hex>")` pins a
//! manifest by digest.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use olaya_core::{
    ANNOTATION_PATH, ANNOTATION_ROLE, Descriptor, MEDIA_CONFIG, MEDIA_TYPE_MANIFEST, ManifestStore,
    ModelConfig, OciManifest, RepoManifest, StoreError, StoredModel, is_safe_relative_path,
    store_component,
};
use sha2::{Digest, Sha256};

pub const BLOBS_DIR: &str = "blobs";
pub const MANIFEST_ROOT: &str = "v1/library";
pub const OCI_SCHEMA_VERSION: u32 = 2;
/// Maximum size of the JSON config blob read from the store.
const MAX_CONFIG_BYTES: u64 = 16 * 1024 * 1024;
/// Maximum size of an OCI manifest JSON document.
const MAX_OCI_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;

/// A local content-addressed OCI store implementing [`ManifestStore`].
#[derive(Clone, Debug)]
pub struct Store {
    root: PathBuf,
}

/// Summary of [`Store::remove`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StoreRemoval {
    /// Canonical model name that was removed.
    pub model: String,
    /// Manifest JSON files deleted (named and digest-pinned, including aliases).
    pub manifests: usize,
    /// Content blobs deleted because no remaining manifest referenced them.
    pub blobs: usize,
    /// Bytes reclaimed from manifest files and blobs.
    pub freed_bytes: u64,
    /// True when blob garbage collection was skipped because a remaining
    /// manifest could not be read.
    pub gc_skipped: bool,
}

impl Store {
    /// Open (or lazily create) a store rooted at `root`.
    pub fn open(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn blob_path(&self, digest: &str) -> Result<PathBuf, StoreError> {
        let hex = digest_hex(digest)?;
        Ok(self.root.join(BLOBS_DIR).join(format!("sha256-{hex}")))
    }

    pub fn has_blob(&self, digest: &str, size: u64) -> bool {
        self.blob_path(digest)
            .ok()
            .and_then(|path| std::fs::metadata(path).ok())
            .is_some_and(|metadata| metadata.is_file() && metadata.len() == size)
    }

    pub fn read_blob(&self, digest: &str) -> Result<Vec<u8>, StoreError> {
        let path = self.blob_path(digest)?;
        match std::fs::read(&path) {
            Ok(bytes) => Ok(bytes),
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
                Err(StoreError::MissingBlob(digest.to_owned()))
            }
            Err(source) => Err(io(&path, source)),
        }
    }

    /// Convert a `RepoManifest` and its materialized files into an
    /// `OciManifest`, storing every blob and the manifest by name and digest.
    pub fn import(
        &self,
        manifest: &RepoManifest,
        model_dir: &Path,
    ) -> Result<StoredModel, StoreError> {
        manifest.validate()?;
        self.ensure_layout()?;
        // `store_component` maps different names to the same directory (for
        // example `a:b` and `a_b`). Refuse to overwrite a slot that belongs to
        // a different model before writing any blobs.
        self.ensure_manifest_slot(&manifest.name, &manifest.name)?;
        for alias in &manifest.aliases {
            self.ensure_manifest_slot(alias, &manifest.name)?;
        }
        let base = match &manifest.subfolder {
            Some(subfolder) => model_dir.join(subfolder),
            None => model_dir.to_path_buf(),
        };

        let mut layers = Vec::with_capacity(manifest.files.len());
        for file in &manifest.files {
            let source = base.join(&file.path);
            let (digest, size) = self.put_file(&source)?;
            if let Some(expected) = &file.sha256 {
                let expected = format!("sha256:{}", expected.to_ascii_lowercase());
                if expected != digest {
                    return Err(StoreError::DigestMismatch {
                        path: source,
                        expected,
                        actual: digest,
                    });
                }
            }
            if let Some(expected) = file.size
                && expected != size
            {
                return Err(StoreError::SizeMismatch {
                    path: source,
                    expected,
                    actual: size,
                });
            }

            let mut annotations = BTreeMap::new();
            annotations.insert(ANNOTATION_PATH.to_owned(), file.path.clone());
            annotations.insert(ANNOTATION_ROLE.to_owned(), file.role.name().to_owned());
            let urls = file.url.clone().map(|url| vec![url]).unwrap_or_default();
            layers.push(Descriptor {
                media_type: file.role.media_type().to_owned(),
                digest,
                size,
                urls,
                annotations,
            });
        }

        let model_config = ModelConfig {
            name: manifest.name.clone(),
            aliases: manifest.aliases.clone(),
            model_format: manifest.format.clone(),
            family: manifest.family.clone(),
            arch: manifest.arch.clone(),
            parameter_size: String::new(),
            context_length: 0,
            languages: Vec::new(),
            description: manifest.description.clone(),
            source: manifest.source.clone().unwrap_or_default(),
            revision: manifest.revision.clone(),
            license: manifest.license.clone(),
            release_date: manifest.release_date.clone(),
            engine: Some(manifest.engine.clone()),
            layout: Some(manifest.layout.clone()),
            calibration_mode: manifest.calibration_mode,
            subfolder: manifest.subfolder.clone(),
            capabilities: Some(manifest.capabilities.clone()),
            router: manifest.router.clone(),
        };
        let config_bytes =
            serde_json::to_vec(&model_config).map_err(|source| StoreError::Json {
                path: PathBuf::from("<config>"),
                source,
            })?;
        let config_descriptor = self.store_bytes(&config_bytes, MEDIA_CONFIG)?;

        let oci = OciManifest {
            schema_version: OCI_SCHEMA_VERSION,
            media_type: MEDIA_TYPE_MANIFEST.to_owned(),
            config: config_descriptor,
            layers,
        };
        let bytes = serde_json::to_vec(&oci).map_err(|source| StoreError::Json {
            path: PathBuf::from("<manifest>"),
            source,
        })?;
        let digest = format!("sha256:{}", sha256_hex(&bytes));

        let digest_hex = digest_hex(&digest)?;
        let dir = self.manifest_dir(&manifest.name);
        std::fs::create_dir_all(&dir).map_err(|source| io(&dir, source))?;
        let manifest_path = dir.join(format!("{}.json", store_name(&manifest.name)));
        write_atomic(&manifest_path, &bytes)?;
        write_atomic(&dir.join(format!("sha256-{digest_hex}.json")), &bytes)?;

        for alias in &manifest.aliases {
            let alias_dir = self.manifest_dir(alias);
            std::fs::create_dir_all(&alias_dir).map_err(|source| io(&alias_dir, source))?;
            write_atomic(
                &alias_dir.join(format!("{}.json", store_name(alias))),
                &bytes,
            )?;
            write_atomic(&alias_dir.join(format!("sha256-{digest_hex}.json")), &bytes)?;
        }

        Ok(StoredModel {
            manifest: oci,
            config: model_config,
            digest,
            manifest_path,
        })
    }

    /// List stored canonical model names, read from the store layout.
    pub fn list(&self) -> Result<Vec<String>, StoreError> {
        let root = self.root.join(MANIFEST_ROOT);
        let entries = match std::fs::read_dir(&root) {
            Ok(entries) => entries,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(source) => return Err(io(&root, source)),
        };
        let mut names = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|source| io(&root, source))?;
            if !entry
                .file_type()
                .map_err(|source| io(&root, source))?
                .is_dir()
            {
                continue;
            }
            let manifests = entry.path().join("manifests");
            let files = match std::fs::read_dir(&manifests) {
                Ok(files) => files,
                Err(source) if source.kind() == std::io::ErrorKind::NotFound => continue,
                Err(source) => return Err(io(&manifests, source)),
            };
            for file in files {
                let file = file.map_err(|source| io(&manifests, source))?;
                let name = file.file_name().to_string_lossy().into_owned();
                if name.starts_with("sha256-") || !name.ends_with(".json") {
                    continue;
                }
                let reference = name.trim_end_matches(".json").to_owned();
                let canonical = self
                    .manifest(&reference)
                    .map(|stored| stored.config.name)
                    .ok()
                    .filter(|name| !name.is_empty())
                    .unwrap_or_else(|| reference.clone());
                names.push(canonical);
            }
        }
        names.sort();
        names.dedup();
        Ok(names)
    }

    /// Load a stored manifest by name, alias, or `name@sha256:<hex>`.
    pub fn manifest(&self, reference: &str) -> Result<StoredModel, StoreError> {
        let (name, pin) = parse_reference(reference)?;
        let dir = self.manifest_dir(name);
        let path = match pin {
            Some(digest) => dir.join(format!("sha256-{}.json", digest_hex(digest)?)),
            None => dir.join(format!("{}.json", store_name(name))),
        };
        let metadata = match std::fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
                return Err(StoreError::Reference(format!(
                    "manifest not found: {}",
                    path.display()
                )));
            }
            Err(source) => return Err(io(&path, source)),
        };
        if metadata.len() > MAX_OCI_MANIFEST_BYTES {
            return Err(StoreError::Reference(format!(
                "manifest {} is larger than the {MAX_OCI_MANIFEST_BYTES}-byte limit",
                path.display()
            )));
        }
        let bytes = std::fs::read(&path).map_err(|source| io(&path, source))?;

        let actual = format!("sha256:{}", sha256_hex(&bytes));
        if let Some(expected) = pin
            && expected.to_ascii_lowercase() != actual
        {
            return Err(StoreError::DigestMismatch {
                path,
                expected: expected.to_owned(),
                actual,
            });
        }

        let manifest: OciManifest =
            serde_json::from_slice(&bytes).map_err(|source| StoreError::Json {
                path: path.clone(),
                source,
            })?;
        if manifest.schema_version != OCI_SCHEMA_VERSION
            || manifest.media_type != MEDIA_TYPE_MANIFEST
        {
            return Err(StoreError::Reference(format!(
                "unsupported OCI manifest at {}",
                path.display()
            )));
        }
        if manifest.config.size > MAX_CONFIG_BYTES {
            return Err(StoreError::Reference(format!(
                "config blob is larger than the {MAX_CONFIG_BYTES}-byte limit"
            )));
        }
        let config_path = self.blob_path(&manifest.config.digest)?;
        let metadata = std::fs::metadata(&config_path).map_err(|source| {
            if source.kind() == std::io::ErrorKind::NotFound {
                StoreError::MissingBlob(manifest.config.digest.clone())
            } else {
                io(&config_path, source)
            }
        })?;
        if metadata.len() != manifest.config.size {
            return Err(StoreError::SizeMismatch {
                path: config_path,
                expected: manifest.config.size,
                actual: metadata.len(),
            });
        }
        let config_bytes = self.read_blob(&manifest.config.digest)?;
        let config: ModelConfig =
            serde_json::from_slice(&config_bytes).map_err(|source| StoreError::Json {
                path: config_path,
                source,
            })?;

        Ok(StoredModel {
            manifest,
            config,
            digest: actual,
            manifest_path: path,
        })
    }

    /// Copy every layer blob into `destination`, preserving `subfolder`.
    pub fn materialize(&self, stored: &StoredModel, destination: &Path) -> Result<(), StoreError> {
        let base = match &stored.config.subfolder {
            Some(subfolder) => {
                // The config blob comes from the store, which may have been
                // written by another process or an older version. Never let a
                // subfolder escape the destination.
                if !is_safe_relative_path(subfolder) {
                    return Err(StoreError::InvalidPath(subfolder.clone()));
                }
                destination.join(subfolder)
            }
            None => destination.to_path_buf(),
        };
        for layer in &stored.manifest.layers {
            let path = layer
                .annotations
                .get(ANNOTATION_PATH)
                .ok_or_else(|| StoreError::MissingPath(layer.digest.clone()))?;
            if !is_safe_relative_path(path) {
                return Err(StoreError::InvalidPath(path.clone()));
            }
            let target = base.join(path);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(|source| io(parent, source))?;
            }
            let blob = self.blob_path(&layer.digest)?;
            let metadata = std::fs::metadata(&blob).map_err(|source| io(&blob, source))?;
            if metadata.len() != layer.size {
                return Err(StoreError::SizeMismatch {
                    path: blob,
                    expected: layer.size,
                    actual: metadata.len(),
                });
            }
            std::fs::copy(&blob, &target).map_err(|source| io(&target, source))?;
        }
        Ok(())
    }

    /// Verify every blob's size and sha256.
    pub fn verify(&self, stored: &StoredModel) -> Result<(), StoreError> {
        for descriptor in stored.manifest.blobs() {
            let path = self.blob_path(&descriptor.digest)?;
            let metadata = match std::fs::metadata(&path) {
                Ok(metadata) => metadata,
                Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
                    return Err(StoreError::MissingBlob(descriptor.digest.clone()));
                }
                Err(source) => return Err(io(&path, source)),
            };
            if metadata.len() != descriptor.size {
                return Err(StoreError::SizeMismatch {
                    path,
                    expected: descriptor.size,
                    actual: metadata.len(),
                });
            }
            let (digest, _) = self.hash_file(&path)?;
            if digest != descriptor.digest {
                return Err(StoreError::DigestMismatch {
                    path,
                    expected: descriptor.digest.clone(),
                    actual: digest,
                });
            }
        }
        Ok(())
    }

    /// Remove a stored model and reclaim blobs that no other manifest uses.
    ///
    /// Deletes the named and digest-pinned manifest files for the canonical
    /// name and every alias, then garbage-collects content blobs that are no
    /// longer referenced by any manifest left in the store. Materialized model
    /// files are not touched; remove those through the hub or CLI.
    ///
    /// Blob garbage collection is skipped (and `gc_skipped` is set) when a
    /// remaining manifest cannot be read, because deleting a referenced blob
    /// would corrupt that model. Store maintenance is not synchronized with
    /// other processes; do not run it while another process is importing.
    pub fn remove(&self, reference: &str) -> Result<StoreRemoval, StoreError> {
        let stored = self.manifest(reference)?;
        let canonical = stored.config.name.clone();
        let mut names = BTreeSet::new();
        names.insert(canonical.clone());
        for alias in &stored.config.aliases {
            names.insert(alias.clone());
        }

        let mut report = StoreRemoval {
            model: canonical.clone(),
            ..StoreRemoval::default()
        };

        for name in &names {
            let dir = self.manifest_dir(name);
            // The named manifest is the slot's ownership marker. Never delete
            // a directory whose manifest belongs to another canonical model
            // (possible after manual edits or a botched older store layout).
            let named = dir.join(format!("{}.json", store_name(name)));
            if named.is_file() {
                match self.manifest(name) {
                    Ok(existing) if existing.config.name == canonical => {}
                    _ => continue,
                }
            }
            let metadata = match std::fs::symlink_metadata(&dir) {
                Ok(metadata) => metadata,
                Err(source) if source.kind() == std::io::ErrorKind::NotFound => continue,
                Err(source) => return Err(io(&dir, source)),
            };
            // Do not follow symlinks or treat a file as a manifest directory.
            if !metadata.is_dir() {
                continue;
            }
            let (files, bytes) = directory_usage(&dir)?;
            std::fs::remove_dir_all(&dir).map_err(|source| io(&dir, source))?;
            report.manifests += files;
            report.freed_bytes += bytes;
            if let Some(component) = dir.parent() {
                // Removes the per-model component directory only when empty.
                let _ = std::fs::remove_dir(component);
            }
        }

        match self.referenced_digests()? {
            Some(referenced) => {
                let (blobs, bytes) = self.remove_unreferenced_blobs(&referenced)?;
                report.blobs = blobs;
                report.freed_bytes += bytes;
            }
            None => report.gc_skipped = true,
        }
        Ok(report)
    }

    /// Collect every blob digest referenced by a manifest still in the store.
    ///
    /// Returns `None` when any manifest cannot be read or parsed: a partial
    /// picture must not drive blob deletion.
    fn referenced_digests(&self) -> Result<Option<BTreeSet<String>>, StoreError> {
        let library = self.root.join(MANIFEST_ROOT);
        let components = match std::fs::read_dir(&library) {
            Ok(components) => components,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Some(BTreeSet::new()));
            }
            Err(source) => return Err(io(&library, source)),
        };
        let mut referenced = BTreeSet::new();
        for component in components {
            let component = component.map_err(|source| io(&library, source))?;
            let file_type = component
                .file_type()
                .map_err(|source| io(&component.path(), source))?;
            if !file_type.is_dir() {
                continue;
            }
            let manifests = component.path().join("manifests");
            let entries = match std::fs::read_dir(&manifests) {
                Ok(entries) => entries,
                Err(source) if source.kind() == std::io::ErrorKind::NotFound => continue,
                Err(_) => return Ok(None),
            };
            for entry in entries {
                let entry = match entry {
                    Ok(entry) => entry,
                    Err(_) => return Ok(None),
                };
                let file_type = match entry.file_type() {
                    Ok(file_type) => file_type,
                    Err(_) => return Ok(None),
                };
                // Symlinks and directories are not manifests we wrote.
                if !file_type.is_file() || !entry.file_name().to_string_lossy().ends_with(".json") {
                    continue;
                }
                let path = entry.path();
                let Some(manifest) = read_oci_manifest(&path) else {
                    return Ok(None);
                };
                let Some(config) = normalize_digest(&manifest.config.digest) else {
                    return Ok(None);
                };
                referenced.insert(config);
                for layer in manifest.layers {
                    let Some(digest) = normalize_digest(&layer.digest) else {
                        return Ok(None);
                    };
                    referenced.insert(digest);
                }
            }
        }
        Ok(Some(referenced))
    }

    fn remove_unreferenced_blobs(
        &self,
        referenced: &BTreeSet<String>,
    ) -> Result<(usize, u64), StoreError> {
        let dir = self.root.join(BLOBS_DIR);
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok((0, 0)),
            Err(source) => return Err(io(&dir, source)),
        };
        let mut removed = 0usize;
        let mut bytes = 0u64;
        for entry in entries {
            let entry = entry.map_err(|source| io(&dir, source))?;
            let file_type = entry
                .file_type()
                .map_err(|source| io(&entry.path(), source))?;
            // Only regular, content-addressed files are store-managed.
            if !file_type.is_file() {
                continue;
            }
            let Some(digest) = entry
                .file_name()
                .to_str()
                .and_then(|name| name.strip_prefix("sha256-"))
                .and_then(|hex| normalize_digest(&format!("sha256:{hex}")))
            else {
                continue;
            };
            if referenced.contains(&digest) {
                continue;
            }
            let path = entry.path();
            let len = entry.metadata().map_err(|source| io(&path, source))?.len();
            std::fs::remove_file(&path).map_err(|source| io(&path, source))?;
            removed += 1;
            bytes += len;
        }
        Ok((removed, bytes))
    }

    fn ensure_layout(&self) -> Result<(), StoreError> {
        for dir in [self.root.join(BLOBS_DIR), self.root.join(MANIFEST_ROOT)] {
            std::fs::create_dir_all(&dir).map_err(|source| io(&dir, source))?;
        }
        Ok(())
    }

    fn manifest_dir(&self, name: &str) -> PathBuf {
        self.root
            .join(MANIFEST_ROOT)
            .join(store_name(name))
            .join("manifests")
    }

    fn ensure_manifest_slot(&self, reference: &str, canonical: &str) -> Result<(), StoreError> {
        let slot = self
            .manifest_dir(reference)
            .join(format!("{}.json", store_name(reference)));
        if !slot.is_file() {
            return Ok(());
        }
        match self.manifest(reference) {
            Ok(existing) if existing.config.name == canonical => Ok(()),
            Ok(existing) => Err(StoreError::Reference(format!(
                "store entry {reference:?} already belongs to {:?}",
                existing.config.name
            ))),
            Err(error) => Err(StoreError::Reference(format!(
                "store entry {reference:?} exists but is unreadable: {error}"
            ))),
        }
    }

    fn store_bytes(&self, bytes: &[u8], media_type: &str) -> Result<Descriptor, StoreError> {
        let digest = format!("sha256:{}", sha256_hex(bytes));
        let path = self.blob_path(&digest)?;
        if !path.is_file() {
            write_atomic(&path, bytes)?;
        }
        Ok(Descriptor {
            media_type: media_type.to_owned(),
            digest,
            size: bytes.len() as u64,
            urls: Vec::new(),
            annotations: BTreeMap::new(),
        })
    }

    fn put_file(&self, source: &Path) -> Result<(String, u64), StoreError> {
        let mut input = File::open(source).map_err(|error| io(source, error))?;
        let temp =
            self.root
                .join(BLOBS_DIR)
                .join(format!(".tmp-{}-{}", std::process::id(), nanos()));
        let mut output = File::create(&temp).map_err(|error| io(&temp, error))?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 1024 * 1024];
        let mut size = 0u64;
        loop {
            let read = input.read(&mut buffer).map_err(|error| io(source, error))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            output
                .write_all(&buffer[..read])
                .map_err(|error| io(&temp, error))?;
            size += read as u64;
        }
        output.sync_all().map_err(|error| io(&temp, error))?;
        drop(output);

        let digest = format!("sha256:{}", hex::encode(hasher.finalize()));
        let path = self.blob_path(&digest)?;
        // A blob already at this content address must have the right length;
        // a truncated file left by a crash must not be mistaken for the real
        // content. (A same-length corruption is caught by `Store::verify`.)
        let existing_size = std::fs::metadata(&path)
            .ok()
            .filter(|metadata| metadata.is_file())
            .map(|metadata| metadata.len());
        if existing_size.is_some_and(|existing| existing == size) {
            std::fs::remove_file(&temp).map_err(|error| io(&temp, error))?;
        } else {
            if existing_size.is_some() {
                std::fs::remove_file(&path).map_err(|error| io(&path, error))?;
            }
            match std::fs::rename(&temp, &path) {
                Ok(()) => {}
                // A concurrent import may have won the race with identical
                // content-addressed bytes.
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    std::fs::remove_file(&temp).map_err(|error| io(&temp, error))?;
                }
                Err(error) => return Err(io(&path, error)),
            }
        }
        Ok((digest, size))
    }

    fn hash_file(&self, path: &Path) -> Result<(String, u64), StoreError> {
        let mut input = File::open(path).map_err(|error| io(path, error))?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 1024 * 1024];
        let mut size = 0u64;
        loop {
            let read = input.read(&mut buffer).map_err(|error| io(path, error))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            size += read as u64;
        }
        Ok((format!("sha256:{}", hex::encode(hasher.finalize())), size))
    }
}

/// Whether a stored manifest describes the same model files as `manifest`.
///
/// Used to skip needless imports and to avoid materializing a stale store
/// entry after a registry manifest changed.
pub(crate) fn matches_manifest(stored: &StoredModel, manifest: &RepoManifest) -> bool {
    if stored.config.name != manifest.name
        || stored.config.aliases != manifest.aliases
        || stored.config.model_format != manifest.format
        || stored.config.family != manifest.family
        || stored.config.arch != manifest.arch
        || stored.config.layout.as_deref() != Some(manifest.layout.as_str())
        || stored.config.calibration_mode != manifest.calibration_mode
        || stored.config.revision != manifest.revision
        || stored.config.capabilities.as_ref() != Some(&manifest.capabilities)
        || stored.config.router != manifest.router
        || stored.config.subfolder != manifest.subfolder
    {
        return false;
    }
    let expected: BTreeSet<&str> = manifest
        .files
        .iter()
        .map(|file| file.path.as_str())
        .collect();
    let actual: BTreeSet<&str> = stored
        .manifest
        .layers
        .iter()
        .filter_map(|layer| layer.annotations.get(ANNOTATION_PATH).map(String::as_str))
        .collect();
    if expected != actual {
        return false;
    }
    manifest.files.iter().all(|file| {
        stored.manifest.layers.iter().any(|layer| {
            layer
                .annotations
                .get(ANNOTATION_PATH)
                .is_some_and(|path| path == &file.path)
                && layer
                    .annotations
                    .get(ANNOTATION_ROLE)
                    .is_some_and(|role| role == file.role.name())
                && file.size.is_none_or(|size| layer.size == size)
                && file.sha256.as_ref().is_none_or(|sha| {
                    layer.digest == format!("sha256:{}", sha.to_ascii_lowercase())
                })
        })
    })
}

impl ManifestStore for Store {
    fn list(&self) -> Result<Vec<String>, StoreError> {
        Store::list(self)
    }

    fn get(&self, reference: &str) -> Result<StoredModel, StoreError> {
        Store::manifest(self, reference)
    }

    fn put(&self, manifest: &RepoManifest, model_dir: &Path) -> Result<StoredModel, StoreError> {
        Store::import(self, manifest, model_dir)
    }

    fn materialize(&self, model: &StoredModel, destination: &Path) -> Result<(), StoreError> {
        Store::materialize(self, model, destination)
    }

    fn verify(&self, model: &StoredModel) -> Result<(), StoreError> {
        Store::verify(self, model)
    }
}

fn parse_reference(reference: &str) -> Result<(&str, Option<&str>), StoreError> {
    let parsed = match reference.split_once('@') {
        None => (reference, None),
        Some((name, digest)) if digest.starts_with("sha256:") => (name, Some(digest)),
        Some(_) => return Err(StoreError::Reference(reference.to_owned())),
    };
    if parsed.0.is_empty() {
        return Err(StoreError::Reference(reference.to_owned()));
    }
    Ok(parsed)
}

fn directory_usage(dir: &Path) -> Result<(usize, u64), StoreError> {
    let mut files = 0usize;
    let mut bytes = 0u64;
    for entry in std::fs::read_dir(dir).map_err(|source| io(dir, source))? {
        let entry = entry.map_err(|source| io(dir, source))?;
        let file_type = entry
            .file_type()
            .map_err(|source| io(&entry.path(), source))?;
        // Symlinks and nested directories are not part of the manifest layout;
        // `remove_dir_all` still removes them without following symlinks.
        if file_type.is_file() {
            if entry.file_name().to_string_lossy().ends_with(".json") {
                files += 1;
            }
            bytes += entry
                .metadata()
                .map_err(|source| io(&entry.path(), source))?
                .len();
        }
    }
    Ok((files, bytes))
}

/// Read an OCI manifest for garbage collection. Any unreadable, oversized, or
/// unsupported file yields `None` so the caller skips blob deletion.
fn read_oci_manifest(path: &Path) -> Option<OciManifest> {
    let metadata = std::fs::metadata(path).ok()?;
    if metadata.len() > MAX_OCI_MANIFEST_BYTES {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    let manifest: OciManifest = serde_json::from_slice(&bytes).ok()?;
    if manifest.schema_version != OCI_SCHEMA_VERSION || manifest.media_type != MEDIA_TYPE_MANIFEST {
        return None;
    }
    Some(manifest)
}

fn normalize_digest(digest: &str) -> Option<String> {
    digest_hex(digest).ok().map(|hex| format!("sha256:{hex}"))
}

fn digest_hex(digest: &str) -> Result<String, StoreError> {
    let hex = digest
        .strip_prefix("sha256:")
        .ok_or_else(|| StoreError::Digest(digest.to_owned()))?;
    if hex.len() != 64 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(StoreError::Digest(digest.to_owned()));
    }
    Ok(hex.to_ascii_lowercase())
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn store_name(name: &str) -> String {
    store_component(name)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), StoreError> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("blob");
    let temp = path.with_file_name(format!(
        ".{file_name}.tmp-{}-{}",
        std::process::id(),
        nanos()
    ));
    let result = (|| -> Result<(), StoreError> {
        let mut file = File::create(&temp).map_err(|error| io(&temp, error))?;
        file.write_all(bytes).map_err(|error| io(&temp, error))?;
        file.sync_all().map_err(|error| io(&temp, error))?;
        drop(file);
        std::fs::rename(&temp, path).map_err(|error| io(path, error))
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

fn nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

fn io(path: &Path, source: std::io::Error) -> StoreError {
    StoreError::Io {
        path: path.to_path_buf(),
        source,
    }
}
