use std::sync::Arc;

use olaya::{
    CalibrationMode, Capabilities, Engine, FileRole, ManifestFile, ManifestStore, RepoManifest,
    Store, StoreError, StoreRegistry,
};
use tempfile::tempdir;

fn file(path: &str, role: FileRole, size: u64) -> ManifestFile {
    ManifestFile {
        path: path.to_owned(),
        role,
        sha256: None,
        size: Some(size),
        url: None,
    }
}

fn manifest(
    name: &str,
    aliases: &[&str],
    subfolder: Option<&str>,
    files: Vec<ManifestFile>,
) -> RepoManifest {
    RepoManifest {
        schema_version: 1,
        name: name.to_owned(),
        aliases: aliases.iter().map(|alias| (*alias).to_owned()).collect(),
        family: "tiny".to_owned(),
        engine: "candle".to_owned(),
        arch: "custom".to_owned(),
        format: "safetensors".to_owned(),
        layout: "tiny-v1".to_owned(),
        subfolder: subfolder.map(str::to_owned),
        files,
        calibration_mode: CalibrationMode::Raw,
        capabilities: Capabilities::classification(),
        router: None,
        license: "Apache-2.0".to_owned(),
        description: "tiny test model".to_owned(),
        release_date: None,
        source: None,
        revision: None,
    }
}

#[test]
fn imports_materializes_and_pins() {
    let models = tempdir().expect("model dir");
    std::fs::write(models.path().join("model.bin"), b"weights").expect("weight");
    std::fs::create_dir_all(models.path().join("tokenizer")).expect("tokenizer dir");
    std::fs::write(models.path().join("tokenizer/tokenizer.json"), b"{}").expect("tokenizer");

    let repo = manifest(
        "tiny",
        &["tiny-latest"],
        None,
        vec![
            file("model.bin", FileRole::Weights, 7),
            file("tokenizer/tokenizer.json", FileRole::Tokenizer, 2),
        ],
    );

    let root = tempdir().expect("store dir");
    let store = Store::open(root.path().join("store"));
    let stored = store.put(&repo, models.path()).expect("import");

    assert_eq!(stored.manifest.layers.len(), 2);
    assert!(stored.digest.starts_with("sha256:"));
    assert_eq!(stored.config.family, "tiny");
    assert!(stored.manifest_path.is_file());

    let roundtrip = stored.to_repo_manifest();
    roundtrip.validate().expect("roundtrip manifest validates");
    assert_eq!(roundtrip.name, "tiny");
    assert_eq!(roundtrip.aliases, vec!["tiny-latest".to_owned()]);
    assert_eq!(roundtrip.family, "tiny");
    assert_eq!(roundtrip.files.len(), 2);
    assert_eq!(roundtrip.files[0].path, "model.bin");
    assert_eq!(roundtrip.files[0].size, Some(7));
    assert!(roundtrip.files[0].sha256.is_some());

    let by_name = store.get("tiny").expect("by name");
    assert_eq!(by_name.digest, stored.digest);
    let by_alias = store.get("tiny-latest").expect("by alias");
    assert_eq!(by_alias.digest, stored.digest);
    let pinned = store
        .get(&format!("tiny@{}", stored.digest))
        .expect("by digest");
    assert_eq!(pinned.digest, stored.digest);

    let names = store.list().expect("list");
    assert_eq!(names, vec!["tiny".to_owned()]);

    store.verify(&stored).expect("verify");

    let restored = tempdir().expect("restore dir");
    store
        .materialize(&stored, restored.path())
        .expect("materialize");
    assert_eq!(
        std::fs::read(restored.path().join("model.bin")).expect("weight"),
        b"weights"
    );
    assert_eq!(
        std::fs::read(restored.path().join("tokenizer/tokenizer.json")).expect("tokenizer"),
        b"{}"
    );
}

#[test]
fn materializes_a_subfolder() {
    let repo_dir = tempdir().expect("repo dir");
    let subfolder = repo_dir.path().join("multilingual");
    std::fs::create_dir_all(&subfolder).expect("subfolder");
    std::fs::write(subfolder.join("model.bin"), b"multi").expect("weight");
    let repo = manifest(
        "tiny-multi",
        &[],
        Some("multilingual"),
        vec![file("model.bin", FileRole::Weights, 5)],
    );

    let root = tempdir().expect("store dir");
    let store = Store::open(root.path().join("store"));
    let stored = store.put(&repo, repo_dir.path()).expect("import");
    assert_eq!(stored.config.subfolder.as_deref(), Some("multilingual"));
    let roundtrip = stored.to_repo_manifest();
    roundtrip.validate().expect("roundtrip manifest validates");
    assert_eq!(roundtrip.subfolder.as_deref(), Some("multilingual"));

    let destination = tempdir().expect("destination");
    store
        .materialize(&stored, destination.path())
        .expect("materialize");
    assert_eq!(
        std::fs::read(destination.path().join("multilingual/model.bin")).expect("weight"),
        b"multi"
    );
}

#[test]
fn rejects_a_declared_digest_mismatch() {
    let repo_dir = tempdir().expect("repo dir");
    std::fs::write(repo_dir.path().join("model.bin"), b"weights").expect("weight");
    let mut repo = manifest(
        "tiny",
        &[],
        None,
        vec![file("model.bin", FileRole::Weights, 7)],
    );
    repo.files[0].sha256 = Some("00".repeat(32));

    let root = tempdir().expect("store dir");
    let store = Store::open(root.path().join("store"));
    assert!(matches!(
        store.put(&repo, repo_dir.path()),
        Err(StoreError::DigestMismatch { .. })
    ));
}

#[test]
fn detects_blob_tampering() {
    let repo_dir = tempdir().expect("repo dir");
    std::fs::write(repo_dir.path().join("model.bin"), b"weights").expect("weight");
    let repo = manifest(
        "tiny",
        &[],
        None,
        vec![file("model.bin", FileRole::Weights, 7)],
    );

    let root = tempdir().expect("store dir");
    let store = Store::open(root.path().join("store"));
    let stored = store.put(&repo, repo_dir.path()).expect("import");
    store.verify(&stored).expect("clean store verifies");

    let blob = store
        .blob_path(&stored.manifest.layers[0].digest)
        .expect("blob path");
    std::fs::write(&blob, b"tampered").expect("tamper");
    assert!(store.verify(&stored).is_err());
}

#[test]
fn a_store_can_back_the_registry() {
    let repo_dir = tempdir().expect("repo dir");
    std::fs::write(repo_dir.path().join("model.bin"), b"weights").expect("weight");
    let repo = manifest(
        "tiny",
        &["tiny-latest"],
        None,
        vec![file("model.bin", FileRole::Weights, 7)],
    );
    let root = tempdir().expect("store dir");
    let store: Arc<dyn ManifestStore> = Arc::new(Store::open(root.path().join("store")));
    store.put(&repo, repo_dir.path()).expect("import");

    let engine = Engine::builder()
        .registry(Arc::new(StoreRegistry::new(store.clone())))
        .expect("store registry loads")
        .store(store)
        .build()
        .expect("engine builds");
    let manifests = engine.manifests();
    assert_eq!(manifests.len(), 1);
    assert_eq!(manifests[0].name, "tiny");
    assert_eq!(manifests[0].aliases, vec!["tiny-latest".to_owned()]);
    assert_eq!(manifests[0].files.len(), 1);
}

#[test]
fn the_engine_builder_accepts_a_custom_store() {
    let root = tempdir().expect("store dir");
    let store: Arc<dyn ManifestStore> = Arc::new(Store::open(root.path().join("store")));
    let engine = Engine::builder()
        .store(store)
        .build()
        .expect("engine builds");
    assert!(engine.list_models().is_empty());
}

#[test]
fn works_as_a_trait_object() {
    let repo_dir = tempdir().expect("repo dir");
    std::fs::write(repo_dir.path().join("model.bin"), b"weights").expect("weight");
    let repo = manifest(
        "tiny",
        &[],
        None,
        vec![file("model.bin", FileRole::Weights, 7)],
    );

    let root = tempdir().expect("store dir");
    let store: Arc<dyn ManifestStore> = Arc::new(Store::open(root.path().join("store")));
    let stored = store.put(&repo, repo_dir.path()).expect("import");
    assert_eq!(store.get("tiny").expect("get").digest, stored.digest);
    store.verify(&stored).expect("verify");
}
