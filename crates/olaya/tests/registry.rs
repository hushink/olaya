use std::collections::BTreeMap;
use std::sync::Arc;

use olaya::{
    CalibrationMode, Capabilities, Engine, ManifestRegistry, RegistryError, RepoManifest,
    RouterConfig, load_manifests,
};
use tempfile::tempdir;

struct StaticRegistry(Vec<RepoManifest>);

impl ManifestRegistry for StaticRegistry {
    fn manifests(&self) -> Result<Vec<RepoManifest>, RegistryError> {
        Ok(self.0.clone())
    }
}

fn router_manifest(name: &str, aliases: &[&str]) -> RepoManifest {
    RepoManifest {
        schema_version: 1,
        name: name.to_owned(),
        aliases: aliases.iter().map(|alias| (*alias).to_owned()).collect(),
        family: "router".to_owned(),
        engine: "router".to_owned(),
        arch: "router".to_owned(),
        format: "router".to_owned(),
        layout: "router".to_owned(),
        subfolder: None,
        files: Vec::new(),
        calibration_mode: CalibrationMode::Clamp,
        capabilities: Capabilities::classification(),
        router: Some(RouterConfig {
            strategy: "script".to_owned(),
            routes: BTreeMap::from([("english".to_owned(), "laya".to_owned())]),
            default: "laya".to_owned(),
        }),
        license: "Apache-2.0".to_owned(),
        description: "test router".to_owned(),
        release_date: None,
        source: None,
        revision: None,
    }
}

#[test]
fn a_file_may_be_named_after_an_alias() {
    let dir = tempdir().expect("registry tempdir");
    let manifest = router_manifest("laya-multilingual", &["laya-multi"]);
    std::fs::write(
        dir.path().join("laya-multi.json"),
        serde_json::to_vec(&manifest).expect("manifest serializes"),
    )
    .expect("write manifest");

    let manifests = load_manifests(dir.path()).expect("alias-named manifest loads");
    assert_eq!(manifests.len(), 1);
    assert_eq!(manifests[0].name, "laya-multilingual");
}

#[test]
fn a_file_whose_stem_matches_neither_name_nor_alias_is_rejected() {
    let dir = tempdir().expect("registry tempdir");
    let manifest = router_manifest("laya-latest", &["laya:latest"]);
    std::fs::write(
        dir.path().join("other.json"),
        serde_json::to_vec(&manifest).expect("manifest serializes"),
    )
    .expect("write manifest");

    let error = load_manifests(dir.path()).expect_err("mismatch is rejected");
    assert!(matches!(error, RegistryError::NameMismatch { .. }));
}

#[test]
fn the_repository_registry_has_twenty_one_manifests() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../registry");
    let manifests = load_manifests(dir).expect("repository registry loads");
    assert_eq!(manifests.len(), 21);
}

#[test]
fn the_engine_accepts_a_custom_registry() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../registry");
    let manifests = load_manifests(dir).expect("repository registry loads");
    let engine = Engine::builder()
        .registry(Arc::new(StaticRegistry(manifests)))
        .expect("custom registry loads")
        .build()
        .expect("engine builds");
    assert_eq!(engine.manifests().len(), 21);
}

#[test]
fn an_empty_registry_is_reported() {
    let dir = tempdir().expect("registry tempdir");
    let error = load_manifests(dir.path()).expect_err("empty registry");
    assert!(matches!(error, RegistryError::Empty { .. }));
}
