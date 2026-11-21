use std::collections::BTreeMap;
use std::path::Path;

use olaya::{
    CalibrationMode, Capabilities, Engine, FileRole, ManifestFile, RepoManifest, RouterConfig,
    State, route_state,
};
use serde_json::json;
use tempfile::tempdir;

fn laya_router() -> RouterConfig {
    RouterConfig {
        strategy: "script".to_owned(),
        routes: BTreeMap::from([
            ("english".to_owned(), "laya".to_owned()),
            ("multilingual".to_owned(), "laya-multilingual".to_owned()),
        ]),
        default: "laya".to_owned(),
    }
}

#[test]
fn english_routes_to_the_english_checkpoint() {
    let state = State::text("I was charged twice. Please refund the extra charge.");
    assert_eq!(route_state(&laya_router(), &state), "laya");
}

#[test]
fn chinese_routes_to_the_multilingual_checkpoint() {
    let state = State::text("我这个月被重复扣款了，请退还多扣的费用。");
    assert_eq!(route_state(&laya_router(), &state), "laya-multilingual");
}

#[test]
fn cyrillic_routes_to_the_multilingual_checkpoint() {
    let state = State::text("Мне дважды списали оплату за подписку.");
    assert_eq!(route_state(&laya_router(), &state), "laya-multilingual");
}

#[test]
fn json_state_is_classified_by_its_text() {
    let english = State::from_value(json!({"message": "Please refund the duplicate charge."}))
        .expect("state");
    assert_eq!(route_state(&laya_router(), &english), "laya");

    let chinese = State::from_value(json!({"message": "请退还多扣的费用。"})).expect("state");
    assert_eq!(route_state(&laya_router(), &chinese), "laya-multilingual");
}

#[test]
fn empty_or_punctuation_only_state_uses_the_default() {
    let state = State::text("123 !!!");
    assert_eq!(route_state(&laya_router(), &state), "laya");
}

#[test]
fn unknown_strategy_uses_the_default() {
    let mut router = laya_router();
    router.strategy = "custom".to_owned();
    let state = State::text("我这个月被重复扣款了。");
    assert_eq!(route_state(&router, &state), "laya");
}

fn target_manifest(name: &str, source: &Path) -> RepoManifest {
    RepoManifest {
        schema_version: 1,
        name: name.to_owned(),
        aliases: Vec::new(),
        family: "tiny".to_owned(),
        engine: "candle".to_owned(),
        arch: "custom".to_owned(),
        format: "safetensors".to_owned(),
        layout: "tiny-v1".to_owned(),
        subfolder: None,
        files: vec![ManifestFile {
            path: "model.bin".to_owned(),
            role: FileRole::Weights,
            sha256: None,
            size: Some(4),
            url: None,
        }],
        calibration_mode: CalibrationMode::Raw,
        capabilities: Capabilities::classification(),
        router: None,
        license: "Apache-2.0".to_owned(),
        description: "router target".to_owned(),
        release_date: None,
        source: Some(format!("local:{}", source.display())),
        revision: None,
    }
}

#[test]
fn pulling_a_router_materializes_all_targets() {
    let root = tempdir().expect("root");
    let english = tempdir().expect("english source");
    let multilingual = tempdir().expect("multilingual source");
    std::fs::write(english.path().join("model.bin"), b"test").expect("english weights");
    std::fs::write(multilingual.path().join("model.bin"), b"test").expect("multilingual weights");

    let router = RepoManifest {
        schema_version: 1,
        name: "router".to_owned(),
        aliases: vec!["router:latest".to_owned()],
        family: "router".to_owned(),
        engine: "router".to_owned(),
        arch: "router".to_owned(),
        format: "router".to_owned(),
        layout: "router".to_owned(),
        subfolder: None,
        files: Vec::new(),
        calibration_mode: CalibrationMode::Raw,
        capabilities: Capabilities::classification(),
        router: Some(RouterConfig {
            strategy: "script".to_owned(),
            routes: BTreeMap::from([
                ("english".to_owned(), "target-en".to_owned()),
                ("multilingual".to_owned(), "target-multi".to_owned()),
            ]),
            default: "target-en".to_owned(),
        }),
        license: "Apache-2.0".to_owned(),
        description: "router".to_owned(),
        release_date: None,
        source: None,
        revision: None,
    };

    let engine = Engine::builder()
        .manifests([
            router,
            target_manifest("target-en", english.path()),
            target_manifest("target-multi", multilingual.path()),
        ])
        .models_dir(root.path().join("models"))
        .cache_dir(root.path().join("cache"))
        .tmp_dir(root.path().join("tmp"))
        .store_dir(root.path().join("store"))
        .build()
        .expect("engine builds");

    let pulled = engine.pull("router:latest").expect("router pull");
    assert_eq!(pulled.name, "router");
    for target in ["target-en", "target-multi"] {
        assert_eq!(
            std::fs::read(root.path().join("models").join(target).join("model.bin"))
                .expect("target weights"),
            b"test"
        );
        assert!(engine.resolve(target).is_some(), "target is in the catalog");
    }
}
