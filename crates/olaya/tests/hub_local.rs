use olaya::{CalibrationMode, Capabilities, FileRole, Hub, HubConfig, ManifestFile, RepoManifest};
use tempfile::tempdir;

#[test]
fn pulls_a_local_repository_that_ships_a_manifest() {
    let source = tempdir().expect("source tempdir");
    std::fs::write(source.path().join("model.bin"), b"test").expect("write weight");

    let manifest = RepoManifest {
        schema_version: 1,
        name: "tiny".to_owned(),
        aliases: vec!["tiny-latest".to_owned()],
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
        description: "tiny test model".to_owned(),
        release_date: None,
        source: None,
        revision: None,
    };
    std::fs::write(
        source.path().join("olaya.manifest.json"),
        serde_json::to_vec_pretty(&manifest).expect("manifest serializes"),
    )
    .expect("write manifest");

    let root = tempdir().expect("model root");
    let config = HubConfig {
        models_dir: root.path().join("models"),
        cache_dir: root.path().join("cache"),
        tmp_dir: root.path().join("tmp"),
        store_dir: root.path().join("store"),
        ..HubConfig::default()
    };
    let hub = Hub::new(config.clone());
    let reference = format!("local:{}", source.path().display());
    let pulled = hub.pull(&reference, None).expect("pull succeeds");

    assert_eq!(pulled.name, "tiny");
    assert_eq!(pulled.aliases, vec!["tiny-latest".to_owned()]);
    let target = config.model_dir("tiny").join("model.bin");
    assert_eq!(std::fs::read(target).expect("copied weight"), b"test");

    let store = olaya::Store::open(&config.store_dir);
    let stored = store.manifest("tiny").expect("stored manifest");
    assert_eq!(stored.manifest.layers.len(), 1);
    assert_eq!(
        store.manifest("tiny-latest").expect("alias").digest,
        stored.digest
    );
    store.verify(&stored).expect("stored blobs verify");
    assert!(
        config
            .store_dir
            .join("v1/library/tiny/manifests/tiny.json")
            .is_file()
    );

    // Cache and staging are disposable: removing them must not touch the
    // materialized model or the authoritative store blobs.
    std::fs::remove_dir_all(&config.cache_dir).expect("remove cache");
    std::fs::remove_dir_all(&config.tmp_dir).expect("remove tmp");
    assert!(config.model_dir("tiny").join("model.bin").is_file());
    store
        .verify(&stored)
        .expect("store blobs survive cache removal");

    drop(source);
    std::fs::remove_dir_all(config.model_dir("tiny")).expect("remove materialized model");
    let restored = hub
        .pull(&reference, Some(&pulled))
        .expect("pull restores from the store without the source");
    assert_eq!(restored.name, "tiny");
    assert_eq!(
        std::fs::read(config.model_dir("tiny").join("model.bin")).expect("restored weight"),
        b"test"
    );
}

#[test]
fn refuses_to_replace_an_existing_catalog_model_from_an_uncatalogued_reference() {
    let existing = RepoManifest {
        schema_version: 1,
        name: "tiny".to_owned(),
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
        description: "trusted".to_owned(),
        release_date: None,
        source: None,
        revision: None,
    };

    let source = tempdir().expect("source tempdir");
    std::fs::write(source.path().join("model.bin"), b"evil").expect("write weight");
    let replacement = RepoManifest {
        description: "untrusted".to_owned(),
        files: vec![ManifestFile {
            path: "model.bin".to_owned(),
            role: FileRole::Weights,
            sha256: None,
            size: Some(4),
            url: None,
        }],
        ..existing.clone()
    };
    std::fs::write(
        source.path().join("olaya.manifest.json"),
        serde_json::to_vec_pretty(&replacement).expect("manifest serializes"),
    )
    .expect("write manifest");

    let root = tempdir().expect("model root");
    let config = HubConfig {
        models_dir: root.path().join("models"),
        cache_dir: root.path().join("cache"),
        tmp_dir: root.path().join("tmp"),
        store_dir: root.path().join("store"),
        ..HubConfig::default()
    };
    let engine = olaya::Engine::builder()
        .manifests(vec![existing])
        .hub(config)
        .build()
        .expect("engine");
    let reference = format!("local:{}", source.path().display());
    let error = engine.pull(&reference).expect_err("clobber is refused");
    assert!(error.to_string().contains("refusing to replace"), "{error}");
}

#[test]
fn a_corrupt_download_is_removed_for_the_next_pull() {
    let source = tempdir().expect("source tempdir");
    std::fs::write(source.path().join("model.bin"), b"test").expect("write weight");

    let manifest = RepoManifest {
        schema_version: 1,
        name: "tiny".to_owned(),
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
            sha256: Some("00".repeat(32)),
            size: Some(4),
            url: None,
        }],
        calibration_mode: CalibrationMode::Raw,
        capabilities: Capabilities::classification(),
        router: None,
        license: "Apache-2.0".to_owned(),
        description: "tiny test model".to_owned(),
        release_date: None,
        source: None,
        revision: None,
    };
    std::fs::write(
        source.path().join("olaya.manifest.json"),
        serde_json::to_vec_pretty(&manifest).expect("manifest serializes"),
    )
    .expect("write manifest");

    let root = tempdir().expect("model root");
    let config = HubConfig {
        models_dir: root.path().join("models"),
        cache_dir: root.path().join("cache"),
        tmp_dir: root.path().join("tmp"),
        store_dir: root.path().join("store"),
        ..HubConfig::default()
    };
    let hub = Hub::new(config.clone());
    let reference = format!("local:{}", source.path().display());
    let error = hub.pull(&reference, None).expect_err("digest mismatch");
    assert!(error.to_string().contains("digest mismatch"), "{error}");
    assert!(
        !config.model_dir("tiny").join("model.bin").exists(),
        "the corrupt file must be removed so the next pull re-downloads it"
    );
}

#[test]
fn pulls_a_local_repository_with_a_subfolder() {
    let source = tempdir().expect("source tempdir");
    let subfolder = source.path().join("multilingual");
    std::fs::create_dir_all(&subfolder).expect("create subfolder");
    std::fs::write(subfolder.join("model.bin"), b"multi").expect("write weight");

    let manifest = RepoManifest {
        schema_version: 1,
        name: "tiny-multi".to_owned(),
        aliases: vec![],
        family: "tiny".to_owned(),
        engine: "candle".to_owned(),
        arch: "custom".to_owned(),
        format: "safetensors".to_owned(),
        layout: "tiny-v1".to_owned(),
        subfolder: Some("multilingual".to_owned()),
        files: vec![ManifestFile {
            path: "model.bin".to_owned(),
            role: FileRole::Weights,
            sha256: None,
            size: Some(5),
            url: None,
        }],
        calibration_mode: CalibrationMode::Raw,
        capabilities: Capabilities::classification(),
        router: None,
        license: "Apache-2.0".to_owned(),
        description: "tiny multilingual test model".to_owned(),
        release_date: None,
        source: None,
        revision: None,
    };
    std::fs::write(
        source.path().join("olaya.manifest.json"),
        serde_json::to_vec_pretty(&manifest).expect("manifest serializes"),
    )
    .expect("write manifest");

    let root = tempdir().expect("model root");
    let config = HubConfig {
        models_dir: root.path().join("models"),
        cache_dir: root.path().join("cache"),
        tmp_dir: root.path().join("tmp"),
        store_dir: root.path().join("store"),
        ..HubConfig::default()
    };
    let hub = Hub::new(config.clone());
    let reference = format!("local:{}", source.path().display());
    hub.pull(&reference, None).expect("pull succeeds");

    let target = config
        .model_dir("tiny-multi")
        .join("multilingual")
        .join("model.bin");
    assert_eq!(std::fs::read(target).expect("copied weight"), b"multi");

    let store = olaya::Store::open(&config.store_dir);
    let stored = store.manifest("tiny-multi").expect("stored manifest");
    assert_eq!(stored.config.subfolder.as_deref(), Some("multilingual"));
    let restored = root.path().join("restored");
    store
        .materialize(&stored, &restored)
        .expect("materialize from store");
    assert_eq!(
        std::fs::read(restored.join("multilingual").join("model.bin")).expect("restored weight"),
        b"multi"
    );
}

#[test]
fn pulls_files_that_declare_an_absolute_url() {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    let address = listener.local_addr().expect("address");
    let body = b"derived decision layer".to_vec();
    let served = body.clone();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        let mut buffer = [0u8; 1024];
        let _ = stream.read(&mut buffer);
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            served.len()
        )
        .expect("headers");
        stream.write_all(&served).expect("body");
    });

    let manifest = RepoManifest {
        schema_version: 1,
        name: "url-only".to_owned(),
        aliases: Vec::new(),
        family: "tiny".to_owned(),
        engine: "candle".to_owned(),
        arch: "custom".to_owned(),
        format: "safetensors".to_owned(),
        layout: "tiny-v1".to_owned(),
        subfolder: None,
        files: vec![ManifestFile {
            path: "decision.json".to_owned(),
            role: FileRole::Decision,
            sha256: Some(
                "5e0382c544ac88350680776477419b1a261fae3c679d73be3326151970a9f253".to_owned(),
            ),
            size: Some(body.len() as u64),
            url: Some(format!("http://{address}/decision.json")),
        }],
        calibration_mode: CalibrationMode::Raw,
        capabilities: Capabilities::classification(),
        router: None,
        license: "Apache-2.0".to_owned(),
        description: "url-hosted derived file".to_owned(),
        release_date: None,
        source: None,
        revision: None,
    };

    let root = tempdir().expect("model root");
    let config = HubConfig {
        models_dir: root.path().join("models"),
        cache_dir: root.path().join("cache"),
        tmp_dir: root.path().join("tmp"),
        store_dir: root.path().join("store"),
        allow_private_network: true,
        ..HubConfig::default()
    };
    let hub = Hub::new(config.clone());
    let pulled = hub
        .pull("https://example.invalid/url-only", Some(&manifest))
        .expect("pull succeeds");
    server.join().expect("server");

    assert_eq!(pulled.name, "url-only");
    assert_eq!(
        std::fs::read(config.model_dir("url-only").join("decision.json")).expect("url file"),
        body
    );
    let store = olaya::Store::open(&config.store_dir);
    let stored = store.manifest("url-only").expect("stored manifest");
    store.verify(&stored).expect("stored blobs verify");
}
