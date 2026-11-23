# olaya

English | [简体中文](README.zh-CN.md)

The facade crate for the [olaya](https://github.com/hushink/olaya) System One runtime. It wires the
protocol layer, candle model families, model catalog, download hub, and storage into one embeddable
engine, with an optional TypeSafe-compatible HTTP service.

## What it provides

- `Engine` and a validated model catalog backed by a replaceable `ManifestRegistry`.
- The shaul-based download hub and a content-addressed OCI `ManifestStore` with digest pinning.
- Built-in candle families: Laya, NLI, GLiClass, Von, Decima, Qwen3Guard, Decision, Kev, and Clef.
- Optional `serve` feature: `POST /v1/systemone`, `GET /v1/models`, and `/api/*` model management
  over axum.
- Request batching, tracing, Prometheus metrics, and request IDs.

## Example

```rust
use olaya::{BuiltinFamilies, Engine, Questions, State, load_manifests};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifests = load_manifests("registry")?;
    let engine = Engine::builder()
        .builtin_families(BuiltinFamilies::all())
        .manifests(manifests)
        .build()?;

    let questions = Questions::new().noul("urgent", "Does this need a human now?");
    let response = engine.ask(
        "laya",
        &State::text("The site is down and checkout is blocked."),
        &questions,
    )?;

    println!("{:?}", response.answers.get("urgent"));
    Ok(())
}
```

`load_manifests` reads a directory of validated manifests. Create one with
`olaya models export --dir registry`, or pass a catalog loaded by your own `ManifestRegistry`.

Choose the device and precision explicitly:

```rust
use olaya::{BuiltinFamilies, Engine, olaya_candle::device};
use olaya::candle_core::DType;

let device = device::resolve("auto")?;
let engine = Engine::builder()
    .builtin_families(BuiltinFamilies::all())
    .registry_dir("registry")?
    .device(device)
    .dtype(DType::F16)
    .tf32(true)
    .max_loaded(2)
    .default_model("laya:latest")
    .build()?;
```

Batch requests sharing one forward pass:

```rust
use olaya::{Questions, State};

let requests = vec![
    (State::text("Refund please"), Questions::new().noul("refund", "Is this a refund request?")),
    (State::text("The app crashes"), Questions::new().noul("bug", "Is this a bug report?")),
];
let refs = requests.iter().map(|(s, q)| (s, q)).collect::<Vec<_>>();
for result in engine.ask_batch("laya", &refs) {
    println!("{:?}", result?);
}
```

Pull from the Hub and use the content store as a catalog:

```rust
use std::sync::Arc;
use olaya::{Hub, HubConfig, Store, StoreRegistry};

let config = HubConfig::default();
let hub = Hub::new(config.clone());
let manifest = hub.pull("laya:latest", None)?;

let store = Arc::new(Store::open(&config.store_dir));
let engine = Engine::builder()
    .builtin_families(olaya::BuiltinFamilies::all())
    .registry(Arc::new(StoreRegistry::new(store.clone())))?
    .store(store)
    .build()?;
println!("loaded {} manifests", engine.manifests().len());
```

Downloads use `https://hf-mirror.com` by default. Set `HF_ENDPOINT=https://huggingface.co` or
`HubConfig::hf_endpoint` to use the official Hugging Face endpoint. `HubConfig` also exposes
`offline`, `proxy`, `modelscope_endpoint`, and the four storage directories.

## Features

| Feature | Effect |
| --- | --- |
| default | All built-in families, the download hub, and the remote catalog |
| `serve` | TypeSafe-compatible HTTP server and model management API |
| `cuda` / `metal` | GPU backends |
| `all-families` | Explicitly enable every family |
| `laya`, `nli`, `gliclass`, `von`, `decima`, `qwen3guard`, `decision`, `kev`, `clef` | Individual families |

## Documentation

- [Library API guide](https://github.com/hushink/olaya/blob/HEAD/docs/library.md)
- [Configuration reference](https://github.com/hushink/olaya/blob/HEAD/docs/configuration.md)
- [Workspace README](https://github.com/hushink/olaya#readme)
- [Design notes (Chinese)](https://github.com/hushink/olaya/blob/HEAD/docs/design.md)
- [Client SDK](https://github.com/hushink/olaya/blob/HEAD/crates/olaya-sdk/README.md)
