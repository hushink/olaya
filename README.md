# olaya

English | [简体中文](README.zh-CN.md)

**A native Rust runtime for System One decision models.**

Olaya turns text or structured state into calibrated, typed decisions — a probability, a choice, a
score, or a bounding box — without generating text. It is available as an embeddable library, a
TypeSafe-compatible HTTP server, a command-line tool, and an independent client SDK.

> [!NOTE]
> Models run with [candle](https://github.com/huggingface/candle). Olaya does not require ONNX
> Runtime.

## Highlights

- **Typed decision primitives** — noul (yes/no), choice, score, and box questions with calibrated
  answers.
- **Model catalog** — Laya, NLI, GLiClass, Von, Decima, Qwen3Guard, Decision, Kev, and Clef
  families, described by manifests with pinned revisions and file hashes.
- **Embeddable** — a small engine API plus replaceable `ManifestRegistry` and `ManifestStore`
  traits.
- **TypeSafe-compatible server** — `POST /v1/systemone`, `GET /v1/models`, and dynamic model
  management under `/api/*`, with authentication, multipart images, request batching, and metrics.
- **Independent client SDK** — [`olaya-sdk`](crates/olaya-sdk/README.md) targets the official
  TypeSafe/Jev API with typed requests, errors, retries, and images.
- **Hardware-ready** — CPU, CUDA, and Metal; F32, F16, and BF16 weights with optional TF32.
- **Observability** — tracing, Prometheus metrics, and `x-typesafe-request-id` correlation.

## Requirements

- Rust 1.92 or newer when building from source.
- A GPU is optional. CPU inference uses F32; CUDA and Metal can use F16/BF16.

## Quick start

### Install the CLI

```sh
cargo install olaya-cli

# or from a checkout
cargo install --path crates/olaya-cli
```

For GPU builds, add the backend feature:

```sh
cargo install olaya-cli --features cuda
cargo install olaya-cli --features metal
```

On Windows, `nvcc` must find the MSVC host compiler. Run the CUDA command from an **x64 Native
Tools Command Prompt**, or set `NVCC_CCBIN` to the full path of `cl.exe`.

### Pull a model and ask a question

```sh
olaya models list
olaya models pull laya
```

Create `questions.json`:

```json
{
  "billing": {
    "type": "noul",
    "instructions": "Is this message about billing?"
  },
  "tone": {
    "type": "choice",
    "instructions": "What is the tone of this message?",
    "criteria": {
      "calm": "A neutral or polite message",
      "angry": "An upset or hostile message"
    }
  },
  "urgency": {
    "type": "score",
    "instructions": "How urgent is this message?",
    "criteria": ["Can wait", "Needs attention this week", "Needs attention today"]
  }
}
```

```sh
olaya ask --model laya \
  --state "I was charged twice for my subscription this month." \
  --questions questions.json
```

The CLI prints the answers as JSON, keyed by question name.

> [!NOTE]
> `olaya models pull` downloads weights and associated files into `~/.olaya/models`; the first
> request after a pull loads the model into memory.

### Serve the TypeSafe-compatible API

```sh
olaya serve --host 127.0.0.1 --port 11435
```

```sh
curl http://127.0.0.1:11435/v1/models

curl http://127.0.0.1:11435/v1/systemone \
  -H "Content-Type: application/json" \
  --data-binary @request.json
```

`request.json`:

```json
{
  "model": "laya",
  "state": "I was charged twice for my subscription this month.",
  "questions": {
    "billing": {
      "type": "noul",
      "instructions": "Is this message about billing?"
    },
    "tone": {
      "type": "choice",
      "instructions": "What is the tone of this message?",
      "criteria": {
        "calm": "A neutral or polite message",
        "angry": "An upset or hostile message"
      }
    }
  }
}
```

Useful server options:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `11435` | HTTP port |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `metal` |
| `--dtype` | `f32` | `f32`, `f16`, or `bf16` on CUDA/Metal |
| `--tf32` | off | Allow TF32 tensor-core GEMMs on CUDA |
| `--max-batch` | `32` | Requests collected into one forward pass |
| `--batch-wait-ms` | `2` | Collection window for the first request |
| `--max-body-bytes` | `8388608` | Maximum HTTP request body |
| `--max-queue` | `64` | Maximum queued requests before backpressure |
| `--api-key` | `OLAYA_API_KEY` | Bearer token for every route except `/health` |

> [!WARNING]
> The server binds to loopback by default. A non-loopback bind requires `--api-key` (or an explicit
> `--allow-unauthenticated`). The server speaks HTTP; terminate TLS at a reverse proxy when exposing
> it outside a trusted network.

### Configuration file

`serve`, `ask`, and the model commands share a TOML configuration. The CLI looks for `./olaya.toml`,
then `./sys1.toml`, then `~/.olaya/config.toml`; override the path with `--config` or
`OLAYA_CONFIG`. Relative paths are resolved against the configuration file. If no file exists, the
CLI writes a feature-aware template to `~/.olaya/config.toml` on first use.

```toml
[engine]
# "auto" uses CUDA when the build supports it and a GPU is available, then Metal,
# and falls back to CPU.
device = "auto"
dtype = "f32"
tf32 = false
default_model = "laya:latest"
max_loaded = 3

[serve]
host = "127.0.0.1"
port = 11435
# api_key = "..."
allow_unauthenticated = false
max_batch = 32
batch_wait_ms = 2
max_body_bytes = 8388608
max_queue = 64

[hub]
# models_dir = "models"
# cache_dir = "cache"
# tmp_dir = "tmp"
# store_dir = "store"
# offline = false
hf_endpoint = "https://hf-mirror.com"
# modelscope_endpoint = "https://www.modelscope.cn"
# proxy = "http://127.0.0.1:7890"

[registry]
# dir = "./registry"
# store = false
# url = "https://github.com/hushink/olaya/releases/latest/download/registry-catalog.json"
timeout_secs = 30
```

Precedence is CLI flag > environment variable > file value > built-in default. Boolean CLI/env
values accept explicit false (`--tf32=false`, `OLAYA_OFFLINE=false`) so they can override a file
value of `true`. Run `olaya doctor` to print the loaded configuration path and every effective
setting. See [docs/configuration.md](docs/configuration.md) for the full key and environment-variable
reference, per-command scope, and examples.

### Use it as a Rust library

```toml
[dependencies]
olaya = "0.1"
```

```rust
use olaya::{BuiltinFamilies, Engine, Questions, State, load_manifests};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Materialize a local registry with `olaya models export --dir registry`,
    // or point this at any directory of validated model manifests.
    let manifests = load_manifests("registry")?;

    let engine = Engine::builder()
        .builtin_families(BuiltinFamilies::all())
        .manifests(manifests)
        .build()?;

    let questions = Questions::new()
        .noul("billing", "Is this message about billing?")
        .choice_with_instructions(
            "tone",
            "What is the tone of this message?",
            [("calm", "A neutral or polite message"), ("angry", "An upset or hostile message")],
        )
        .score_with_instructions(
            "urgency",
            "How urgent is this message?",
            ["Can wait", "Needs attention this week", "Needs attention today"],
        );

    let response = engine.ask(
        "laya",
        &State::text("I was charged twice for my subscription this month."),
        &questions,
    )?;

    println!("{:?}", response.answers.get("billing"));
    Ok(())
}
```

The engine also supports `ask_typed` for the typed answer set, `ask_batch` and
`ask_batch_responses` for multiple requests sharing a forward pass, and explicit `load`/`unload`.

Choose the device and precision explicitly when embedding:

```rust
use olaya::{BuiltinFamilies, Engine, olaya_candle::device};
use olaya::candle_core::DType;

let device = device::resolve("auto")?; // cuda/metal when compiled and usable, else cpu
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

Batch several requests into one forward pass:

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

Pull from the hub and reuse the content store:

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
println!("engine has {} manifests", engine.manifests().len());
```

Embed the HTTP server with the `serve` feature:

```rust
use std::sync::Arc;
use olaya::{DecisionRuntime, ServeConfig};

let runtime: Arc<dyn DecisionRuntime> = Arc::new(engine);
let config = ServeConfig {
    host: "127.0.0.1".to_owned(),
    port: 11435,
    api_key: Some("secret".to_owned()),
    ..ServeConfig::default()
};
olaya::serve::serve(runtime, config).await?;
```

[`docs/library.md`](docs/library.md) covers custom `ManifestRegistry` implementations, store
verification, metrics, explicit load/unload, error types, and concurrency.

### Use the client SDK

[`olaya-sdk`](crates/olaya-sdk/README.md) is an independent crate for the official TypeSafe/Jev
API. It can target `https://api.typesafe.ai`, a compatible gateway, or a local `olaya serve`.

```toml
[dependencies]
olaya-sdk = "0.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use olaya_sdk::{ChoiceQuestion, Client, NoulQuestion, questions};

#[tokio::main]
async fn main() -> Result<(), olaya_sdk::Error> {
    let client = Client::from_env()?; // TYPESAFE_API_KEY

    let response = client
        .system_one(
            "I was charged twice for my subscription this month.",
            questions! {
                "billing" => NoulQuestion::new("Is this message about billing?"),
                "tone" => ChoiceQuestion::labels(["calm", "angry"]),
            },
        )
        .await?;

    println!("billing: {}", response.noul("billing").expect("answer").noul);
    Ok(())
}
```

The SDK stores responses in typed `SystemOneResponse` values, preserves unknown answer kinds,
supports multipart and inline images, and ships an optional blocking client (`blocking` feature)
and gzip request compression (`gzip` feature).

## Models

The catalog describes each model with a manifest: family, architecture, layout, file roles,
SHA-256 hashes, license, aliases, and optional routing.

| Family | Example model names | Typical use |
| --- | --- | --- |
| Laya | `laya`, `laya-multilingual`, `laya-typed-decisions`, `laya-latest` | General decision and routing |
| NLI | `nli-modernbert`, `nli-modernbert-base`, `nli-deberta-v3` | Zero-shot natural-language inference |
| GLiClass | `gliclass-large`, `gliclass-v4-edge`, `gliclass-instruct-edge` | Label and option classification |
| Von | `von` | Option-marker NLI |
| Decima | `decima-small`, `decima-base`, `decima-base-clean`, `decima-agent`, `decima-guard`, `decima-guard-small` | Ordinal and late-interaction decisions |
| Qwen3Guard | `qwen3guard` | Safety preset classification |
| Decision | `decision-eos-0.8b` | Candidate endpoint decisions |
| Kev | `kev-0.8b` | Pointer-based choice |
| Clef | `clef-flash` | Joint schema decisions over long context |

Aliases are accepted anywhere a model name is used, for example `laya:latest` or `kev:0.8b`. Inspect
the catalog with:

```sh
olaya models list
olaya models show laya
```

Model weights are downloaded from Hugging Face or ModelScope according to the manifest. Hugging Face
downloads default to `https://hf-mirror.com`; set `HF_ENDPOINT=https://huggingface.co` to use the
official endpoint. The catalog itself is published with each release and cached locally.

## Catalog and storage

On a fresh install, the CLI seeds `~/.olaya/registry` with the manifests embedded in the binary, then
reads the catalog bundle from the latest GitHub Release and caches it under
`~/.olaya/cache/registry-catalog.json`. Later commands reuse the cache; if both the remote catalog
and cache are unavailable, the local registry directory is used. To use a different catalog:

```sh
# Local directory of manifests
olaya --registry-dir ./registry models list
# OLAYA_REGISTRY_DIR=./registry

# A hosted catalog bundle
OLAYA_REGISTRY_URL=https://example.com/registry-catalog.json olaya models list
```

Materialize the catalog for offline or embedded use:

```sh
olaya models export --dir ./registry
```

If you distribute models through the content-addressed store instead, `--registry-store` makes the
store act as the catalog.

The OCI store and the materialized model directory are intentional, non-conflicting layers: the
store holds immutable `sha256` blobs plus OCI manifests (deduplication, integrity, distributable
catalog), while `models_dir` is the ordinary file tree that candle tokenizers and safetensors loaders
need. `models_dir` is disposable: deleting it does not lose the model; the next load materializes it
from the store. `Store::materialize` currently copies bytes, so the same content occupies both
locations; hardlink/reflink or direct blob loaders are possible future space optimizations.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLAYA_HOME` | `~/.olaya` | Base directory for defaults |
| `OLAYA_MODELS_DIR` | `$OLAYA_HOME/models` | Materialized model directories |
| `OLAYA_CACHE_DIR` | `$OLAYA_HOME/cache` | Downloads, staging, and the catalog cache |
| `OLAYA_STORE_DIR` | `$OLAYA_HOME/store` | Content-addressed OCI store |
| `OLAYA_TMP_DIR` | `$OLAYA_HOME/tmp` | Per-task download staging |
| `OLAYA_REGISTRY_DIR` | `./registry`, then `$OLAYA_HOME/registry` | Filesystem catalog or local fallback |
| `OLAYA_REGISTRY_URL` | Latest release asset | Hosted catalog bundle |
| `HF_ENDPOINT` | `https://hf-mirror.com` | Hugging Face endpoint (`https://huggingface.co` for official) |
| `MODELSCOPE_ENDPOINT` | ModelScope | ModelScope endpoint override |
| `OLAYA_PROXY` | unset | Proxy for model downloads |
| `OLAYA_OFFLINE` | `false` | Disable network access |
| `OLAYA_REGISTRY_STORE` | `false` | Use the content store as the catalog |
| `OLAYA_REGISTRY_TIMEOUT_SECS` | `30` | Catalog HTTP timeout |
| `OLAYA_DEVICE` | `auto` | Device preference |
| `OLAYA_DTYPE` | `f32` | CUDA/Metal weight dtype |
| `OLAYA_TF32` | `false` | TF32 on CUDA |
| `OLAYA_DEFAULT_MODEL` | `laya:latest` | Fallback model |
| `OLAYA_MAX_LOADED` | `3` | Models kept loaded |
| `OLAYA_HOST` / `OLAYA_PORT` | `127.0.0.1` / `11435` | `serve` bind |
| `OLAYA_API_KEY` | unset | Bearer token for `olaya serve` |
| `OLAYA_ALLOW_UNAUTHENTICATED` | `false` | Allow non-loopback without a key |
| `OLAYA_MAX_BATCH` / `OLAYA_BATCH_WAIT_MS` | `32` / `2` | Batched forward pass |
| `OLAYA_MAX_BODY_BYTES` / `OLAYA_MAX_QUEUE` | `8388608` / `64` | HTTP body and queue limits |

See [docs/configuration.md](docs/configuration.md) for the complete table.

See [registry/README.md](registry/README.md) for the manifest layout and catalog format.

## HTTP API

The server exposes the TypeSafe System One API and a small model-management API:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness check |
| `GET` | `/v1/models` | List model names and aliases |
| `POST` | `/v1/systemone` | Ask noul, choice, score, and box questions |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/api/tags` | List catalog models |
| `GET` | `/api/ps` | List loaded models |
| `POST` | `/api/show` | Show a manifest |
| `POST` | `/api/load` | Load a model |
| `POST` | `/api/unload` | Unload a model |
| `POST` | `/api/pull` | Pull a model |
| `DELETE` | `/api/delete` | Delete a model |

`POST /v1/systemone` accepts JSON or `multipart/form-data` for images. Image and box requests require
`TypeSafe-Danger-Zone-Images: true`; long-context requests can send
`TypeSafe-Danger-Zone-Long-Context: true`. Responses include an `x-typesafe-request-id` header.

The request and response schemas match the official TypeSafe API:
[System One](https://docs.typesafe.ai/api/system-one) and
[List models](https://docs.typesafe.ai/api/list-models).

## Architecture

| Crate | Role |
| --- | --- |
| [`olaya-core`](crates/olaya-core/README.md) | Protocol primitives, validation, manifests, registry/store traits, metrics |
| [`olaya-candle`](crates/olaya-candle/README.md) | Candle encoders, model families, tokenizer, checkpoint loading |
| [`olaya`](crates/olaya/README.md) | Engine, catalog, hub, OCI store, server feature |
| [`olaya-cli`](crates/olaya-cli/README.md) | The `olaya` binary |
| [`olaya-sdk`](crates/olaya-sdk/README.md) | Independent TypeSafe/Jev client |

```text
olaya-sdk ── HTTP ──┐
                    ├──> olaya serve ──> Engine ──> olaya-candle families
olaya-cli ──────────┘
olaya-core: protocol, manifests, registry/store contracts, metrics
```

The crate graph, request lifecycle, batching, storage, and precision strategy are described in the
[design notes (Chinese)](docs/design.md).

## Logging and metrics

Set `RUST_LOG` to control tracing:

```sh
RUST_LOG=olaya=debug,olaya_candle=debug olaya serve
```

`GET /metrics` exposes request totals, errors, model loads, and latency histograms in Prometheus
text format. Pass through the `x-typesafe-request-id` value when reporting an issue.

## Documentation

- [CLI guide](crates/olaya-cli/README.md)
- [Configuration reference](docs/configuration.md)
- [Library API guide](docs/library.md)
- [Client SDK guide](crates/olaya-sdk/README.md)
- [Model registry and manifests](registry/README.md)
- [Design notes (Chinese)](docs/design.md)
- [TypeSafe System One API](https://docs.typesafe.ai/api/system-one)
