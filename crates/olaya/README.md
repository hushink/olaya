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

Downloads use `https://hf-mirror.com` by default. Set `HF_ENDPOINT=https://huggingface.co` or
`HubConfig::hf_endpoint` to use the official Hugging Face endpoint.

## Features

| Feature | Effect |
| --- | --- |
| default | All built-in families, the download hub, and the remote catalog |
| `serve` | TypeSafe-compatible HTTP server and model management API |
| `cuda` / `metal` | GPU backends |
| `all-families` | Explicitly enable every family |
| `laya`, `nli`, `gliclass`, `von`, `decima`, `qwen3guard`, `decision`, `kev`, `clef` | Individual families |

## Documentation

- [Workspace README](https://github.com/hushink/olaya#readme)
- [Design notes (Chinese)](https://github.com/hushink/olaya/blob/HEAD/docs/design.md)
- [Client SDK](https://github.com/hushink/olaya/blob/HEAD/crates/olaya-sdk/README.md)
