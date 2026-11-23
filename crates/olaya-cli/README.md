# olaya-cli

English | [简体中文](README.zh-CN.md)

The `olaya` command-line binary: ask one decision request, manage the model catalog, or serve the
TypeSafe-compatible HTTP API.

## Install

```sh
cargo install olaya-cli

# from a checkout
cargo install --path crates/olaya-cli

# GPU builds
cargo install olaya-cli --features cuda
cargo install olaya-cli --features metal
```

On Windows, run the CUDA command from an **x64 Native Tools Command Prompt**, or set `NVCC_CCBIN`
to the full path of `cl.exe`; otherwise `nvcc` fails with `Cannot find compiler 'cl.exe' in PATH`.

## Commands

| Command | Purpose |
| --- | --- |
| `olaya ask` | Run one decision request |
| `olaya serve` | Start the HTTP server |
| `olaya models list/show/pull` | Inspect or download models |
| `olaya models export` | Materialize the catalog into a registry directory |
| `olaya models add/remove` | Edit the runtime catalog |
| `olaya models stored/manifest/verify` | Inspect the OCI content store |
| `olaya doctor` | Print paths, catalog source, and diagnostics |

Common invocations:

```sh
# one request; questions.json contains the TypeSafe questions object
olaya ask --model laya --state "I was charged twice." --questions questions.json

# read the state from stdin
printf '%s' '{"message":"hello"}' | olaya ask --model laya --state - --questions questions.json

# pull a router and all of its targets
olaya models pull laya:latest

# inspect effective config, storage, and catalog source
olaya doctor

# local/offline catalog
olaya --registry-dir ./registry models list
OLAYA_OFFLINE=true olaya models list

# use the OCI content store as the catalog
olaya --registry-store models list
```

## Configuration

`serve`, `ask`, and the model commands share a TOML configuration. The CLI searches
`./olaya.toml`, `./sys1.toml`, then `~/.olaya/config.toml`; use `--config` or `OLAYA_CONFIG` to
choose another file. Relative paths are resolved against the file. If no file exists, the CLI writes
a feature-aware template to `~/.olaya/config.toml` on first use.

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

[registry]
# dir = "./registry"
# store = false
# url = "https://github.com/hushink/olaya/releases/latest/download/registry-catalog.json"
timeout_secs = 30
```

CLI flags override environment variables, which override file values. Boolean flags accept explicit
false (`--tf32=false`, `OLAYA_OFFLINE=false`) so a file value of `true` can be overridden.
`olaya doctor` prints the loaded config path and every effective setting. See
[docs/configuration.md](https://github.com/hushink/olaya/blob/HEAD/docs/configuration.md) for the full reference.

## Ask

```sh
olaya models pull laya
olaya ask --model laya \
  --state "I was charged twice for my subscription this month." \
  --questions questions.json
```

`questions.json` is an object keyed by question name; see the
[workspace README](https://github.com/hushink/olaya#quick-start) for a complete example.

```json
{
  "billing": {
    "type": "noul",
    "instructions": "Is this message about billing?"
  }
}
```

The command prints the answers as JSON.

## Serve

```sh
olaya serve --host 127.0.0.1 --port 11435
```

The server exposes `POST /v1/systemone`, `GET /v1/models`, `/health`, `/metrics`, and the model
management API under `/api/*`.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `metal` (global) |
| `--dtype` | `f32` | `f32`, `f16`, or `bf16` on CUDA/Metal (global) |
| `--tf32` | off | Allow TF32 GEMMs on CUDA (global) |
| `--max-batch` | `32` | Requests per batched forward pass |
| `--batch-wait-ms` | `2` | Collection window for a batch |
| `--max-body-bytes` | `8388608` | Maximum HTTP request body |
| `--max-queue` | `64` | Maximum queued requests |
| `--api-key` | `OLAYA_API_KEY` | Bearer token for all routes except `/health` |
| `--allow-unauthenticated[=BOOL]` | `false` | Allow a non-loopback bind without a key |

> [!WARNING]
> A non-loopback bind requires `--api-key` unless `--allow-unauthenticated` is set. Use a TLS
> reverse proxy when exposing the server outside a trusted network.

## Model catalog

On a fresh install, the CLI seeds `~/.olaya/registry` with the manifests embedded in the binary, then
fetches the catalog bundle from the latest GitHub Release and caches it under
`~/.olaya/cache/registry-catalog.json`. Later commands reuse the cache; if the remote catalog and
cache are unavailable, the local registry directory is used.

```sh
olaya models list
olaya models show laya
olaya models pull laya
olaya models export --dir ./registry
```

- `--registry-dir` / `OLAYA_REGISTRY_DIR` selects a filesystem catalog.
- `OLAYA_REGISTRY_URL` selects a hosted catalog bundle.
- `--registry-store` uses the content-addressed store as the catalog.
- Pulling a router such as `laya:latest` pulls every routed checkpoint (`laya` and
  `laya-multilingual`). In a terminal, per-file progress refreshes a single bar/percent line on
  stderr; redirected output receives periodic progress logs instead.
- Hugging Face downloads default to `https://hf-mirror.com`; set
  `HF_ENDPOINT=https://huggingface.co` to use the official endpoint.
- `models list` groups aliases under each canonical model; `--all` emits the flat API-style list.
- `olaya models export` writes the canonical `v1/library/<family>/manifests/<name>.json` layout and
  is idempotent; `--force` overwrites changed files.

Downloads and storage live under `OLAYA_HOME` (default `~/.olaya`). See the
[workspace README](https://github.com/hushink/olaya#catalog-and-storage) for every directory and
environment variable.

## Logging and metrics

```sh
RUST_LOG=olaya=debug,olaya_candle=debug olaya serve
```

`GET /metrics` exposes Prometheus counters and latency histograms. Responses include
`x-typesafe-request-id` for log correlation.

## Documentation

- [Configuration reference](https://github.com/hushink/olaya/blob/HEAD/docs/configuration.md)
- [Library API guide](https://github.com/hushink/olaya/blob/HEAD/docs/library.md)
- [Workspace README](https://github.com/hushink/olaya#readme)
- [Model registry](https://github.com/hushink/olaya/blob/HEAD/registry/README.md)
- [Client SDK](https://github.com/hushink/olaya/blob/HEAD/crates/olaya-sdk/README.md)
