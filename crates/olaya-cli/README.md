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
| `--device` | `cpu` | `cpu`, `cuda`, or `metal` |
| `--dtype` | `f32` | `f32`, `f16`, or `bf16` on CUDA/Metal |
| `--tf32` | off | Allow TF32 GEMMs on CUDA |
| `--max-batch` | `32` | Requests per batched forward pass |
| `--batch-wait-ms` | `2` | Collection window for a batch |
| `--api-key` | `OLAYA_API_KEY` | Bearer token for all routes except `/health` |

> [!WARNING]
> A non-loopback bind requires `--api-key` unless `--allow-unauthenticated` is set. Use a TLS
> reverse proxy when exposing the server outside a trusted network.

## Model catalog

On a fresh install, the CLI fetches the catalog bundle from the latest GitHub Release and caches it
under `~/.olaya/cache/registry-catalog.json`. Later commands reuse the cache.

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
  `laya-multilingual`). Per-file download progress is logged to stderr.
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

- [Workspace README](https://github.com/hushink/olaya#readme)
- [Model registry](https://github.com/hushink/olaya/blob/HEAD/registry/README.md)
- [Client SDK](https://github.com/hushink/olaya/blob/HEAD/crates/olaya-sdk/README.md)
