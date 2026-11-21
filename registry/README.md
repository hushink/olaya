# Model registry

English | [简体中文](README.zh-CN.md)

This directory is the manifest source for the olaya model catalog. The runtime does not load these
files from the crate; the CLI fetches a generated catalog bundle from the latest release (or a
directory you provide) and caches it locally.

## Layout

```text
registry/
└── v1/
    └── library/
        └── <family>/
            └── manifests/
                └── <name>.json
```

Each file is a `RepoManifest`. The file stem must match the manifest `name` or one of its aliases.
The loader validates every file and sorts the result by canonical model name.

## Manifest

```json
{
  "schema_version": 1,
  "name": "example",
  "aliases": ["example:latest"],
  "family": "laya",
  "engine": "candle",
  "arch": "modernbert",
  "format": "safetensors",
  "layout": "laya-markers-v1",
  "files": [
    {
      "path": "model.safetensors",
      "role": "weights",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "size": 123456789
    },
    {
      "path": "tokenizer/tokenizer.json",
      "role": "tokenizer"
    }
  ],
  "license": "Apache-2.0",
  "description": "An example model manifest.",
  "revision": "0123456789abcdef0123456789abcdef01234567"
}
```

- `family` selects the registered `FamilyFactory`.
- `engine: "candle"` models load in-process; `engine: "router"` models dispatch to other catalog
  entries through a `router` configuration.
- `arch`, `format`, and `layout` tell the family how to read the checkpoint.
- `files[].role` is one of `weights`, `tokenizer`, `calibration`, `decision`, `config`, `pickle`, or
  `other`.
- `revision` pins the Hugging Face or ModelScope revision; `sha256` and `size` verify downloaded
  bytes.
- `files[].url` can point at derived files outside the model repository, such as a
  `decision.json` or `calibration.json` hosted alongside a release.

## Catalog bundle

The runtime consumes a `CatalogBundle`:

```json
{
  "schema_version": 1,
  "manifests": []
}
```

The bundle is a release artifact, not a checked-in file. The release pipeline generates it from
this directory and publishes it as `registry-catalog.json`; the CLI fetches it from the latest
GitHub Release by default and caches it under `~/.olaya/cache/registry-catalog.json`. Override the
source with `OLAYA_REGISTRY_URL` or point `--registry-dir` at this directory directly.

## Models

| Family | Manifest names | Notes |
| --- | --- | --- |
| Laya | `laya`, `laya-multilingual`, `laya-typed-decisions`, `laya-latest` | `laya-latest` is a script/language router |
| NLI | `nli-modernbert`, `nli-modernbert-base`, `nli-deberta-v3` | Zero-shot NLI cross-encoders |
| GLiClass | `gliclass-large`, `gliclass-v4-edge`, `gliclass-instruct-edge` | DeBERTa and ModernBERT variants |
| Von | `von` | Option-marker NLI |
| Decima | `decima-small`, `decima-base`, `decima-base-clean`, `decima-agent`, `decima-guard`, `decima-guard-small` | Ordinal and late-interaction decisions |
| Qwen3Guard | `qwen3guard` | Preset safety questions |
| Decision | `decision-eos-0.8b` | Qwen3.5 backbone with a candidate-endpoint head |
| Kev | `kev-0.8b` | Qwen3.5 backbone with a pointer head |
| Clef | `clef-flash` | Qwen3.5 joint schema head, text path |

Aliases such as `laya:latest`, `kev:0.8b`, or `gliclass:large` resolve to the same manifests.

## Content store

Models can also be distributed through the content-addressed OCI store. After a pull, the store
contains the model bytes and an OCI manifest; `--registry-store` lets the CLI use that store as the
catalog, so a distributed store is self-sufficient.

The store and registry contracts live in `olaya-core`; the default filesystem implementations live
in `olaya`. See the [workspace README](../README.md) for storage paths and environment variables.
