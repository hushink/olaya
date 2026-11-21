# olaya-core

English | [简体中文](README.zh-CN.md)

The protocol and primitive layer for [olaya](https://github.com/hushink/olaya). This crate has no
model runtime, no candle dependency, and no HTTP server.

## What it contains

- System One `Question`, `Answer`, `State`, `Content`, `Questions`, and `Answers` types.
- Request validation and the TypeSafe-compatible request parser.
- Calibration helpers and canonical Python-style JSON rendering.
- Model manifest data structures, including router manifests.
- The `ManifestRegistry` and `ManifestStore` extension traits.
- `RuntimeMetrics`: atomic counters, fixed-bucket latency histograms, serializable snapshots, and
  Prometheus text rendering.

Use [`olaya-candle`](https://github.com/hushink/olaya/blob/HEAD/crates/olaya-candle/README.md) for
the candle family SDK, [`olaya`](https://github.com/hushink/olaya/blob/HEAD/crates/olaya/README.md)
for the engine and server, or [`olaya-cli`](https://github.com/hushink/olaya/blob/HEAD/crates/olaya-cli/README.md)
for the command-line tool.

## Documentation

- [Workspace README](https://github.com/hushink/olaya#readme)
- [Design notes (Chinese)](https://github.com/hushink/olaya/blob/HEAD/docs/design.md)
