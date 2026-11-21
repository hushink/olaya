# olaya-candle

English | [简体中文](README.zh-CN.md)

Candle implementations of the [olaya](https://github.com/hushink/olaya) System One model families,
plus the reusable SDK for building new families.

## What it provides

- `Family`, `FamilyFactory`, `Backbone`, and `LoadContext` for third-party model families.
- Encoders for ModernBERT, DeBERTa, BERT, Qwen3, and Qwen3.5 (including layer-streaming reads).
- A Hugging Face tokenizer adapter and a dependency-free `torch.save` state-dict reader.
- Built-in families: Laya, NLI, GLiClass, Von, Decima, Qwen3Guard, Decision, Kev, and Clef.
- CPU, CUDA, and Metal devices, with dtype-aware forwards for supported encoders.

Embedders can depend on this crate directly and register factories with their own engine, or use
the [`olaya`](https://github.com/hushink/olaya/blob/HEAD/crates/olaya/README.md) facade, which wires
the built-in families and the HTTP service.

## Features

Family features (`laya`, `nli`, `gliclass`, `von`, `decima`, `qwen3guard`, `decision`, `kev`,
`clef`) enable the corresponding implementation and its encoder/tokenizer dependencies. `cuda` and
`metal` select a GPU backend.

## Documentation

- [Workspace README](https://github.com/hushink/olaya#readme)
- [Design notes (Chinese)](https://github.com/hushink/olaya/blob/HEAD/docs/design.md)
