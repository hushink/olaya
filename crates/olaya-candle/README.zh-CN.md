# olaya-candle

[English](README.md) | 简体中文

[olaya](https://github.com/hushink/olaya) System One 模型 family 的 candle 实现，以及用于构建新
family 的可复用 SDK。

## 提供的能力

- 面向第三方模型 family 的 `Family`、`FamilyFactory`、`Backbone` 和 `LoadContext`。
- ModernBERT、DeBERTa、BERT、Qwen3 和 Qwen3.5 encoder（包括逐层流式读取）。
- Hugging Face tokenizer 适配器和无额外依赖的 `torch.save` state-dict 读取器。
- 内置 family：Laya、NLI、GLiClass、Von、Decima、Qwen3Guard、Decision、Kev 和 Clef。
- CPU、CUDA 和 Metal 设备，以及支持 dtype 的 encoder 前向。

嵌入方可以直接依赖这个 crate，把 factory 注册进自己的 engine；也可以使用
[`olaya`](https://github.com/hushink/olaya/blob/HEAD/crates/olaya/README.zh-CN.md) 门面，由它接入
内置 family 和 HTTP 服务。

## Feature

Family feature（`laya`、`nli`、`gliclass`、`von`、`decima`、`qwen3guard`、`decision`、`kev`、
`clef`）会启用对应实现及其 encoder/tokenizer 依赖。`cuda` 和 `metal` 用于选择 GPU 后端。

## 文档

- [工作区 README](https://github.com/hushink/olaya/blob/HEAD/README.zh-CN.md)
- [设计说明](https://github.com/hushink/olaya/blob/HEAD/docs/design.md)
