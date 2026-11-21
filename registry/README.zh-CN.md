# 模型目录

[English](README.md) | 简体中文

本目录是 olaya 模型目录的 manifest 源。运行时不会从 crate 内直接读取这些文件；CLI 会从最新 release
拉取生成的 catalog bundle（或读取你指定的目录），并缓存在本地。

## 布局

```text
registry/
└── v1/
    └── library/
        └── <family>/
            └── manifests/
                └── <name>.json
```

每个文件都是一个 `RepoManifest`。文件名必须与 manifest 的 `name` 或某个 alias 一致。加载器会校验
每个文件，并按规范模型名排序。

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

- `family` 选择已注册的 `FamilyFactory`。
- `engine: "candle"` 的模型在进程内加载；`engine: "router"` 的模型通过 `router` 配置转发到其他目录项。
- `arch`、`format` 和 `layout` 告诉 family 如何读取 checkpoint。
- `files[].role` 取值为 `weights`、`tokenizer`、`calibration`、`decision`、`config`、`pickle`
  或 `other`。
- `revision` 固定 Hugging Face 或 ModelScope 的 revision；`sha256` 和 `size` 用于校验下载内容。
- `files[].url` 可以指向模型仓库之外的派生文件，例如与 release 一同托管的 `decision.json` 或
  `calibration.json`。

## Catalog bundle

运行时消费的是 `CatalogBundle`：

```json
{
  "schema_version": 1,
  "manifests": []
}
```

bundle 是发布产物，不提交到仓库。发布流程会从本目录生成 `CatalogBundle`，并作为
`registry-catalog.json` 发布；CLI 默认从最新 GitHub Release 拉取，并缓存到
`~/.olaya/cache/registry-catalog.json`。可以用 `OLAYA_REGISTRY_URL` 覆盖来源，或让
`--registry-dir` 直接指向本目录。

## 模型

| Family | Manifest 名称 | 说明 |
| --- | --- | --- |
| Laya | `laya`、`laya-multilingual`、`laya-typed-decisions`、`laya-latest` | `laya-latest` 是 script/language router |
| NLI | `nli-modernbert`、`nli-modernbert-base`、`nli-deberta-v3` | Zero-shot NLI cross-encoder |
| GLiClass | `gliclass-large`、`gliclass-v4-edge`、`gliclass-instruct-edge` | DeBERTa 与 ModernBERT 变体 |
| Von | `von` | Option-marker NLI |
| Decima | `decima-small`、`decima-base`、`decima-base-clean`、`decima-agent`、`decima-guard`、`decima-guard-small` | Ordinal 与 late-interaction 决策 |
| Qwen3Guard | `qwen3guard` | Preset 安全问题 |
| Decision | `decision-eos-0.8b` | Qwen3.5 backbone + candidate-endpoint head |
| Kev | `kev-0.8b` | Qwen3.5 backbone + pointer head |
| Clef | `clef-flash` | Qwen3.5 joint schema head，文本路径 |

`laya:latest`、`kev:0.8b`、`gliclass:large` 等 alias 会解析到同一 manifest。

## 内容存储

模型也可以通过内容寻址 OCI store 分发。拉取后，store 中会包含模型字节和 OCI manifest；
`--registry-store` 可以让 CLI 把该 store 当作目录，因此分发的 store 可以自给自足。

store 与 registry 的契约定义在 `olaya-core`，默认文件系统实现在 `olaya`。存储路径和环境变量见
[工作区 README](../README.zh-CN.md)。
