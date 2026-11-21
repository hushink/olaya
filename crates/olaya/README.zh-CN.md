# olaya

[English](README.md) | 简体中文

[olaya](https://github.com/hushink/olaya) System One 运行时的门面 crate。它将协议层、candle 模型
family、模型目录、下载 hub 和存储组合成可嵌入的 engine，并提供可选的 TypeSafe 兼容 HTTP 服务。

## 提供的能力

- `Engine` 和经过校验的模型目录，目录来源由可替换的 `ManifestRegistry` 提供。
- 基于 shaul 的下载 hub，以及带 digest 固定的内容寻址 OCI `ManifestStore`。
- 内置 candle family：Laya、NLI、GLiClass、Von、Decima、Qwen3Guard、Decision、Kev 和 Clef。
- 可选 `serve` feature：基于 axum 的 `POST /v1/systemone`、`GET /v1/models` 和 `/api/*` 模型管理。
- 请求批处理、tracing、Prometheus 指标和 request ID。

## 示例

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

`load_manifests` 读取经过校验的 manifest 目录。可用 `olaya models export --dir registry` 生成，
或传入由自定义 `ManifestRegistry` 加载的目录。

下载默认使用 `https://hf-mirror.com`。设置 `HF_ENDPOINT=https://huggingface.co` 或
`HubConfig::hf_endpoint` 可改用 Hugging Face 官方端点。

## Feature

| Feature | 作用 |
| --- | --- |
| 默认 | 全部内置 family、下载 hub 和远端目录 |
| `serve` | TypeSafe 兼容 HTTP 服务和模型管理 API |
| `cuda` / `metal` | GPU 后端 |
| `all-families` | 显式启用全部 family |
| `laya`、`nli`、`gliclass`、`von`、`decima`、`qwen3guard`、`decision`、`kev`、`clef` | 单独启用某个 family |

## 文档

- [工作区 README](https://github.com/hushink/olaya/blob/HEAD/README.zh-CN.md)
- [设计说明](https://github.com/hushink/olaya/blob/HEAD/docs/design.md)
- [客户端 SDK](https://github.com/hushink/olaya/blob/HEAD/crates/olaya-sdk/README.zh-CN.md)
