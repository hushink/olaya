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

显式选择设备和精度：

```rust
use olaya::{BuiltinFamilies, Engine, olaya_candle::device};
use olaya::candle_core::DType;

let device = device::resolve("auto")?;
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

合批多个请求：

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

从 Hub 拉取并复用 content store：

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
println!("loaded {} manifests", engine.manifests().len());
```

下载默认使用 `https://hf-mirror.com`。设置 `HF_ENDPOINT=https://huggingface.co` 或
`HubConfig::hf_endpoint` 可改用 Hugging Face 官方端点。`HubConfig` 还提供 `offline`、`proxy`、
`modelscope_endpoint` 和四个存储目录。

## Feature

| Feature | 作用 |
| --- | --- |
| 默认 | 全部内置 family、下载 hub 和远端目录 |
| `serve` | TypeSafe 兼容 HTTP 服务和模型管理 API |
| `cuda` / `metal` | GPU 后端 |
| `all-families` | 显式启用全部 family |
| `laya`、`nli`、`gliclass`、`von`、`decima`、`qwen3guard`、`decision`、`kev`、`clef` | 单独启用某个 family |

## 文档

- [库 API 指南](https://github.com/hushink/olaya/blob/HEAD/docs/library.md)
- [配置参考](https://github.com/hushink/olaya/blob/HEAD/docs/configuration.md)
- [工作区 README](https://github.com/hushink/olaya/blob/HEAD/README.zh-CN.md)
- [设计说明](https://github.com/hushink/olaya/blob/HEAD/docs/design.md)
- [客户端 SDK](https://github.com/hushink/olaya/blob/HEAD/crates/olaya-sdk/README.zh-CN.md)
