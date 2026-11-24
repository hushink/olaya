# 作为 Rust 库使用

`olaya` 是一个进程内决策引擎，不依赖 HTTP 服务。它把 catalog、candle 模型、下载 hub、OCI content
store 和内置 family 组合在一起；`olaya-candle` 可以单独用于自定义 family。

```toml
[dependencies]
olaya = "0.1"
# 需要 HTTP 服务时：
# olaya = { version = "0.1", features = ["serve"] }
# 需要 CUDA/Metal 时：
# olaya = { version = "0.1", features = ["cuda"] }  # 或 "metal"
```

## 最小引擎

```rust
use olaya::{BuiltinFamilies, Engine, Questions, State, load_manifests};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifests = load_manifests("registry")?; // 任意经过校验的 manifest 目录
    let engine = Engine::builder()
        .builtin_families(BuiltinFamilies::all())
        .manifests(manifests)
        .build()?;

    let questions = Questions::new()
        .noul("billing", "Is this message about billing?")
        .choice_with_instructions(
            "tone",
            "What is the tone of this message?",
            [("calm", "Neutral"), ("angry", "Upset")],
        );

    let state = State::text("I was charged twice for my subscription.");
    let response = engine.ask("laya", &state, &questions)?;
    println!("{response:?}");
    Ok(())
}
```

`Registry` 目录可以是 `olaya models export --dir registry` 物化的 catalog，也可以是仓库根目录的
`registry/`。

## 设备、精度与加载上限

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

- `dtype` 只影响 CUDA/Metal 权重；CPU 始终按 F32 计算，head 保持 F32。
- `tf32(true)` 只在 CUDA 上设置 GEMM reduced precision。
- `max_loaded` 控制同时驻留的模型数量，按 LRU 卸载；至少为 1。

`device::resolve` 支持 `auto`、`cpu`、`cuda`、`cuda:<index>`、`metal`。`auto` 只在编译了对应
feature 且设备可用时选择 accelerator。

## 单请求、类型化答案与批处理

```rust
use olaya::{Answers, Questions, State};

let requests: Vec<(State, Questions)> = vec![
    (State::text("Refund please"), Questions::new().noul("refund", "Is this a refund request?")),
    (State::text("The app crashes"), Questions::new().noul("bug", "Is this a bug report?")),
];

let refs: Vec<(&State, &Questions)> = requests
    .iter()
    .map(|(state, questions)| (state, questions))
    .collect();

// 类型化的 Answers（按 question type 分派）
let answers: Vec<Result<Answers, _>> = engine.ask_batch("laya", &refs);

// 或保留完整 SystemOneResponse（包含 usage / request id 等）
let responses = engine.ask_batch_responses("laya", &refs);
```

单请求使用 `engine.ask(...)`；需要原始 typed map 时使用 `engine.ask_typed(...)`。批处理在一次
`predict_batch` 中共享模型 forward；不支持 batch 的 family 由 trait 默认实现逐请求回退。

## 显式加载与卸载

```rust
engine.load("laya")?;
assert!(engine.loaded_models().contains(&"laya".to_owned()));
let handle = engine.model("laya")?;
let answer = handle.ask(&State::text("hello"), &Questions::new())?;
handle.unload();
engine.unload("laya");
```

`load` 会解析 manifest、从 `models_dir` 或 content store 物化文件并构造 family。请求路径也会按需
自动加载。

## 从 Hub 拉取模型

```rust
use olaya::{Hub, HubConfig};

let mut config = HubConfig::default();
config.models_dir = "D:/models".into();
config.cache_dir = "D:/cache".into();
config.tmp_dir = "D:/tmp".into();
config.store_dir = "D:/store".into();
config.hf_endpoint = Some("https://huggingface.co".into());
config.offline = false;
config.ensure_dirs()?;

let hub = Hub::new(config);
let manifest = hub.pull("laya:latest", None)?;
println!("pulled {} ({})", manifest.name, manifest.revision.unwrap_or_default());
```

`HubConfig::from_env()` 读取 CLI 使用的同一组环境变量；`Hub::pull` 在 `offline = true` 时直接拒绝
网络访问。拉取成功后文件会归一化到 `models_dir/<name>/`，并导入 `store_dir` 中的 content store。

## 使用 OCI store 作为 catalog

```rust
use std::sync::Arc;
use olaya::{Engine, Store, StoreRegistry};

let store = Arc::new(Store::open("D:/store"));
let registry = Arc::new(StoreRegistry::new(store.clone()));
let engine = Engine::builder()
    .builtin_families(olaya::BuiltinFamilies::all())
    .registry(registry)?
    .store(store)
    .build()?;
```

`Store::open` 的布局是 `blobs/sha256-*` + `manifests/<name>/`。`Store::verify` 会重新计算每个 blob
的 sha256；CLI 的 `olaya models verify <model>` 是同一实现。

## 内容寻址 store 与物化目录的关系

`olaya` 有三层不同的“模型存在形式”，它们不是互相替代的关系：

| 层 | 目录/位置 | 寻址方式 | 作用 |
| --- | --- | --- | --- |
| OCI content store | `[hub].store_dir` | `sha256:<hex>` blob + OCI manifest | 不可变内容、去重、完整性校验、可作为 catalog 分发 |
| 物化模型目录 | `[hub].models_dir/<name>` | manifest 中的逻辑路径（`config.json`、`tokenizer/tokenizer.json`、`model.safetensors` 等） | candle/tokenizer 需要的普通文件树 |
| 内存中的模型 | 进程内 LRU | canonical model name | 已加载的 `Family`/tensor，受 `max_loaded` 控制 |

内容寻址解决的是“这份字节是什么、是否可信、能否复用”，物化解决的是“模型加载器如何按路径读到它”。
当前 `LoadContext`、`HfTokenizer::from_file` 和 candle 的 safetensors 加载都接收文件系统路径，因此
`Engine::build_family` 在 `models_dir` 缺少文件时会先调用 `Store::materialize`，把 blob 复制回
manifest 规定的目录结构，再加载。`Store::materialize` 目前是 `std::fs::copy`。

因此：

- `--registry-store` 只是把 store 当作 **catalog**（manifest 来源），不会取消物化。
- `models pull` 会同时下载、导入 store，并保留 `models_dir` 的可读文件；如果 store 已有匹配内容，
  则直接从 store 物化，不重新下载。
- `models_dir` 是可丢弃的投影：删除它不会丢模型，下一次 `load` 会从 store 重建；store 才是权威内容。
- 需要真正回收磁盘时，使用 `olaya models remove <name> --purge` 或 `Store::remove`：删除物化
  目录、该模型的 store manifest（含 alias 与按 digest 固定的副本）、文件系统 registry manifest，
  并回收不再被任何剩余 manifest 引用的 blob。若仍有 manifest 无法读取，GC 会跳过并返回
  `gc_skipped`，避免误删。`DELETE /api/delete` 不同，它只把模型从运行中进程的 catalog 移除，
  不删除磁盘内容。
- `cache_dir`/`tmp_dir` 同样可丢弃：删除只会丢 resume 状态、catalog 缓存或 manifest staging；
  已下载模型和 blob 不受影响。`HubConfig::ensure_dirs`/`Hub::pull` 会按需重建它们，并拒绝与
  `models_dir`/`store_dir` 重叠的布局。
- 字节层面确实会有一份重复（blob + 物化文件）。若同盘部署且关心空间，可以在 `Store::materialize`
  中使用 hardlink/reflink（需要处理只读权限或写时复制），或者让加载器直接读取 blob；这两者都还没有
  实现，当前选择 `copy` 以保持物化文件可独立替换、避免改坏 blob。

## 自定义 catalog

`ManifestRegistry` 只要求：

```rust
use olaya::{ManifestRegistry, RegistryError, RepoManifest};

struct MyCatalog { /* database, HTTP, etc. */ }

impl ManifestRegistry for MyCatalog {
    fn manifests(&self) -> Result<Vec<RepoManifest>, RegistryError> {
        // 返回已通过 RepoManifest::validate() 的条目
        todo!()
    }
}
```

把它注入 builder：

```rust
let engine = Engine::builder()
    .builtin_families(olaya::BuiltinFamilies::all())
    .registry(std::sync::Arc::new(MyCatalog { /* ... */ }))?
    .build()?;
```

## 把引擎嵌入 HTTP 服务

启用 `serve` feature：

```rust
use std::sync::Arc;
use olaya::{DecisionRuntime, Engine, ServeConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::builder()
        .builtin_families(olaya::BuiltinFamilies::all())
        .registry_dir("registry")?
        .build()?;
    let runtime: Arc<dyn DecisionRuntime> = Arc::new(engine);

    let config = ServeConfig {
        host: "127.0.0.1".to_owned(),
        port: 11435,
        api_key: Some("secret".to_owned()),
        max_batch: 32,
        max_wait: std::time::Duration::from_millis(2),
        ..ServeConfig::default()
    };
    olaya::serve::serve(runtime, config).await?;
    Ok(())
}
```

`ServeConfig` 的其余字段（`allow_unauthenticated`、`max_body_bytes`、`max_queue`、
`request_timeout`、`max_operations`、`cors_origins`、`allowed_hosts`）都来自
`ServeConfig::default()` 或显式设置；CLI 的 `[serve]` 配置最终映射到同一个结构。
`request_timeout` 默认 600 秒（`Duration::ZERO` 关闭），并覆盖慢速 body 读取；
`max_operations` 限制并发 pull/load。`cors_origins` 默认为空，表示不发送任何 CORS 响应头；只有
显式列出的浏览器 Origin 才会被放行。`allowed_hosts` 为空时，loopback 绑定只接受 loopback Host
名，以阻止 DNS rebinding。

> 引擎构建（registry/catalog 读取和 Hub 客户端）是同步的，可能访问网络或文件系统。在 `tokio`
> 应用中应在进入 async 任务之前构建，或放进 `tokio::task::spawn_blocking`；不要在 async 上下文里
> 直接构造 `RemoteRegistry`/`reqwest::blocking::Client`。

## 指标

```rust
let metrics = engine.metrics_handle(); // 可注入其他组件
let snapshot = engine.metrics();
println!(
    "requests={} errors={}",
    snapshot.requests_total, snapshot.request_errors_total
);
```

`RuntimeMetrics` 是 `Arc` 共享的原子计数器 + 固定桶直方图；serve 的 `/metrics` 输出 Prometheus
text。`EngineBuilder::metrics(handle)` 可以在构建前注入同一个 handle。

## 错误类型

- `ManifestError`：manifest 校验、重复 name/alias、未知 engine。
- `RegistryError`：catalog 读取/远端拉取/解析。
- `RuntimeError`：pull、store、下载、推理。
- `Error`：`ask`/`load` 的顶层错误，包含 `ModelNotFound`、`Validation`、`Model` 等分支。
- `StoreError`：blob 缺失、digest/size 不匹配、IO。

库调用方通常把错误向上传递为 `Box<dyn std::error::Error>`；需要区分时直接匹配具体 enum。

## 线程与并发

`Engine` 是 `Send + Sync`，可以 `Arc` 共享给多个线程/任务；内部 catalog、LRU 和 metrics 使用
`RwLock`/原子量。一次 `ask` 内部串行访问对应模型，跨模型可并行。HTTP 层通过 `Batcher` 把并发请求
合并成 `predict_batch`。
