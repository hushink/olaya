# olaya 设计定稿

> 基于 candle 的 System One 决策模型运行时。架构以 `ollaya` 为蓝本，推理引擎替换为 candle；只从 `rust-sys1` 取 question/answer 原语。

## 1. 不变量

1. **candle 是唯一运行时引擎**。发布包不依赖 `ort` / `onnxruntime` / `tract` / `onnx`。
2. **`olaya-core` 引擎无关**：无 candle、无 HTTP、无 `shaul`。
3. **`olaya-candle` 是唯一接触 candle 的扩展层**，也是第三方 family 的公共 SDK，并包含可复用的内置 family 实现（laya/nli/gliclass/von/decima，feature-gated）。`olaya` 只做引擎、存储、registry 与 HTTP。
4. **`olaya` 的 `serve` feature 通过 `DecisionRuntime` 面对引擎**，不依赖任何 family；HTTP 依赖全部可选。
5. **Q/A 原语只来自 rust-sys1**：`Content / State / Question / Answer / Questions / Answers / Usage` 及其校验。其余全部按 ollaya 重做。
6. **进程内推理**：没有 daemon ↔ runner 子进程协议。

## 2. crate 图

```text
olaya-core
   ↑                 ↑
olaya-candle ────────┘
   ↑
olaya (serve feature: axum /v1 + /api)
   ↑
olaya-cli
```

`olaya-sdk` 独立于上图：它只通过 HTTP 调用 `/v1/systemone` 与 `/v1/models`，不依赖任何内部 crate。

| crate | 发布 | 职责 |
|---|---|---|
| `olaya-core` | ✅ | Q/A 原语、协议 DTO、`Renderer/Readout/TokenEncoder/Calibration`、manifest、`ManifestRegistry/ManifestStore` trait、`DecisionRuntime` |
| `olaya-candle` | ✅ | `Backbone/Family/FamilyFactory/LoadContext`、`encoders`（ModernBERT / DeBERTa-v2/v3 / BERT / Qwen3 / Qwen3.5，含逐层流式读取）、tokenizer、PyTorch `torch.save` 检查点读取（`torch` feature）、内置 family（laya/nli/gliclass/von/decima/qwen3guard/decision/kev/clef，feature-gated） |
| `olaya` | ✅ | `Engine`、hub（shaul）、默认 OCI store（`Store`/`StoreRegistry`）、`FsRegistry`、catalog、`BuiltinFamilies` 注册 `olaya-candle::families`、async 包装；`serve` feature 提供 axum `/v1/*` + `/api/*` |
| `olaya-cli` | ✅ | bin `olaya`：serve/ask/models/doctor |
| `olaya-sdk` | ✅ | 独立 TypeSafe/Jev 客户端：类型化 Q/A、错误体、重试/超时、图片、异步 + 可选 blocking/gzip；零内部依赖 |
| `xtask` / `conformance` / `benches` | ❌ | 校验、catalog 生成、golden、基准 |

## 3. 核心类型

```rust
pub enum Question { Noul(..), Choice(..), Score(..), Box(..) }
pub enum Answer   { Noul(..), Choice(..), Score(..), Box(..) }
pub enum Content  { Null, String(String), Object(Map<String, Value>), Array(Vec<Value>) }
pub struct State(Content);            // string | object | array
pub struct Questions(IndexMap<String, Question>);
pub struct Answers(IndexMap<String, Answer>);
```

约束：choice 1–255 项，score 1–10 档，questions 非空，assets 非空且每项有 `part` 或 `format+data_base64`。`choice` 可带 `"ordinal": true`（Decima 扩展），此时必须是 2–10 项。`instructions` 缺失或为 `null` 时，所有 family 统一用 question id 作为文本（`Question::instructions_or_id`），非字符串内容按 Python `json.dumps` 渲染。

## 4. 决策层（`olaya-core`）

```rust
pub enum QType { Choice = 0, Score = 1, Noul = 2, Box = 3 }

pub trait TokenEncoder: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<u32>, DecisionError>;
}

pub struct Encoded {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub qtype: QType,
    pub state_tokens: usize,
    pub truncated: bool,
}

pub trait Renderer: Send + Sync {
    fn encode_state(&self, state: &State, enc: &dyn TokenEncoder) -> Result<Vec<u32>, DecisionError>;
    fn render(&self, state_ids: &[u32], q: &Question, enc: &dyn TokenEncoder) -> Result<Encoded, DecisionError>;
}

pub trait Readout: Send + Sync {
    fn read(&self, q: &Question, logits: &[f32], cal: &Calibration, state_tokens: usize) -> Result<Answer, DecisionError>;
}
```

`Calibration` 支持 `CalibrationMode::{Raw, Clamp, OnnxBundle}`；默认 `Clamp`（`[0.5, 5.0]` 或 manifest 的 `temperature_range`）。`CalibrationFile.temperature_map` 支持 `von-entropy-length-v1`：按问题自身 logits 的归一化熵、`state_tokens` 和选项数计算温度，`Calibration::temperature_for` 在所有 family 的答案构造中统一应用。

`Encoded.truncated` 与各 family 自己的行结构（NLI 的 premise、Von 的 state row、Decima 的 state row 等）在状态被裁掉时返回 `DecisionError::StateTruncated`；engine 把它升级为 `RuntimeError::StateTruncated { model }`，不会静默用截断后的状态作答。默认 `Family::predict`（laya）在渲染后检查 `Encoded.truncated`。

`ordinal` choice 由 Decima 的累计链接 head 读取：`qtype` 内部改为 `Score`，概率取 `softmax(ordinal_log_probs(..., config.temperature))`（与上游 `decima.probs` 一致），因此不再经过 choice 校准（避免二次温度），答案仍是 `ChoiceAnswer`，标签顺序与 `criteria` 一致。其他内置 family 在 engine 的能力检查（`Capabilities.ordinal`）和 family 自身两处拒绝 ordinal choice，HTTP 返回 400。

## 5. candle SDK（`olaya-candle`）

```rust
pub struct LoadContext<'a> {
    pub dir: &'a Path,
    pub manifest: &'a RepoManifest,
    pub device: &'a Device,
    pub dtype: DType,
    pub tokenizer: &'a dyn TokenEncoder,
}

pub struct BackboneOutput { pub logits: Tensor, pub act: Option<Tensor> }

pub trait Backbone: Send + Sync {
    fn forward(&self, encoded: &Encoded) -> Result<BackboneOutput, ModelError>;
    fn forward_batch(&self, batch: &[Encoded]) -> Result<Vec<BackboneOutput>, ModelError> { /* default */ }
}

pub trait Family: Send + Sync {
    fn id(&self) -> &'static str;
    fn capabilities(&self) -> Capabilities;
    fn tokenizer(&self) -> &dyn TokenEncoder;
    // 默认 None：family 可自行实现 predict，例如 NLI 的每选项一行。
    fn renderer(&self) -> Option<&dyn Renderer> { None }
    fn backbone(&self) -> Option<&dyn Backbone> { None }
    fn readout(&self) -> Option<&dyn Readout> { None }
    fn calibration(&self) -> &Calibration;
    fn predict(&self, state: &State, questions: &Questions)
        -> Result<SystemOneResponse, ModelError>;
}

pub trait FamilyFactory: Send + Sync {
    fn family(&self) -> &'static str;
    fn load(&self, ctx: &LoadContext<'_>) -> Result<Arc<dyn Family>, ModelError>;
}
```

第三方 family = 依赖 `olaya-core` + `olaya-candle`，实现 `Family` + `FamilyFactory`，在 `Engine::builder().family(...)` 注册。内置 family 位于 `olaya-candle::families::{laya,nli,gliclass,von,decima,qwen3guard,decision,kev,clef}`，由 `olaya` 的 `BuiltinFamilies` 注册；嵌入方也可以只依赖 `olaya-candle` 直接使用这些 factory。`gliclass` 同时支持 DeBERTa-v2/v3 与 ModernBERT encoder（权重统一在 `model.encoder_model.*`），segment embedding 折叠与 average label pooling 对两类 backbone 共用。`qwen3guard` 由 `encoders::qwen3`（无 KV cache 的 Qwen3 因果 trunk）+ 固定 preset 的候选 token 读取组成。`encoders::qwen35` 是 Qwen3.5 文本 trunk：4 层 Gated DeltaNet 对应 1 层全注意力，chunked gated-delta rule（64-token chunk、UT transform 的块倍增求逆、chunk 间 scan）、带输出门的 GQA、partial RoPE 与 zero-centered RMSNorm；`decision` 在该 trunk 上读取每个 option 段末 token 的 contextual endpoint 与全局 query，经共享 bilinear + MLP head 打分；`kev` 在同一 trunk 上先合并 PEFT LoRA，再以 `k(h_opt)·q(h_decide)/16` pointer head 读取每问一行；`clef` 用一个 packed 行编码全部问题，joint schema head 从 question/option span 读取 hidden、经 evidence routing 与 Transformer decoder 联合打分全部 option。`Qwen35StreamingEncoder` 为 9B clef 逐层读取 safetensors 并把单层扩宽到 fp32，峰值内存约为“一层 + 词表”，不要求整模型进内存。

`Backbone` trait 是**行级 System One 接口**：接收 `Encoded`，输出 `BackboneOutput`。`olaya-candle::encoders::*` 是**纯架构编码器**：接收 token ids + attention mask，输出 hidden states，不知道 Question/Answer、manifest 或校准；它们被多个 family 复用。两者不是同一层，因此 `encoders` 与 `families` 保持分离。

## 6. Engine

```rust
Engine::ask(model, &State, &Questions) -> Result<SystemOneResponse, Error>
Engine::ask_typed(...) -> Result<Answers, Error>
Engine::ask_batch(model, &[(&State, &Questions)]) -> Vec<Result<Answers, Error>>
Engine::ask_batch_responses(model, &[(&State, &Questions)]) -> Vec<Result<SystemOneResponse, Error>>
Engine::model(name) -> Result<ModelHandle, Error>
Engine::list_models() / load(name) / unload(name)
```

`DecisionRuntime::predict_batch`（默认逐条调用 `predict`）与 `Family::predict_batch` 把同一模型的多个请求合并成一次 family 前向。kev / decision 把各请求的所有 question row 汇总到一个 `batch * width ≤ 32k` 的批次，qwen3guard 把每个请求的最多两行合并，共用一次 trunk；支持多行的 laya / nli / von 也能通过同一接口批处理。任何批次失败会按请求回退到单条 `predict`，因此批处理不改变错误语义。`Engine::ask_batch` 按输入顺序返回结果，`ask_batch_responses` 保留完整 usage。

同一请求内部也按 row 合批：`Family::predict` 的默认组合先用 renderer 渲出全部问题 row，再调用一次 `Backbone::forward_batch`（默认实现逐行回退）。laya / von / gliclass 覆盖了 `forward_batch`，把 `batch * width ≤ 32k` 的 row 送进一次 encoder；laya 的 decision head 是双向 attention，因此额外传 key-padding mask，padding 位置不参与任何真实 token 的 softmax。nli / decima 的多 option row 已经按各自预算合批。`crates/olaya/tests/row_batch_parity.rs` 比较“多问题一次 ask”与“逐问题 ask”的答案。

### 6.1 日志与指标

库只使用 `tracing` 宏，不安装 subscriber；CLI 在 `main` 里初始化 `EnvFilter`（默认 `info`，`RUST_LOG=olaya=debug,olaya_candle=debug` 可细化）。span 结构：

- `load_model{model,family,arch,device,dtype}`（info）：缓存未命中时创建 family，记录 load 秒数、loaded 计数和错误；
- `ask{model,questions}` / `ask_batch{model,requests}`（debug）：成功时记录 canonical model、input/output tokens、batch size 与耗时，失败记录错误种类；
- `http request{request_id,method,path,status,elapsed_ms}`（debug；5xx 为 warn）：由 serve 中间件统一记录，`x-typesafe-request-id` 同时写入响应头；
- `encoders::*` 的 load（debug）与 Qwen3.5 逐层流式读取（trace）。

不记录 state/question 文本、token 内容或 API key；只记录计数、模型名、路径和耗时。

`olaya-core::metrics::RuntimeMetrics` 是 `Arc` 共享的原子计数器 + 固定桶直方图，clone 共享，`snapshot() -> MetricsSnapshot`（serde）和 `render_prometheus()` 两种出口。`EngineBuilder::metrics(handle)` 可注入同一个 handle；`DecisionRuntime::runtime_metrics()` 让 serve 的 `AppState` 拿到它，HTTP 中间件把 `http_*` 指标写入同一快照，`GET /metrics` 输出 Prometheus text 0.0.4。直方图桶固定，scrape 与请求路径都不分配（除字符串渲染）。测试覆盖克隆共享、桶累计语义、Prometheus 输出与 gauge 不下溢（`olaya-core` unit tests）；`crates/olaya/tests/metrics_smoke.rs` 验证 Engine 的 load/request/unload 计数和共享 handle。

## 7. Manifest 与存储

### 7.1 清单源

清单的源文件在仓库根目录 `registry/v1/library/<model>/manifests/<alias>.json`；`olaya` 启动时递归加载所有 `*.json`，并校验文件 stem 与 manifest 的 `name`/`aliases` 匹配。`BuiltinFamilies` 只注册 family factory，不再硬编码 manifest。`--registry-dir` / `OLAYA_REGISTRY_DIR` 覆盖，默认 `./registry`，其次 `~/.olaya/registry`。

发布包不包含根目录 `registry/`，但 `olaya-cli` 通过 `cargo run -p xtask -- sync-embedded-registry` 把 21 份清单生成到 `crates/olaya-cli/src/registry_seed.rs` 并随包编译。CLI 首次使用隐式 `~/.olaya/registry` 时先用这份 seed 写入清单，保证全新离线安装也有可用 catalog；随后优先读取远端/缓存 catalog，远端和缓存都不可用时回退本地 seed。`cargo run -p xtask -- sync-registry [--output PATH]` 从 `registry/` 构建 `CatalogBundle { schema_version, manifests }`，默认写入 `target/registry-catalog.json`（不进仓库）。release-plz 的 release job 在发布成功后从对应 tag 生成 `registry-catalog.json`，并上传到它创建的每个 GitHub Release；手动创建的 release 由 `.github/workflows/release.yml` 补传。`RemoteRegistry`（`olaya` 的 `remote` feature）在首次使用或 `models export` 需要刷新时用 `reqwest` 阻塞客户端拉取 `OLAYA_REGISTRY_URL`（默认 `https://github.com/hushink/olaya/releases/latest/download/registry-catalog.json`），校验 schema 与每个 manifest 后写入 `<cache_dir>/registry-catalog.json`；后续命令默认读缓存，`refresh=true` 时网络失败会回退缓存并告警。CLI 只在默认 registry 目录缺失或为空且用户没有显式指定目录时启用远端；显式 `--registry-dir` / `OLAYA_REGISTRY_DIR` 解析失败会直接报错，绝不静默替换。`olaya models export [--dir PATH] [--force]` 把有效 catalog（本地目录、远端/缓存或 `--registry-store`）按 `v1/library/<family>/manifests/<name>.json` 写出，比较解析后的 manifest 而不是字节，内容相同跳过，不同需 `--force`。workspace 测试 `registry_builds_a_valid_catalog_bundle` 验证 manifest 源能生成合法 bundle；CI 的 `check-embedded-registry` 防止 seed 与 `registry/` 漂移；发布资产与 tag 的一致性由 release 工作流现场从该 tag 生成保证。

清单源本身可替换：`ManifestRegistry` trait（定义在 `olaya-core`）只要求 `manifests() -> Result<Vec<RepoManifest>, RegistryError>`，`FsRegistry` 是 `olaya` 提供的默认目录实现。嵌入方通过 `Engine::builder().registry(Arc::new(MyRegistry))` 接入数据库、HTTP 或远端 OCI registry；实现方只依赖 `olaya-core`。

`RepoManifest.revision` 钉住 Hub 的 commit/tag；`files[].sha256` / `size` 声明内容校验，`Store::import` 在写入 blob 前复核。若所有文件都带 `sha256`，下载层关闭 provider 校验并以 manifest 为准；否则也关闭 shaul 的 provider 校验（shaul 把 Hub 非 LFS 文件的 git blob id 当作普通 SHA-1 比较，会误报合法文件），改由下载层对 shaul 提供的 LFS sha256 做后置校验。Laya 的三个 manifest 已补齐 revision、sha256 与 size。Hugging Face 下载默认走 `https://hf-mirror.com`，可通过 `HF_ENDPOINT`（如 `https://huggingface.co`）覆盖；manifest 中绝对 `huggingface.co` URL 也会按该端点重写，ModelScope 和其它绝对 URL 不受影响。shaul 的 Hub backend 会写入 `<destination>/<repo-name>/<path>`，hub 下载后按 manifest 路径归一化到 `models_dir/<name>/<subfolder>/<path>`（并兼容旧布局）。

`StoreRegistry` 是内置的另一种实现：它把 `ManifestStore` 当作清单源，用 `StoredModel::to_repo_manifest()` 把 OCI manifest + config 还原成 `RepoManifest`。因此分发的 store 可以同时充当 registry，完全脱离 `registry/` 目录；CLI 的 `--registry-store` 选择该模式。

### 7.2 OCI 内容寻址 store

运行时存储采用 OCI distribution v2（`schemaVersion/config/layers`，`Descriptor{digest,size,urls,annotations}`）。`ManifestStore` trait 与 `StoredModel/StoreError` 定义在 `olaya-core`，方法为 `list/get/put/materialize/verify`；`Store` 是 `olaya` 提供的默认文件系统实现。`pull` 成功后 hub 把模型文件按 sha256 导入 store；模型文件缺失但 store 有匹配内容时，hub 与 engine 都从 store 物化回 `models_dir`，不访问网络。自定义实现只依赖 `olaya-core`。

模型作者写 `olaya.manifest.json`：

```json
{
  "schema_version": 1,
  "name": "laya",
  "aliases": ["laya:en"],
  "family": "laya",
  "engine": "candle",
  "arch": "modernbert",
  "format": "safetensors",
  "layout": "laya-markers-v1",
  "files": [
    { "path": "model.safetensors", "role": "weights", "sha256": "..." },
    { "path": "tokenizer/tokenizer.json", "role": "tokenizer" },
    { "path": "rl_agent_config.json", "role": "calibration" }
  ],
  "calibration_mode": "clamp",
  "capabilities": { "noul": true, "choice": true, "score": true },
  "license": "Apache-2.0"
}
```

存储布局（可配置）：

```text
<store_dir>/v1/library/<name>/manifests/<name>.json
<store_dir>/v1/library/<name>/manifests/sha256-<hex>.json
<store_dir>/v1/library/<alias>/manifests/<alias>.json
<store_dir>/blobs/sha256-<hex>
<models_dir>/<name>/<subfolder>/<path>   # 物化后的模型文件
<cache_dir>/state                        # shaul 任务 sidecar / 断点
<cache_dir>/hf                           # HF 缓存
<cache_dir>/modelscope                   # ModelScope 缓存
<tmp_dir>/<task-id>                      # 下载暂存
```

- 每个 layer 用 `org.olaya.path` / `org.olaya.role` annotation 记录原路径与角色；config 是 `ModelConfig`。
- `FileRole` 含 `weights/tokenizer/calibration/decision/config/pickle/other`；`decision.json` 可由 family 读取，缺省时使用 family 内置布局。
- `von` 的权重是 PyTorch `torch.save` 文件（ZIP + pickle）。`olaya-candle::torch` 只解释 state dict 所需的 pickle 子集，并按 storage 直接读取张量字节；不执行 pickle 代码。manifest 用 `role: pickle` 与 `format: torch` 声明。
- 清单同时以 `<name>.json` 和 `sha256-<hex>.json` 保存；`Store::get("name@sha256:<hex>")` 会先校验清单 digest。
- `StoredModel::to_repo_manifest()` 从 OCI manifest + config 还原 `RepoManifest`，store 自描述、可直接分发。
- `Store::verify` 重新计算每个 blob 的 sha256 与大小。
- 默认 `~/.olaya/{models,store,cache,tmp}`；`OLAYA_MODELS_DIR` / `OLAYA_STORE_DIR` / `OLAYA_CACHE_DIR` / `OLAYA_TMP_DIR` 覆盖，优先级 flag > env > config > default。
- CLI 共享配置文件按 `./olaya.toml` → `./sys1.toml` → `~/.olaya/config.toml` 查找，`--config` / `OLAYA_CONFIG` 可覆盖；若都不存在，首次使用会向 `~/.olaya/config.toml` 写一份 feature-aware 模板（设备默认 `auto`，编译进 CUDA/Metal 且运行时可用的 accelerator 优先，否则 CPU）。`[engine]`（device/dtype/tf32/default_model）、`[serve]`（host/port/api_key/batch）、`[hub]`（目录/镜像/代理/offline）和 `[registry]`（dir/store）同时作用于 `serve`、`ask` 和模型命令，相对路径按配置文件目录解析。`olaya doctor` 输出实际加载的配置路径、`device_preference`、有效 `device` 和 `compiled_accelerators`。
- 自定义 store 通过 `Engine::builder().store(Arc::new(MyStore))` 注入；hub 与 engine 只依赖 trait。

## 8. HTTP

`/v1/*` 严格 TypeSafe：

- `POST /v1/systemone`（JSON + multipart）
- `GET /v1/models`
- 400 `{"detail": string|object}`，422 `{"detail": [ValidationError]}`，500 `{"detail": string}`

`/api/*` 模型管理（Ollaya 风格）：

- `POST /api/pull`（返回 manifest；`stream` 进度暂未实现）
- `POST /api/load` / `POST /api/unload`
- `GET /api/tags` / `GET /api/ps`
- `POST /api/show`
- `DELETE /api/delete`

`/health`、`/metrics`。鉴权：loopback 默认无；非 loopback 无 key 拒绝启动；设置 `OLAYA_API_KEY` 后 Bearer。`POST /v1/systemone` 的 JSON / multipart `assets` 通过 `DecisionRuntime::predict_with_assets` 传给 runtime（默认实现调用 `predict` 并丢弃 assets；clef 目前仍走文本路径）。

`AppState` 在 `ServeConfig::max_batch > 1` 时启动一个后台 `Batcher`（默认 32 条 / 2 ms 收集窗口）：无 assets 的 `/v1/systemone` 请求按模型聚成一个 `predict_batch`，在 `spawn_blocking` 中执行后按原顺序回传；带 assets 的请求直接走 `predict_with_assets`，不参与批处理。`--max-batch 1` 完全关闭批处理。`BatchConfig` 也可直接传给库用户自己的 `Batcher`。`Family::predict_batch` 的默认实现逐请求回退，所以跨请求共享一次 trunk 的是 kev / decision / qwen3guard；laya / nli / von / gliclass 在请求内部合批，跨请求仍逐条。实测 16 个 qwen3guard 请求：库级逐条 `1.268s`、`ask_batch_responses` `1.004s`（1.3×）；HTTP 并发下两种模式都约 `1.03s`，因为 16 个 `spawn_blocking` 已经占满 GPU，批处理主要降低调度开销而不是总 FLOPs。

`serve_contract` 用 mock `DecisionRuntime` 覆盖 health、models、metrics、systemone 成功/错误映射、Bearer 鉴权、动态 load/unload/pull/delete、box/assets danger-zone 头、multipart 的 `request` + 文件 part、以及非 loopback 无 key 拒绝启动。

## 9. Feature 矩阵

- `olaya-candle`：默认 `modernbert` + `tokenizer`；内置 family features `laya` / `nli` / `gliclass` / `von` / `decima` / `qwen3guard` / `decision` / `kev` / `clef`（分别启用所需 encoder、tokenizer 与 `torch`）；可选 `bert`、`deberta`、`qwen3`、`qwen35`（Gated DeltaNet trunk）、`torch`（PyTorch 检查点读取）、`metal`、`cuda`。
- `olaya`：默认 `laya` + `nli` + `gliclass` + `von` + `decima` + `qwen3guard` + `decision` + `kev` + `clef` + `hub`，family feature 直接转发到 `olaya-candle`；可选 `qwen35`、`all-families`、`cuda`、`metal`、`remote`、`async`。
- `olaya-cli`：默认 `serve`；`olaya` 默认带 `laya` + `nli` + `gliclass` + `von` + `decima` + `qwen3guard` + `decision` + `kev` + `clef` + `hub`；`cuda` / `metal` 转发到 `olaya`，未启用对应 feature 时 `--device cuda|metal` 返回明确错误。
- `olaya` 的 `serve` feature：默认启用 `serve-auth` + `serve-multipart` + `serve-metrics`；库用户不开启时不会拉入 axum/tokio。

### 9.1 ONNX Runtime 后端（评估结论）

`RepoManifest.engine` 目前只接受 `candle` / `remote` / `router`；没有 ONNX Runtime 后端。评估后暂不引入：所有 family 的决策头、行构造、校准与 LoRA/pointer 读取都在 candle 张量上实现，ORT 只能替代 encoder 前向，收益集中在很小的编码器；而引入 `ort` 会新增一个平台相关原生依赖、第二套 FP16/量化路径和一套与 goldens 平行的验证矩阵。当前 candle 路径在 CUDA 上已与 ONNX 参考对齐到 `≤3e-6`（ModernBERT/DeBERTa 系），并支持 F16/BF16 + 跨请求/跨问题批处理，因此把 ORT 留作后续独立 crate（`engine: "onnx"` + 可选 feature）的扩展点，而不是门面默认依赖。新增后端时只需实现 `Family`/`FamilyFactory`，manifest 的 `engine` 字段与 store 分发机制无需变化（需同步放宽 `validate()` 的枚举）。

## 10. 验证

`conformance/cases/<family>.json` 是各 family 共享的请求集（JSON state、Unicode、无 instructions、2/10 档、40 选项、ordinal、非法请求、长状态等）；`conformance/fixtures/<family>/<model>.jsonl` 每行一个请求及其上游结果（完整精度概率、精确 choice/legend，或 `invalid_request` / `state_truncated` / `input_too_long`）。`olaya-conformance` 通过公开的 engine 重放这些请求：choice/legend/答案类型精确，概率容差 decima 与 NLI 均为 `1e-5`（两条独立 fp32 栈；长 option row 实测 ≤ `1.1e-6`），kev 为 `1e-4`（合并 LoRA + pointer head，实测约 `3e-8`），qwen3guard 为 `1e-5`（transformers fp32），decision 为 `1e-4`（作者 PyTorch fp32），clef 为 `2e-2`（9B 上游只发布 BF16 权重，candle 以 fp32 流式运行；实测低置信选项约 `9e-3`）；错误按种类匹配。`fixture_requests_match_the_parser` 额外保证上游答出的请求都能通过共享 parser。当前 golden 覆盖 decima（6 个 checkpoint）、NLI（3 个：modernbert-large / modernbert-base / deberta-v3-large）、kev（15 个请求）、clef（7 个请求）、qwen3guard（9 个请求）与 decision（13 个请求）；laya / gliclass / von 仍以 smoke 中的 ORT / PyTorch / gliclass fp32 锚点为准（完整 golden 需要重跑 ollaya 的 ORT bundle）。`gliclass-instruct-edge` 的 32M ModernBERT 路径在 `gliclass_smoke` 中对 `gliclass` 0.1.22 fp32 前向验证（≤ `4e-7`）；同时修复了 ModernBERT `global_rope_theta`/`local_rope_theta` 标量配置的读取（此前缺失时统一回退到 `160000/10000`，会把 sliding 层的 RoPE 基数算错）。

goldens 由 `conformance/scripts/generate_goldens.py` 生成：decima 用 decima 2.3.0 的 PyTorch fp32 `decima.model.Decima`，NLI 用 transformers 5.17 fp32 的 `AutoModelForSequenceClassification`，两者都按 olaya 的 row 构造重放；clef 由 `conformance/scripts/clef_reference.py` 在 meta 设备上加载 mmap 权重，只取 `lm_head` 的 option token 行。qwen3guard 与 decision 的 fixture 来自已有上游 fp32 结果。命令见 `conformance/scripts/` 与各 crate 的测试文件。Laya 的 ORT parity 仍以内联期望值保留在 `laya_parity.rs`，serve 契约测试直接解析 TypeSafe 的 422 形状。

CI 由 `.github/workflows/ci.yml` 固定为 `fmt`、`clippy -D warnings`（含 `olaya-sdk --all-features`）、`test --workspace`、`olaya-sdk --all-features` 测试、serve 契约、catalog 生成校验、feature 矩阵（`all-families`、各 family 单开与 SDK）和 `cargo doc`；macOS job 额外 `cargo check --features metal`；`cargo run -p xtask -- check` 在本地跑同一组检查。发布只由 `.github/workflows/release-plz.yml`（`release-plz.toml`）驱动：`release-pr` 维护版本与 changelog，`release` 发布尚未出现在 crates.io 的版本、创建 per-package tag/GitHub Release，并在同一 job 中生成和上传 `registry-catalog.json`；手动创建的 GitHub Release 仍由 `release.yml` 补传。不再提供独立的 `cargo publish` 封装，避免绕过 tag/release/catalog 产生不一致状态。

`crates/olaya/tests/dtype_parity.rs` 在同一 CUDA/Metal 设备上分别以 F32 与 F16/BF16 构建 engine，逐请求比较答案：ModernBERT 族与 qwen3guard 的半精度概率差在 `1e-3` 量级，BERT / DeBERTa 则精确到 0（仍走 F32）。`crates/olaya/tests/bench.rs` 通过 `OLAYA_BENCH_DTYPE` / `OLAYA_BENCH_TF32` 复用同一套计时逻辑。

性能实现：kev / decision 的多问题 row 以 `batch * width ≤ 32k` 分批，只做一次 Qwen3.5 前向；GPU 全注意力按 query 分块（score budget 8M 元素），避免 `[seq, seq]` 在长上下文下 OOM；`Qwen35StreamingEncoder` 复用文件句柄并按字节预算缓存前缀层，clef 在 CPU 上默认 4 GiB layer cache（`OLAYA_CLEF_LAYER_CACHE_MB` 可调，CUDA 默认 0）；`tokenizers` 关闭 `esaxx_fast` 的 C++ 实现，消除 MSVC 静态 CRT 与 CUDA kernels 动态 CRT 的链接冲突。`crates/olaya/tests/bench.rs` 输出加载时间与 warm inference 时间。

精度策略：`EngineBuilder::dtype` 在 CPU 上被规范化为 F32（candle 的 CPU half 内核会先拓宽到 F32，反而更慢），CUDA / Metal 上按请求使用 F16 / BF16。ModernBERT（laya、nli-modernbert、von、gliclass-instruct-edge、decima 的 modernbert 变体）与 Qwen3（qwen3guard）的整条 encoder 走半精度；laya head、NLI classifier、von / decima / gliclass scorer 等决策头保持 F32，只在 encoder 输出处做一次 dtype 转换，避免小头部累积误差。`encoders::modernbert` 内置了一份 dtype 安全的 forward：上游 candle-transformers 把 4-D attention mask 写成 F32，无法直接跑半精度；同时它读取 `global_rope_theta` / `local_rope_theta` 标量与 `rope_parameters` 两种配置（`gliclass-instruct-edge` 的 sliding 层使用 `local_rope_theta=160000`）。DeBERTa 系（nli-deberta-v3、gliclass-v4-edge/large）与 Qwen3.5 流式 trunk（kev / decision / clef）仍为 F32：DeBERTa 的 `XSoftmax` 用 `f32::MIN` 构造 mask，半精度下会变成 `0 * -inf = NaN`；Qwen3.5 的 chunked gated-delta rule 依赖 F32 归约。candle 的 BERT（decima-small）有同样的 mask 问题，也保留 F32。CUDA 上 `--tf32` 通过 `set_gemm_reduced_precision_f32(true)` 让 F32 GEMM 使用 TF32 tensor core，属于进程级开关，与 F16/BF16 互不影响。

`qwen3guard` 由 [`qwen3guard_smoke`](../crates/olaya/tests/qwen3guard_smoke.rs) 覆盖：7 个 probe 与上游 transformers fp32（同一 `decision.json` rows、tokenizer 与候选 token）对比，最大概率差 `5.7e-6`；另外验证 preset 子集、非 preset 问题拒绝、以及超过 32k 的输入返回 `InputTooLong`。

`encoders::qwen35` 由 [`qwen35_parity`](../crates/olaya-candle/tests/qwen35_parity.rs) 覆盖：仓库内置一个同构的微型 checkpoint（`tests/fixtures/qwen35-tiny`，3 层 GDN + 1 层全注意力、2 组 value heads、70-token 多 chunk 输入），与 transformers fp32 对比 hidden / candidate logit ≤ `1e-6`；设置 `QWEN35_BASE_DIR` 可切换到真实 Qwen3.5-0.8B-Base。

`decision` 由 [`decision_smoke`](../crates/olaya/tests/decision_smoke.rs) 覆盖：8 个请求（choice / noul / score、JSON state、Unicode、40 选项）与作者 `decision/model.py` 的 PyTorch fp32 参考对比，token 数、答案类型、choice/legend 精确，概率 `≤1e-4`；另验证缺失 instructions、1 档 score 与超过 16k token 的 row 被拒绝。

`kev` 由 [`kev_smoke`](../crates/olaya/tests/kev_smoke.rs) 覆盖：10 个请求（string / JSON / array state、choice / noul / score、多问题、Unicode、delimiter 转义）与作者 `kev.api` / `kev.model` 的 PyTorch fp32 参考对比（Qwen3.5-0.8B-Base、手动合并 PEFT LoRA、`head.pt`），token 数、choice 标签、noul / score 值、legend 精确，概率 `≤1e-4`（实测约 `3e-8`）；另验证 box question 被拒绝、超过 8192 token 的 state 返回 `STATE_TRUNCATED`、row 超限返回 `InputTooLong`。

`clef` 由三层测试覆盖：[`clef_smoke`](../crates/olaya/tests/clef_smoke.rs) 在真实 9B checkpoint 上重放 choice/score/noul 与 JSON state，token 数与 BF16 上游参考一致、概率差 `≤5e-3`；`JointSchemaHead` 的微型 fixture 对单问与三问（choice/score/noul）与作者 PyTorch 实现对比，概率差 `≤3e-8`；`encode_record` 的 12 个用例逐 token 对比 input ids 与 question/option span。9B 上游只提供 BF16 权重，candle 以 fp32 流式运行，因此 conformance goldens 使用 `2e-2` 容差；box、超长 state / schema 分别返回 `invalid_request` / `STATE_TRUNCATED` / `InputTooLong`。图像与视频输入尚未进入 Rust 原语，clef 当前走文本 System One 路径。

## 11. 里程碑

1. v1：laya 三 checkpoint + 全链路 + 扩展 SDK。
2. v1.1（已实现）：laya 脚本/语言 router；`laya-latest` 按 `strategy: "script"` 路由到 `laya` / `laya-multilingual`。同期落地 OCI 内容寻址 store、digest 钉版本，以及可替换的 `ManifestRegistry`/`ManifestStore` trait。
3. v1.2（进行中）：ModernBERT/DeBERTa/BERT 系。`nli` 的 ModernBERT 与 DeBERTa-v3 变体已完成（多行 pair、模板、`only_first` 截断），`gliclass-v4-edge`（first-token 池化 + v4 calibrator）、`gliclass-large`（v1.0 的 segment embeddings + average label pooling）、`von`（option-marker、Python `str()` 状态、zero-shot noul debias、输入条件温度）与 `decima-small`（BERT/e5 + late-interaction scorer + cumulative-link ordinal head）已完成；`gliclass-large` 把 3 行 segment embedding 折叠进扩展的 word-embedding 表，`von` 通过 `olaya-candle::torch` 直接读取 PyTorch 检查点。与 ollaya ONNX / 上游 PyTorch 的 smoke 请求概率差分别 ≤ `3.3e-6` / `6.2e-7` / `1.3e-6` / `8.4e-7` / `1.4e-6` / `2.3e-7` / `4.5e-7` / `5.3e-7`（NLI ModernBERT / NLI DeBERTa / GLiClass v4-edge / GLiClass large / von / decima-small / decima-base / decima-guard-small），decima 其余三个 checkpoint（agent / guard / base-clean）也已完成，六个变体的 ordinal choice 概率差 ≤ `4.1e-7`。golden 集已覆盖 decima（6 个 checkpoint）与 NLI（2 个）；下一步是 GLiClass / Von / Laya 的 golden 与剩余 family。
4. v1.3（进行中）：Qwen3 系。`qwen3guard-gen-0.6b` 已实现（Qwen3-0.6B trunk + 固定 preset 的 13 个候选 token，preset-only 请求校验，上游 transformers fp32 最大概率差 `5.7e-6`）；Qwen3.5 的 `encoders::qwen35`（Gated DeltaNet + gated attention）已实现，并通过 transformers fp32 微型 checkpoint 的 hidden/logit parity（`≤1e-6`）与真实 0.8B-Base（hidden `≤8.4e-5`）；`decision-eos-0.8b` 已完成（`decision-endpoint-v1` 行构造、candidate-endpoint + global-query head、作者 PyTorch fp32 parity，token 数精确、概率 `≤1e-4`）；`kev-0.8b` 也已完成（`kev-pointer-v1` 行构造、PEFT LoRA 合并、pointer head、`head.pt` 温度校准，作者 PyTorch fp32 parity，概率 `≤1e-4`）；`clef-flash` 已完成（9B 逐层流式 trunk、joint schema head、packed schema 行编码，BF16 上游 parity，概率 `≤2e-2`）。
5. v1.4（0.8B 已实现）：GDN + kev 1.0；`kev-0.8b` 已接入，更大 checkpoint 复用同一 family。
6. v1.5（文本路径已实现）：`clef-flash` 已接入（9B 流式 trunk + joint schema head）；图像/视频输入留待原语扩展。
7. v1.6（性能/精度）：serve 跨请求 `Batcher` 与 `Family::predict_batch`；CUDA F16/BF16 与 TF32；laya / von / gliclass 请求内 row 合批；新增 `nli-modernbert-base`（14 个 transformers fp32 goldens）与 `gliclass-instruct-edge`（32M ModernBERT `ettin-encoder-32m`，GLiClass 0.1.22 fp32 参考 ≤4e-7，CUDA F16/BF16 支持）。catalog 共 21 个 manifest。CUDA 20 问 F32：laya `0.168s`、von `0.177s`、gliclass `0.198s`；F16 下 laya `0.106s`、von `0.102s`。
8. v1.7（可观测性）：`tracing` span/event（engine 配置、load、ask、batch、HTTP、encoder 加载、Qwen3.5 流式层）；`RuntimeMetrics` 原子计数 + 固定桶直方图 + Prometheus 渲染；`DecisionRuntime::runtime_metrics()` 让 serve 与 engine 共用 handle；`x-typesafe-request-id` 响应头与日志关联；CLI 日志固定到 stderr，stdout 保持纯 JSON。
9. v1.8（分发）：`cargo run -p xtask -- sync-registry` 从 `registry/` 生成 `registry-catalog.json`（不编译进二进制）；`RemoteRegistry` 首次使用/导出时拉取 `OLAYA_REGISTRY_URL` 并缓存，网络失败回退缓存；CLI 在默认 registry 缺失/为空时使用远端 catalog，`olaya models export [--dir] [--force]` 按原布局物化清单；workspace 测试验证 bundle 构建。
10. v1.9（客户端 SDK）：新增独立 `olaya-sdk`，实现官方 TypeSafe wire format；类型化 question/answer、`{"detail": ...}` 错误体（含 422 字段路径）、重试/退避/`Retry-After`/总预算、request id、multipart 与 inline base64 图片、`extra_body`、未知答案保留；异步默认，`blocking`/`gzip` 可选。
11. v1.10（发布资产）：`registry-catalog.json` 不再提交到仓库；`cargo run -p xtask -- sync-registry [--output PATH]` 默认写入 `target/`；GitHub Release 发布后 `release.yml` 从对应 tag 生成并上传为该 release 的 `registry-catalog.json` 资产；默认 `OLAYA_REGISTRY_URL` 指向 latest release asset，预发布/固定版本用版本化 URL；CI 每次运行都执行一次 catalog 生成校验。
12. v1.11（发布自动化）：引入 release-plz（`release-plz.toml` + `.github/workflows/release-plz.yml`）；release PR 维护 workspace version 与各 crate changelog，合并后发布五个 publishable crate、创建 per-package tag/GitHub Release；release job 在同一 job 中生成并上传 `registry-catalog.json`，绕开 `GITHUB_TOKEN` 不触发其他 workflow 的限制；`xtask` 与 `olaya-conformance` 排除；release-plz 是唯一发布路径，避免单独 `cargo publish` 绕过 tag/release/catalog。

> `laya-latest` 的 router 配置：`{ "strategy": "script", "routes": { "english": "laya", "multilingual": "laya-multilingual" }, "default": "laya" }`。Latin 占比超过 90% 走 English，否则走 multilingual。

## 12. 客户端 SDK（`olaya-sdk`）

`olaya-sdk` 是仓库内唯一不依赖其他内部 crate 的包，只共享 workspace 的第三方依赖声明。它直接实现 TypeSafe OpenAPI 的 `POST /v1/systemone` 与 `GET /v1/models`，因此可用于官方 API、兼容网关或本地 `olaya serve`。

- 请求模型：`NoulQuestion / ChoiceQuestion / ScoreQuestion / BoxQuestion`、`SystemOneRequest`、`Asset`、`Content`、`State`；`questions!` 宏生成有序 `Questions`。
- 响应模型：`SystemOneResponse { model, answers, usage, request_id }`、四类 `*Answer`；未知 `type` 进入 `Answer::Unknown(Value)`，未知字段忽略。
- 错误：`Error::{MissingApiKey, InvalidApiKey, Config, InvalidRequest, Image, Io, Transport, Timeout, Api, Decode}`；`ApiError` 携带 status、`ErrorBody`、request id、`Retry-After`、attempts，`ValidationError::field_path()` 还原 422 字段路径。
- 重试：默认 `max_retries=2`、`backoff_initial=500ms`、`backoff_max=5s`、`jitter=0.25`、重试 408/429/5xx、`Retry-After`/`retry-after-ms`、30 s 总预算；单请求默认 10 s 超时。
- 图片：`Image` 从字节/文件/base64 构造并检测 PNG/JPEG；字节或文件走 multipart（`part` 引用），base64 走 JSON；有资产或 box 问题时自动加 `TypeSafe-Danger-Zone-Images: true`。
- 传输：默认 reqwest async + rustls；`blocking` feature 提供 `reqwest::blocking` 实现；`gzip` feature 对 ≥1 KiB 的 JSON body 启用 gzip。
- 配置：`TYPESAFE_API_KEY` / `TYPESAFE_BASE_URL` / `TYPESAFE_DEFAULT_MODEL`，显式值优先且显式空值报错；`allow_unauthenticated()` + `without_api_key()` 用于本地无鉴权服务；base URL 支持路径前缀；请求/响应摘要与重试通过 `tracing` 记录，不记录 body 与凭据。

## 13. 发布自动化

- `release-plz.toml`：workspace 级默认打开 changelog、semver check 与 GitHub release；`xtask` 和 `olaya-conformance` 设置 `release = false` / `publish = false`。所有 publishable crate 继承 `[workspace.package].version`，release-plz 同步更新它们。
- `.github/workflows/release-plz.yml`：push 到 `main` / `master` 时并行运行 `release-pr` 和 `release`。release job 使用 `contents: write` + `pull-requests: read` + `id-token: write`；PR job 使用 `contents: write` + `pull-requests: write`，并带不可取消的 concurrency；release job 不设置可取消 concurrency。
- 发布顺序由 release-plz 解析 path 依赖决定，效果等价于 `olaya-sdk → olaya-core → olaya-candle → olaya → olaya-cli`，无需维护固定列表。
- release 成功后，同一 job 运行 `cargo run -p xtask -- sync-registry --output "$RUNNER_TEMP/registry-catalog.json"`，解析 release-plz 的 `releases` 输出，对每个 tag 执行 `gh release upload --clobber`。这样即使 tag/release 由 `GITHUB_TOKEN` 创建、不会触发其他 workflow，catalog 仍会随 release 提供。
- 手动创建的 GitHub Release 由 `release.yml`（`release: published`）补传；与 release-plz 的自动上传重复时 `--clobber` 幂等。
- 当前设置 `release_always = false`，仅在 release PR 合并后发布；首次推送只创建 release PR，不会立即发布 0.1.0。release PR 应使用非 squash merge（GitHub-only）。如需每次 push 都发布 `Cargo.toml` 中尚未出现在 crates.io 的版本，可改为 `true`。
- 首次发布使用 `CARGO_REGISTRY_TOKEN` secret：crates.io 不允许通过 trusted publishing 创建新 crate。release job 在仓库变量 `CRATES_IO_TRUSTED_PUBLISHING` 未设置或不为 `true` 时走该路径；首次发布成功后，在 crates.io 为每个 crate 配置 trusted publisher（owner `hushink`、repo `olaya`、workflow `release-plz.yml`、environment 留空），把该变量设为 `true`，然后删除 secret。
- 后续发布走 GitHub OIDC：release job 的 `id-token: write` 让 release-plz 换取 crates.io 短时 token；`release-pr` job 不注入 registry token。GitHub Release 由 workflow 内 `GITHUB_TOKEN` 创建，不依赖 tag push 事件。
