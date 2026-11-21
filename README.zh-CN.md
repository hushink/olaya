# olaya

[English](README.md) | 简体中文

**基于原生 Rust 的 System One 决策模型运行时。**

olaya 将文本或结构化状态转换为经过校准的类型化决策——概率、选项、分数或边界框——而不生成文本。
它同时提供可嵌入的库、TypeSafe 兼容的 HTTP 服务、命令行工具和独立的客户端 SDK。

> [!NOTE]
> 模型通过 [candle](https://github.com/huggingface/candle) 运行，不依赖 ONNX Runtime。

## 亮点

- **类型化决策原语** — noul（是/否）、choice、score 和 box，以及经过校准的答案。
- **模型目录** — Laya、NLI、GLiClass、Von、Decima、Qwen3Guard、Decision、Kev 和 Clef 系列，
  每个模型都由 manifest 描述，并固定 revision 与文件哈希。
- **可嵌入** — 精简的 engine API，以及可替换的 `ManifestRegistry` 和 `ManifestStore` trait。
- **TypeSafe 兼容服务** — `POST /v1/systemone`、`GET /v1/models`，以及 `/api/*` 下的动态模型管理，
  支持鉴权、multipart 图片、请求批处理和指标。
- **独立客户端 SDK** — [`olaya-sdk`](crates/olaya-sdk/README.zh-CN.md) 面向官方 TypeSafe/Jev API，
  提供类型化请求、错误、重试和图片支持。
- **硬件支持** — CPU、CUDA 和 Metal；F32、F16、BF16，以及可选的 TF32。
- **可观测性** — tracing、Prometheus 指标，以及 `x-typesafe-request-id` 请求关联。

## 环境要求

- 从源码构建需要 Rust 1.92 或更新版本。
- GPU 可选。CPU 推理使用 F32；CUDA 和 Metal 可使用 F16/BF16。

## 快速开始

### 安装 CLI

```sh
cargo install olaya-cli

# 或从代码 checkout 安装
cargo install --path crates/olaya-cli
```

GPU 构建需要增加对应 feature：

```sh
cargo install olaya-cli --features cuda
cargo install olaya-cli --features metal
```

Windows 上 `nvcc` 需要能找到 MSVC 的 `cl.exe`。请在 **x64 Native Tools Command Prompt** 中运行
CUDA 安装命令，或把 `NVCC_CCBIN` 设置为 `cl.exe` 的完整路径。

### 拉取模型并提问

```sh
olaya models list
olaya models pull laya
```

创建 `questions.json`：

```json
{
  "billing": {
    "type": "noul",
    "instructions": "Is this message about billing?"
  },
  "tone": {
    "type": "choice",
    "instructions": "What is the tone of this message?",
    "criteria": {
      "calm": "A neutral or polite message",
      "angry": "An upset or hostile message"
    }
  },
  "urgency": {
    "type": "score",
    "instructions": "How urgent is this message?",
    "criteria": ["Can wait", "Needs attention this week", "Needs attention today"]
  }
}
```

```sh
olaya ask --model laya \
  --state "I was charged twice for my subscription this month." \
  --questions questions.json
```

CLI 会以 JSON 输出答案，并以问题名作为 key。

> [!NOTE]
> `olaya models pull` 会将权重及相关文件下载到 `~/.olaya/models`；拉取后的首次请求会将模型加载到内存。

### 启动 TypeSafe 兼容服务

```sh
olaya serve --host 127.0.0.1 --port 11435
```

```sh
curl http://127.0.0.1:11435/v1/models

curl http://127.0.0.1:11435/v1/systemone \
  -H "Content-Type: application/json" \
  --data-binary @request.json
```

`request.json`：

```json
{
  "model": "laya",
  "state": "I was charged twice for my subscription this month.",
  "questions": {
    "billing": {
      "type": "noul",
      "instructions": "Is this message about billing?"
    },
    "tone": {
      "type": "choice",
      "instructions": "What is the tone of this message?",
      "criteria": {
        "calm": "A neutral or polite message",
        "angry": "An upset or hostile message"
      }
    }
  }
}
```

常用服务参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--host` | `127.0.0.1` | 绑定地址 |
| `--port` | `11435` | HTTP 端口 |
| `--device` | `cpu` | `cpu`、`cuda` 或 `metal` |
| `--dtype` | `f32` | CUDA/Metal 上的 `f32`、`f16` 或 `bf16` |
| `--tf32` | 关闭 | 允许 CUDA 使用 TF32 tensor-core GEMM |
| `--max-batch` | `32` | 单次前向收集的请求数 |
| `--batch-wait-ms` | `2` | 首个请求的等待窗口 |
| `--api-key` | `OLAYA_API_KEY` | 除 `/health` 外所有路由的 Bearer token |

> [!WARNING]
> 服务默认只绑定 loopback。绑定非 loopback 地址时必须提供 `--api-key`（或显式使用
> `--allow-unauthenticated`）。服务本身使用 HTTP；在可信网络之外部署时，请通过反向代理终止 TLS。

### 作为 Rust 库使用

```toml
[dependencies]
olaya = "0.1"
```

```rust
use olaya::{BuiltinFamilies, Engine, Questions, State, load_manifests};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 先用 `olaya models export --dir registry` 物化本地目录，
    // 或指向任意经过校验的模型 manifest 目录。
    let manifests = load_manifests("registry")?;

    let engine = Engine::builder()
        .builtin_families(BuiltinFamilies::all())
        .manifests(manifests)
        .build()?;

    let questions = Questions::new()
        .noul("billing", "Is this message about billing?")
        .choice_with_instructions(
            "tone",
            "What is the tone of this message?",
            [("calm", "A neutral or polite message"), ("angry", "An upset or hostile message")],
        )
        .score_with_instructions(
            "urgency",
            "How urgent is this message?",
            ["Can wait", "Needs attention this week", "Needs attention today"],
        );

    let response = engine.ask(
        "laya",
        &State::text("I was charged twice for my subscription this month."),
        &questions,
    )?;

    println!("{:?}", response.answers.get("billing"));
    Ok(())
}
```

engine 还提供 `ask_typed`（返回类型化答案集）、`ask_batch` / `ask_batch_responses`（共享一次前向），
以及显式的 `load` / `unload`。

### 使用客户端 SDK

[`olaya-sdk`](crates/olaya-sdk/README.zh-CN.md) 是面向官方 TypeSafe/Jev API 的独立 crate，可连接
`https://api.typesafe.ai`、兼容网关或本地 `olaya serve`。

```toml
[dependencies]
olaya-sdk = "0.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use olaya_sdk::{ChoiceQuestion, Client, NoulQuestion, questions};

#[tokio::main]
async fn main() -> Result<(), olaya_sdk::Error> {
    let client = Client::from_env()?; // TYPESAFE_API_KEY

    let response = client
        .system_one(
            "I was charged twice for my subscription this month.",
            questions! {
                "billing" => NoulQuestion::new("Is this message about billing?"),
                "tone" => ChoiceQuestion::labels(["calm", "angry"]),
            },
        )
        .await?;

    println!("billing: {}", response.noul("billing").expect("answer").noul);
    Ok(())
}
```

SDK 将响应解析为类型化的 `SystemOneResponse`，保留未知答案类型，支持 multipart 与 inline 图片，
并提供可选的同步客户端（`blocking` feature）和 gzip 请求压缩（`gzip` feature）。

## 模型

目录中的每个模型由 manifest 描述：family、architecture、layout、文件角色、SHA-256、许可证、
别名以及可选的路由配置。

| 系列 | 示例模型名 | 典型用途 |
| --- | --- | --- |
| Laya | `laya`、`laya-multilingual`、`laya-typed-decisions`、`laya-latest` | 通用决策与路由 |
| NLI | `nli-modernbert`、`nli-modernbert-base`、`nli-deberta-v3` | Zero-shot NLI |
| GLiClass | `gliclass-large`、`gliclass-v4-edge`、`gliclass-instruct-edge` | 标签与选项分类 |
| Von | `von` | Option-marker NLI |
| Decima | `decima-small`、`decima-base`、`decima-base-clean`、`decima-agent`、`decima-guard`、`decima-guard-small` | Ordinal 与 late-interaction 决策 |
| Qwen3Guard | `qwen3guard` | 安全 preset 分类 |
| Decision | `decision-eos-0.8b` | Candidate endpoint 决策 |
| Kev | `kev-0.8b` | Pointer-based choice |
| Clef | `clef-flash` | 长上下文联合 schema 决策 |

所有使用模型名的地方都接受别名，例如 `laya:latest` 或 `kev:0.8b`。查看目录：

```sh
olaya models list
olaya models show laya
```

模型权重会根据 manifest 从 Hugging Face 或 ModelScope 下载。模型目录随每个 release 发布，并在本地缓存。

## 目录与存储

全新安装后，CLI 会从最新 GitHub Release 读取 catalog bundle，缓存到
`~/.olaya/cache/registry-catalog.json`，后续命令复用缓存。切换目录来源：

```sh
# 本地 manifest 目录
olaya --registry-dir ./registry models list
# 或 OLAYA_REGISTRY_DIR=./registry

# 托管的 catalog bundle
OLAYA_REGISTRY_URL=https://example.com/registry-catalog.json olaya models list
```

为离线使用或嵌入场景物化目录：

```sh
olaya models export --dir ./registry
```

如果模型通过内容寻址 store 分发，`--registry-store` 可以直接把 store 当作目录。

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| `OLAYA_HOME` | `~/.olaya` | 默认路径的根目录 |
| `OLAYA_MODELS_DIR` | `$OLAYA_HOME/models` | 物化后的模型目录 |
| `OLAYA_CACHE_DIR` | `$OLAYA_HOME/cache` | 下载、暂存和目录缓存 |
| `OLAYA_STORE_DIR` | `$OLAYA_HOME/store` | 内容寻址 OCI store |
| `OLAYA_TMP_DIR` | `$OLAYA_HOME/tmp` | 单任务下载暂存 |
| `OLAYA_REGISTRY_DIR` | `./registry`，其次 `$OLAYA_HOME/registry` | 文件系统目录 |
| `OLAYA_REGISTRY_URL` | 最新 release asset | 托管 catalog bundle |
| `HF_ENDPOINT` | Hugging Face | Hugging Face endpoint 覆盖 |
| `MODELSCOPE_ENDPOINT` | ModelScope | ModelScope endpoint 覆盖 |
| `OLAYA_PROXY` | 未设置 | 模型下载代理 |
| `OLAYA_OFFLINE` | 未设置 | 禁用网络访问 |
| `OLAYA_API_KEY` | 未设置 | `olaya serve` 的 Bearer token |

manifest 布局与 catalog 格式见 [registry/README.zh-CN.md](registry/README.zh-CN.md)。

## HTTP API

服务同时提供 TypeSafe System One API 和小型模型管理 API：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 存活检查 |
| `GET` | `/v1/models` | 列出模型名与别名 |
| `POST` | `/v1/systemone` | 询问 noul、choice、score、box 问题 |
| `GET` | `/metrics` | Prometheus 指标 |
| `GET` | `/api/tags` | 列出目录中的模型 |
| `GET` | `/api/ps` | 列出已加载模型 |
| `POST` | `/api/show` | 查看 manifest |
| `POST` | `/api/load` | 加载模型 |
| `POST` | `/api/unload` | 卸载模型 |
| `POST` | `/api/pull` | 拉取模型 |
| `DELETE` | `/api/delete` | 删除模型 |

`POST /v1/systemone` 接受 JSON 或用于图片的 `multipart/form-data`。图片与 box 请求需要
`TypeSafe-Danger-Zone-Images: true`；长上下文请求可发送
`TypeSafe-Danger-Zone-Long-Context: true`。响应包含 `x-typesafe-request-id`。

请求与响应 schema 与官方 TypeSafe API 一致：
[System One](https://docs.typesafe.ai/api/system-one) 与
[List models](https://docs.typesafe.ai/api/list-models)。

## 架构

| crate | 职责 |
| --- | --- |
| [`olaya-core`](crates/olaya-core/README.zh-CN.md) | 协议原语、校验、manifest、registry/store trait、指标 |
| [`olaya-candle`](crates/olaya-candle/README.zh-CN.md) | Candle encoder、模型 family、tokenizer、checkpoint 加载 |
| [`olaya`](crates/olaya/README.zh-CN.md) | Engine、catalog、hub、OCI store、serve feature |
| [`olaya-cli`](crates/olaya-cli/README.zh-CN.md) | `olaya` 二进制 |
| [`olaya-sdk`](crates/olaya-sdk/README.zh-CN.md) | 独立的 TypeSafe/Jev 客户端 |

```text
olaya-sdk ── HTTP ──┐
                    ├──> olaya serve ──> Engine ──> olaya-candle families
olaya-cli ──────────┘
olaya-core: protocol, manifests, registry/store contracts, metrics
```

crate 图、请求生命周期、批处理、存储和精度策略见 [docs/design.md](docs/design.md)。

## 日志与指标

通过 `RUST_LOG` 控制 tracing：

```sh
RUST_LOG=olaya=debug,olaya_candle=debug olaya serve
```

`GET /metrics` 以 Prometheus 文本格式暴露请求总数、错误、模型加载和延迟直方图。提交 issue 时请附上
`x-typesafe-request-id`。

## 文档

- [CLI 指南](crates/olaya-cli/README.zh-CN.md)
- [客户端 SDK 指南](crates/olaya-sdk/README.zh-CN.md)
- [模型目录与 manifest](registry/README.zh-CN.md)
- [设计说明](docs/design.md)
- [TypeSafe System One API](https://docs.typesafe.ai/api/system-one)
