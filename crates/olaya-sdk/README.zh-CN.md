# olaya-sdk

[English](README.md) | 简体中文

面向 TypeSafe/Jev **System One** API 的类型化 Rust 客户端，参考
[官方 Python SDK](https://docs.typesafe.ai/sdk/python) 以及
[System One](https://docs.typesafe.ai/api/system-one) / [List models](https://docs.typesafe.ai/api/list-models)
接口。

`olaya-sdk` 不依赖 `olaya`、`olaya-core`、`olaya-candle` 或本工作区的其他 crate。它直接实现官方
wire format，可连接：

- `https://api.typesafe.ai`
- TypeSafe 兼容网关
- 本地 `olaya serve` 实例

## 安装

```sh
cargo add olaya-sdk
```

可选 feature：

```sh
cargo add olaya-sdk --features blocking   # 同步客户端
cargo add olaya-sdk --features gzip       # gzip 请求体压缩
```

## 快速开始

设置 `TYPESAFE_API_KEY` 后：

```rust
use olaya_sdk::{ChoiceQuestion, Client, NoulQuestion, ScoreQuestion, questions};

#[tokio::main]
async fn main() -> Result<(), olaya_sdk::Error> {
    let client = Client::from_env()?;
    let response = client
        .system_one(
            "I was charged twice for my subscription this month. Please refund it.",
            questions! {
                "billing" => NoulQuestion::new("Is this message about billing?"),
                "tone" => ChoiceQuestion::new([
                    ("calm", None),
                    ("frustrated", None),
                    ("angry", None),
                ]),
                "urgency" => ScoreQuestion::new([
                    "Can wait",
                    "Needs attention this week",
                    "Needs attention today",
                ]),
            },
        )
        .await?;

    println!("billing: {}", response.noul("billing").expect("answer").noul);
    println!("tone: {}", response.choice("tone").expect("answer").choice);
    println!("urgency: {}", response.score("urgency").expect("answer").score);
    println!("request id: {:?}", response.request_id);
    Ok(())
}
```

`Client::system_one(...)` 返回 call builder，可以直接 `.await`，也可以调用 `.send()`。单次调用
参数在发送前链式设置：

```rust
let response = client
    .system_one(state, questions)
    .model("jev-preview")
    .timeout(std::time::Duration::from_secs(5))
    .extra_body("beam_width", 4)
    .await?;
```

## 问题类型

| 类型 | 构造方式 | 答案访问 |
| --- | --- | --- |
| 是/否 | `NoulQuestion::new("Is this spam?")` | `response.noul("name")` |
| Choice | `ChoiceQuestion::new([("calm", None), ("angry", None)])` | `response.choice("name")` |
| Score | `ScoreQuestion::new(["low", "medium", "high"])` | `response.score("name")` |
| Box | `BoxQuestion::new("The dog")` | `response.box_answer("name")` |

四种问题可以混在同一个请求中。`instructions` 和 criteria 描述接受 `Content`：字符串、JSON 对象、
数组或 `null`。字符串、map 和数组实现了 `From`；`serde_json::Value` 通过 `Content::try_from`
转换（顶层数字和布尔值会被拒绝）。

也支持按类型遍历：

```rust
for (name, answer) in response.nouls() {
    println!("{name}: {}", answer.noul);
}
for (name, answer) in response.choices() {
    println!("{name}: {} ({:.2})", answer.choice, answer.confidence);
}
```

Score 答案会同时返回评分档位和各档位概率：

```rust
let urgency = response.score("urgency").expect("answer");
println!("score: {}", urgency.score);
println!("p2: {:?}", urgency.probability(2));
for (level, description) in &urgency.legend {
    println!("{level}: {description:?}");
}
```

## 模型

```rust
let models = client.list_models().await?;
for model in &models.models {
    println!("{} — {}", model.name, model.description);
}
```

也可以直接使用 models resource：

```rust
let models = client.models().list().await?;
```

## 错误

```rust
use olaya_sdk::{ApiErrorKind, Error};

match client.system_one(state, questions).await {
    Ok(response) => { /* ... */ }
    Err(Error::Api(error)) => {
        eprintln!(
            "{} {}: {:?} (request id {:?})",
            error.status.as_u16(),
            error.kind().as_str(),
            error.message(),
            error.request_id
        );
        if error.kind() == ApiErrorKind::UnprocessableEntity {
            for validation in error.validation_errors().unwrap_or_default() {
                eprintln!("{}: {}", validation.field_path(), validation.msg);
            }
        }
        if let Some(delay) = error.retry_after {
            eprintln!("server asks to wait {delay:?}");
        }
    }
    Err(error) => eprintln!("{error}"),
}
```

- `Error::Api` 携带状态码、解析后的 `{"detail": ...}`、`x-typesafe-request-id`、服务端建议的
  重试延迟以及尝试次数。
- `Error::Timeout` 区分请求超时与其他传输错误。
- `Error::Decode` 在响应格式错误时报告字段路径。
- `Error::is_retryable(&policy)` 和 `Error::request_id()` 可用于自定义处理。

## 重试与超时

默认值与官方 SDK 一致：重试 2 次，500 ms 到 5 s 的指数退避和 25% 抖动，重试 408/429/5xx，支持
`Retry-After`，每次调用总预算 30 s。每个请求默认超时 10 s。服务器给出的 `Retry-After` 会被限制在
60 秒（`MAX_RETRY_AFTER`），避免恶意服务器无限拖延客户端。响应体上限为 16 MiB
（`MAX_RESPONSE_BYTES`），超过时返回 `Error::ResponseTooLarge`。

```rust
use std::time::Duration;
use olaya_sdk::{Client, RetryPolicy};

let client = Client::builder()
    .api_key("...")
    .retry(
        RetryPolicy::default()
            .max_retries(4)
            .backoff_initial(Duration::from_millis(200))
            .backoff_max(Duration::from_secs(2))
            .deadline(Some(Duration::from_secs(15))),
    )
    .timeout(Duration::from_secs(5))
    .build()?;
```

单次调用覆盖：

```rust
client
    .system_one(state, questions)
    .retry(RetryPolicy::disabled())
    .timeout(Duration::from_secs(30))
    .await?;
```

在 builder 或单次调用上使用 `.no_timeout()` 可以关闭单请求超时。`RetryPolicy` 实现了 `Clone`，
可以复用。

## 图片

附加图片，并在 `state`、instructions 或 criteria 中用命名占位符引用。字节或文件形式的图片会作为
multipart part 发送；inline 图片作为 base64 JSON 发送。

```rust
use olaya_sdk::{Image, NoulQuestion, questions};

let image = Image::open("photo.png")?;
let response = client
    .system_one(
        "Inspect this returned product: <photo>",
        questions! {
            "damaged" => NoulQuestion::new("Is the product damaged?"),
        },
    )
    .image("<photo>", image)
    .await?;
```

使用 `.inline_image("<photo>", image)` 可以改为 base64 JSON；如果数据已在内存中，也可以使用
`Asset::image_bytes(bytes)` / `Asset::inline_base64(text)`。PNG 和 JPEG 会根据字节自动检测。

请求包含 asset 或 box 问题时会自动添加 `TypeSafe-Danger-Zone-Images: true`。可以用
`.danger_zone_images(true/false)` 覆盖；长上下文请求使用 `.danger_zone_long_context(true)`。

## 同步客户端

启用 `features = ["blocking"]` 后：

```rust
use olaya_sdk::blocking::Client;
use olaya_sdk::{NoulQuestion, questions};

let client = Client::from_env()?;
let response = client
    .system_one(
        "I was charged twice.",
        questions! {
            "billing" => NoulQuestion::new("Is this about billing?"),
        },
    )
    .send()?;
println!("{}", response.noul("billing").expect("answer").noul);
```

同步客户端不能在 Tokio runtime 中构造或使用。

## 日志

SDK 通过 `tracing` 输出请求/响应摘要（debug）和重试警告（warn），不会记录 body 或凭据。可通过
`RUST_LOG` 控制级别，例如 `RUST_LOG=olaya_sdk=debug`。

## 前向兼容

- 未知答案类型保留为 `Answer::Unknown(serde_json::Value)`，可通过 `response.unknown_answers()`
  访问，不会导致解码失败。
- 已知响应中的未知字段会被忽略。
- 额外请求字段可以用 `.extra_body("name", value)` 发送。
- 四种官方问题原语均为类型化模型，不需要原始问题字典。

## 环境变量

| 变量 | 用途 | 默认值 |
| --- | --- | --- |
| `TYPESAFE_API_KEY` | API key | 必填 |
| `TYPESAFE_BASE_URL` | API 根地址，可带路径前缀 | `https://api.typesafe.ai` |
| `TYPESAFE_DEFAULT_MODEL` | 默认模型 | `jev-latest` |

显式 builder 值优先于环境变量；显式空值会报错而不是回退。API key 会去除首尾空白；包含内部空白、
控制字符或非 ASCII 字符的 key 会在发送请求前被拒绝。

连接无需鉴权的本地服务：

```rust
let client = olaya_sdk::Client::builder()
    .base_url("http://127.0.0.1:11435")
    .allow_unauthenticated(true)
    .without_api_key() // 忽略环境中的 TYPESAFE_API_KEY
    .build()?;
```

支持带路径前缀的 base URL，例如 `https://openrouter.ai/api` 会请求
`https://openrouter.ai/api/v1/systemone`。

## 与 Python SDK 的差异

- 默认不压缩请求体。启用 `gzip` feature 和 `Compression::Gzip`；zstd 与显式 HTTP/2 开关未暴露。
- 响应始终解码为类型化的 `SystemOneResponse`。不提供自定义 `response_model` 等价物；请使用类型化
  访问器和 `Answer::Unknown`。
- Asset 使用显式命名占位符（`<photo>`），通过 `.image(...)` 或 `.inline_image(...)` 传入；不复刻
  Python SDK 的 `str(asset)` 自动引用。
- 日志使用 `tracing`（`RUST_LOG`），不使用 `TYPESAFE_LOG_LEVEL`。

## Feature

| Feature | 作用 |
| --- | --- |
| `blocking` | 增加 `olaya_sdk::blocking::Client` |
| `gzip` | 对 ≥1 KiB 的 JSON body 启用 `Compression::Gzip` |
