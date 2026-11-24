# olaya-sdk

English | [简体中文](README.zh-CN.md)

A typed Rust client for the TypeSafe/Jev **System One** API, following the
[official Python SDK](https://docs.typesafe.ai/sdk/python) and the
[System One](https://docs.typesafe.ai/api/system-one) / [List models](https://docs.typesafe.ai/api/list-models)
references.

`olaya-sdk` is independent of `olaya`, `olaya-core`, `olaya-candle`, and the rest of the workspace.
It implements the official wire format directly, so it can call:

- `https://api.typesafe.ai`
- a TypeSafe-compatible gateway
- a local `olaya serve` instance

## Install

```sh
cargo add olaya-sdk
```

Optional features:

```sh
cargo add olaya-sdk --features blocking   # synchronous client
cargo add olaya-sdk --features gzip       # gzip request-body compression
```

## Quickstart

Set `TYPESAFE_API_KEY`, then:

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

`Client::system_one(...)` returns a call builder. Await it directly or call `.send()`. Per-call
options chain before sending:

```rust
let response = client
    .system_one(state, questions)
    .model("jev-preview")
    .timeout(std::time::Duration::from_secs(5))
    .extra_body("beam_width", 4)
    .await?;
```

## Question types

| Type | Constructor | Answer accessor |
| --- | --- | --- |
| Yes/no | `NoulQuestion::new("Is this spam?")` | `response.noul("name")` |
| Choice | `ChoiceQuestion::new([("calm", None), ("angry", None)])` | `response.choice("name")` |
| Score | `ScoreQuestion::new(["low", "medium", "high"])` | `response.score("name")` |
| Box | `BoxQuestion::new("The dog")` | `response.box_answer("name")` |

All four can be mixed in one request. `instructions` and criteria descriptions accept `Content`: a
string, JSON object, array, or `null`. Strings, maps, and arrays implement `From`; a
`serde_json::Value` converts with `Content::try_from` (top-level numbers and booleans are rejected).

Typed iteration is also available:

```rust
for (name, answer) in response.nouls() {
    println!("{name}: {}", answer.noul);
}
for (name, answer) in response.choices() {
    println!("{name}: {} ({:.2})", answer.choice, answer.confidence);
}
```

Score answers expose their rubric and per-level probabilities:

```rust
let urgency = response.score("urgency").expect("answer");
println!("score: {}", urgency.score);
println!("p2: {:?}", urgency.probability(2));
for (level, description) in &urgency.legend {
    println!("{level}: {description:?}");
}
```

## Models

```rust
let models = client.list_models().await?;
for model in &models.models {
    println!("{} — {}", model.name, model.description);
}
```

Or use the models resource directly:

```rust
let models = client.models().list().await?;
```

## Errors

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

- `Error::Api` carries the status, parsed `{"detail": ...}` body, `x-typesafe-request-id`, the
  server's retry delay, and the number of attempts.
- `Error::Timeout` distinguishes request timeouts from other transport failures.
- `Error::Decode` reports the offending field path when the response is malformed.
- `Error::is_retryable(&policy)` and `Error::request_id()` are available for custom handling.

## Retries and timeouts

The defaults mirror the official SDK: two retries, exponential backoff from 500 ms to 5 s with 25%
jitter, retries on 408/429/5xx, `Retry-After` support, and a 30 s total budget per call. Each
request also has a 10 s timeout. Server-supplied `Retry-After` delays are capped at 60 s
(`MAX_RETRY_AFTER`) so a hostile server cannot stall a client indefinitely. Response bodies are
limited to 16 MiB (`MAX_RESPONSE_BYTES`); larger bodies return `Error::ResponseTooLarge`.

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

Per-call overrides:

```rust
client
    .system_one(state, questions)
    .retry(RetryPolicy::disabled())
    .timeout(Duration::from_secs(30))
    .await?;
```

Use `.no_timeout()` on the builder or a call to disable the per-request timeout. `RetryPolicy` is
`Clone` and can be shared.

## Images

Attach an image and refer to it by a named placeholder in `state`, instructions, or criteria.
Images supplied as bytes or files are sent as multipart parts; inline images are sent as base64
JSON.

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

Use `.inline_image("<photo>", image)` to send base64 JSON instead, or
`Asset::image_bytes(bytes)` / `Asset::inline_base64(text)` when the data is already in memory.
PNG and JPEG are detected from the bytes.

`TypeSafe-Danger-Zone-Images: true` is added automatically when the request contains assets or box
questions. Override it with `.danger_zone_images(true/false)`; use
`.danger_zone_long_context(true)` for long-context requests.

## Blocking client

With `features = ["blocking"]`:

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

The blocking client must not be constructed or used inside a Tokio runtime.

## Logging

The SDK emits `tracing` debug events for request and response summaries and warning events for
retries. Bodies and credentials are never logged. Configure the level with `RUST_LOG`, for example
`RUST_LOG=olaya_sdk=debug`.

## Forward compatibility

- Unknown answer kinds are preserved as `Answer::Unknown(serde_json::Value)` and appear in
  `response.unknown_answers()`; they do not fail decoding.
- Unknown fields in known responses are ignored.
- Extra request fields can be sent with `.extra_body("name", value)`.
- All four official question primitives are typed; raw question dictionaries are not required.

## Environment

| Variable | Purpose | Default |
| --- | --- | --- |
| `TYPESAFE_API_KEY` | API key | required |
| `TYPESAFE_BASE_URL` | API root, including an optional path prefix | `https://api.typesafe.ai` |
| `TYPESAFE_DEFAULT_MODEL` | Default model | `jev-latest` |

Explicit builder values take precedence over the environment; explicitly empty values are rejected
rather than falling back. API keys are trimmed, and keys with internal whitespace, control
characters, or non-ASCII characters are rejected before any request is sent.

For a local server without authentication:

```rust
let client = olaya_sdk::Client::builder()
    .base_url("http://127.0.0.1:11435")
    .allow_unauthenticated(true)
    .without_api_key() // ignore TYPESAFE_API_KEY from the environment
    .build()?;
```

Base URLs with a path prefix are supported, for example `https://openrouter.ai/api` becomes
`https://openrouter.ai/api/v1/systemone`.

## Differences from the Python SDK

- Request bodies are uncompressed by default. Enable the `gzip` feature and `Compression::Gzip`;
  zstd and explicit HTTP/2 toggles are not exposed.
- Responses always decode into the typed `SystemOneResponse`. A custom `response_model`
  equivalent is not provided; use the typed accessors and `Answer::Unknown` instead.
- Assets use explicit named placeholders (`<photo>`) passed through `.image(...)` or
  `.inline_image(...)`. The Python SDK's automatic `str(asset)` references are not reproduced.
- Logging uses `tracing` (`RUST_LOG`), not `TYPESAFE_LOG_LEVEL`.

## Feature flags

| Feature | Effect |
| --- | --- |
| `blocking` | Adds `olaya_sdk::blocking::Client` |
| `gzip` | Enables `Compression::Gzip` for JSON bodies of at least 1 KiB |
