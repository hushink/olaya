# olaya-cli

[English](README.md) | 简体中文

`olaya` 命令行二进制：执行单次决策请求、管理模型目录，或启动 TypeSafe 兼容的 HTTP 服务。

## 安装

```sh
cargo install olaya-cli

# 或从代码 checkout 安装
cargo install --path crates/olaya-cli

# GPU 构建
cargo install olaya-cli --features cuda
cargo install olaya-cli --features metal
```

Windows 上请在 **x64 Native Tools Command Prompt** 中运行 CUDA 安装命令，或把 `NVCC_CCBIN`
设置为 `cl.exe` 的完整路径；否则 `nvcc` 会报 `Cannot find compiler 'cl.exe' in PATH`。

## 命令

| 命令 | 用途 |
| --- | --- |
| `olaya ask` | 执行单次决策请求 |
| `olaya serve` | 启动 HTTP 服务 |
| `olaya models list/show/pull` | 查看或下载模型 |
| `olaya models export` | 将目录物化到 registry 目录 |
| `olaya models add/remove` | 在文件系统 registry 中写入或删除 manifest；`remove --purge` 同时删除本地权重和 blob |
| `olaya models stored/manifest/verify` | 查看 OCI 内容存储 |
| `olaya doctor` | 输出路径、目录来源和诊断信息 |

常用调用：

```sh
# 单次请求
olaya ask --model laya --state "I was charged twice." --questions questions.json

# 从 stdin 读取 state
printf '%s' '{"message":"hello"}' | olaya ask --model laya --state - --questions questions.json

# 拉取 router 及其全部目标
olaya models pull laya:latest

# 查看生效配置、存储和目录来源
olaya doctor

# 本地/离线 catalog
olaya --registry-dir ./registry models list
OLAYA_OFFLINE=true olaya models list

# 使用 OCI content store 作为 catalog
olaya --registry-store models list
```

## 配置文件

`serve`、`ask` 和模型命令共享一份 TOML 配置。CLI 依次查找 `./olaya.toml`、`./sys1.toml`、
`~/.olaya/config.toml`；可用 `--config` 或 `OLAYA_CONFIG` 指定其他文件。相对路径相对于配置文件
所在目录解析。如果不存在配置文件，CLI 会在首次使用时向 `~/.olaya/config.toml` 写入与编译
feature 相关的模板。

```toml
[engine]
# "auto" 在构建包含 CUDA 且 GPU 可用时使用 CUDA，其次 Metal，否则回退 CPU。
device = "auto"
dtype = "f32"
tf32 = false
default_model = "laya:latest"
max_loaded = 3

[serve]
host = "127.0.0.1"
port = 11435
# api_key = "..."
allow_unauthenticated = false
max_batch = 32
batch_wait_ms = 2
max_body_bytes = 8388608
max_queue = 64
request_timeout_secs = 600
max_operations = 4
# 允许调用 API 的浏览器 Origin；留空关闭 CORS。
# cors_origins = ["https://app.example.com"]
# 额外接受的 Host 头。
# allowed_hosts = ["olaya.internal"]

[hub]
# models_dir = "models"
# cache_dir = "cache"
# tmp_dir = "tmp"
# store_dir = "store"
# offline = false
hf_endpoint = "https://hf-mirror.com"
# proxy = "http://127.0.0.1:7890"
# no_proxy = "localhost,127.0.0.1"
# allow_private_network = false
download_timeout_secs = 60
connect_timeout_secs = 15
download_retries = 5

[registry]
# dir = "./registry"
# store = false
# url = "https://github.com/hushink/olaya/releases/latest/download/registry-catalog.json"
timeout_secs = 30
```

CLI 参数覆盖环境变量，环境变量覆盖配置文件。布尔参数支持显式 false
（`--tf32=false`、`OLAYA_OFFLINE=false`），可覆盖文件里的 `true`。`olaya doctor` 会输出当前加载的
配置路径和每一项生效设置。完整说明见
[docs/configuration.md](https://github.com/hushink/olaya/blob/HEAD/docs/configuration.md)。

## 提问

```sh
olaya models pull laya
olaya ask --model laya \
  --state "I was charged twice for my subscription this month." \
  --questions questions.json
```

`questions.json` 是以问题名为 key 的对象；完整示例见
[工作区 README](https://github.com/hushink/olaya/blob/HEAD/README.zh-CN.md#快速开始)。

```json
{
  "billing": {
    "type": "noul",
    "instructions": "Is this message about billing?"
  }
}
```

命令会以 JSON 输出答案。

## 服务

```sh
olaya serve --host 127.0.0.1 --port 11435
```

服务提供 `POST /v1/systemone`、`GET /v1/models`、`/health`、`/metrics`，以及 `/api/*` 下的模型管理 API。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--device` | `auto` | `auto`、`cpu`、`cuda` 或 `metal`（全局） |
| `--dtype` | `f32` | CUDA/Metal 上的 `f32`、`f16` 或 `bf16`（全局） |
| `--tf32` | 关闭 | 允许 CUDA 使用 TF32 GEMM（全局） |
| `--max-batch` | `32` | 单次批处理前向的请求数 |
| `--batch-wait-ms` | `2` | 批处理等待窗口 |
| `--max-body-bytes` | `8388608` | HTTP 请求体上限 |
| `--max-queue` | `64` | 批处理队列上限 |
| `--request-timeout-secs` | `600` | 单个请求（含读取 body）的超时秒数；`0` 关闭 |
| `--max-operations` | `4` | 同时执行的 `/api/pull` 与 `/api/load` 数量上限 |
| `--cors-origin` | 无 | 允许的浏览器 Origin；可重复或逗号分隔；留空关闭 CORS |
| `--allowed-host` | 无 | 服务端额外接受的 `Host` 值 |
| `--allow-private-network[=BOOL]` | `false` | 允许 manifest 和下载访问 loopback/内网地址（全局） |
| `--api-key` | `OLAYA_API_KEY` | 除 `/health` 外所有路由的 Bearer token |
| `--allow-unauthenticated[=BOOL]` | `false` | 允许非 loopback 无鉴权绑定 |

> [!WARNING]
> 绑定非 loopback 地址时必须提供 `--api-key`，除非显式使用 `--allow-unauthenticated`。在可信网络之外
> 部署时，请使用终止 TLS 的反向代理。

## 模型目录

全新安装后，CLI 会先把编译进二进制的清单写入 `~/.olaya/registry`，再从最新 GitHub Release
拉取 catalog bundle 并缓存到 `~/.olaya/cache/registry-catalog.json`。后续命令复用缓存；当远端
清单和缓存都不可用时，使用本地 registry 目录作为 fallback。

```sh
olaya models list
olaya models show laya
olaya models pull laya
olaya models export --dir ./registry

# 删除本地权重、store manifest 和不再被引用的 blob
olaya models remove laya --purge
```

- `--registry-dir` / `OLAYA_REGISTRY_DIR` 选择文件系统目录。
- `OLAYA_REGISTRY_URL` 选择托管的 catalog bundle。
- `--registry-store` 将内容寻址 store 当作目录。
- 拉取 `laya:latest` 这类 router 时会同时拉取所有路由目标（`laya` 和 `laya-multilingual`）；
  在终端中会用单行进度条/百分比刷新当前文件进度，重定向输出时改为周期性日志。
- Hugging Face 下载默认使用 `https://hf-mirror.com`；设置
  `HF_ENDPOINT=https://huggingface.co` 可使用官方端点。
- `models list` 默认把 alias 归到规范模型下；`--all` 输出与 API 一致的扁平列表。
- `olaya models add <manifest>` 把该 manifest 写入文件系统 registry（用 `--force` 覆盖冲突文件）；
  `olaya models remove <name-or-alias>` 删除匹配的 manifest 文件。不带 `--purge` 时两者都不会删除
  权重或 content store blob，且都要求使用文件系统 registry（`[registry].store = false`）。
- `olaya models remove <name-or-alias> --purge` 还会删除物化目录 `models_dir/<name>`、该模型的
  store manifest、registry manifest，以及不再被任何剩余模型引用的 blob，并报告回收的字节数。
  它适用于两种 catalog 来源。托管 catalog 不会被修改，因此仍在远端目录中的模型仍会出现在列表里，
  下次 pull 会重新下载。
- 隐式 `$OLAYA_HOME/registry` 会合并不与远端 catalog 冲突的本地 manifest，因此 `models add`
  新增的模型会立即生效。与远端名称/别名冲突的本地条目会被忽略（远端优先）；要覆盖远端模型请
  显式传入 `--registry-dir`。
- `olaya models export` 写出规范的 `v1/library/<family>/manifests/<name>.json` 布局，幂等；
  `--force` 覆盖内容不同的文件。

下载与存储位于 `OLAYA_HOME`（默认 `~/.olaya`）。全部目录和环境变量见
[工作区 README](https://github.com/hushink/olaya/blob/HEAD/README.zh-CN.md#目录与存储)。

## 日志与指标

```sh
RUST_LOG=olaya=debug,olaya_candle=debug olaya serve
```

`GET /metrics` 暴露 Prometheus 计数器和延迟直方图。响应包含 `x-typesafe-request-id`，便于日志关联。

## 文档

- [配置参考](https://github.com/hushink/olaya/blob/HEAD/docs/configuration.md)
- [库 API 指南](https://github.com/hushink/olaya/blob/HEAD/docs/library.md)
- [工作区 README](https://github.com/hushink/olaya/blob/HEAD/README.zh-CN.md)
- [模型目录](https://github.com/hushink/olaya/blob/HEAD/registry/README.zh-CN.md)
- [客户端 SDK](https://github.com/hushink/olaya/blob/HEAD/crates/olaya-sdk/README.zh-CN.md)
