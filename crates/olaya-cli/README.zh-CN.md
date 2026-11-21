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
| `olaya models add/remove` | 编辑运行时目录 |
| `olaya models stored/manifest/verify` | 查看 OCI 内容存储 |
| `olaya doctor` | 输出路径、目录来源和诊断信息 |

## 配置文件

`serve`、`ask` 和模型命令共享一份 TOML 配置。CLI 依次查找 `./olaya.toml`、`./sys1.toml`、
`~/.olaya/config.toml`；可用 `--config` 或 `OLAYA_CONFIG` 指定其他文件。相对路径相对于配置文件
所在目录解析。

```toml
[engine]
device = "cuda"
dtype = "f16"
default_model = "laya:latest"

[serve]
host = "127.0.0.1"
port = 11435
max_batch = 32
batch_wait_ms = 2

[hub]
hf_endpoint = "https://hf-mirror.com"

[registry]
# dir = "./registry"
# store = false
```

CLI 参数覆盖环境变量，环境变量覆盖配置文件。`olaya doctor` 会输出当前加载的配置路径和生效的
存储设置。

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
| `--device` | `cpu` | `cpu`、`cuda` 或 `metal` |
| `--dtype` | `f32` | CUDA/Metal 上的 `f32`、`f16` 或 `bf16` |
| `--tf32` | 关闭 | 允许 CUDA 使用 TF32 GEMM |
| `--max-batch` | `32` | 单次批处理前向的请求数 |
| `--batch-wait-ms` | `2` | 批处理等待窗口 |
| `--api-key` | `OLAYA_API_KEY` | 除 `/health` 外所有路由的 Bearer token |

> [!WARNING]
> 绑定非 loopback 地址时必须提供 `--api-key`，除非显式使用 `--allow-unauthenticated`。在可信网络之外
> 部署时，请使用终止 TLS 的反向代理。

## 模型目录

全新安装后，CLI 会从最新 GitHub Release 拉取 catalog bundle，并缓存到
`~/.olaya/cache/registry-catalog.json`；后续命令复用缓存。

```sh
olaya models list
olaya models show laya
olaya models pull laya
olaya models export --dir ./registry
```

- `--registry-dir` / `OLAYA_REGISTRY_DIR` 选择文件系统目录。
- `OLAYA_REGISTRY_URL` 选择托管的 catalog bundle。
- `--registry-store` 将内容寻址 store 当作目录。
- 拉取 `laya:latest` 这类 router 时会同时拉取所有路由目标（`laya` 和 `laya-multilingual`）；
  在终端中会用单行进度条/百分比刷新当前文件进度，重定向输出时改为周期性日志。
- Hugging Face 下载默认使用 `https://hf-mirror.com`；设置
  `HF_ENDPOINT=https://huggingface.co` 可使用官方端点。
- `models list` 默认把 alias 归到规范模型下；`--all` 输出与 API 一致的扁平列表。
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

- [工作区 README](https://github.com/hushink/olaya/blob/HEAD/README.zh-CN.md)
- [模型目录](https://github.com/hushink/olaya/blob/HEAD/registry/README.zh-CN.md)
- [客户端 SDK](https://github.com/hushink/olaya/blob/HEAD/crates/olaya-sdk/README.zh-CN.md)
