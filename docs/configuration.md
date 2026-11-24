# 配置参考

`olaya` 的 `serve`、`ask`、`models` 和 `doctor` 共用一套配置。所有配置项都遵循同一优先级：

```text
CLI 参数 > 环境变量 > 配置文件 > 内置默认值
```

只有内置默认值写在二进制里；其余值都从这三层配置解析，不在命令实现里重复硬编码。

## 配置文件

查找顺序：

1. `--config PATH`
2. `OLAYA_CONFIG`
3. `./olaya.toml`
4. `./sys1.toml`
5. `$OLAYA_HOME/config.toml`（默认 `~/.olaya/config.toml`）

如果都没有找到，第一次运行会写一份带当前编译 feature 信息的模板到
`$OLAYA_HOME/config.toml`。模板里的 `device = "auto"` 不写死机器相关值：CUDA/Metal 编译进二进制且
运行时可用的 accelerator 优先，否则回退 CPU。

配置文件中的相对路径按**配置文件所在目录**解析；CLI/环境变量给出的相对路径按**当前工作目录**解析。

## 完整配置

```toml
[engine]
# auto | cpu | cuda | cuda:<index> | metal
device = "auto"
# CUDA/Metal 权重精度：f32 | f16 | bf16；CPU 始终按 f32 计算
dtype = "f32"
# 仅 CUDA：允许 TF32 tensor-core GEMM
tf32 = false
# 请求未指定模型时使用；空字符串回退到 "laya:latest"
default_model = "laya:latest"
# 同时保留在内存中的模型数量，最小 1
max_loaded = 3

[serve]
host = "127.0.0.1"
port = 11435
# 设置后除 /health 外所有路由要求 Authorization: Bearer <token>
# api_key = "..."
# 非 loopback 地址默认要求 api_key；明确允许无鉴权时才设为 true
allow_unauthenticated = false
# 1 表示关闭跨请求合批
max_batch = 32
# 批次第一个请求等待同伴的时间
batch_wait_ms = 2
# 单个 HTTP 请求体的最大字节数
max_body_bytes = 8388608
# 等待进入批处理的请求上限；超过后调用方等待背压
max_queue = 64
# 单个 HTTP 请求（含慢速 body 读取）的总超时；0 关闭
request_timeout_secs = 600
# /api/pull 与 /api/load 的并发上限
max_operations = 4
# 允许调用 API 的浏览器 Origin；留空关闭 CORS（推荐，尤其在本机无鉴权时）
# cors_origins = ["https://app.example.com"]
# 额外接受的 Host 头（loopback 绑定默认只接受 loopback 主机名）
# allowed_hosts = ["olaya.internal"]

[hub]
# 以下路径默认都在 $OLAYA_HOME 下
# models_dir = "models"
# cache_dir = "cache"
# tmp_dir = "tmp"
# store_dir = "store"
# true 时禁止所有网络访问；远端 catalog 不可用
# offline = false
# Hugging Face 镜像；设为 https://huggingface.co 使用官方端点
# hf_endpoint = "https://hf-mirror.com"
# modelscope_endpoint = "https://www.modelscope.cn"
# HTTP/SOCKS 代理，以及不经过代理的主机列表。
# proxy = "http://127.0.0.1:7890"
# proxy = ""                     # 显式关闭 HTTP(S)_PROXY 环境变量
# no_proxy = "localhost,127.0.0.1"
# 允许 manifest 指向 loopback/内网地址（默认拒绝，降低 SSRF 风险）
# allow_private_network = false
# 下载空闲/读取超时（秒）、连接超时（秒）与 HTTP 重试次数
download_timeout_secs = 60
connect_timeout_secs = 15
download_retries = 5

[registry]
# 显式目录会关闭远端 catalog，只使用该目录
# dir = "./registry"
# 把 OCI content store 当作 catalog 来源，忽略 dir/url
store = false
# 隐式 catalog 的远端 bundle 地址
# url = "https://github.com/hushink/olaya/releases/latest/download/registry-catalog.json"
# 拉取 catalog bundle 的 HTTP 超时
timeout_secs = 30
```

### engine

| 键 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `device` | string | `auto` | `auto`、`cpu`、`cuda`、`cuda:<index>`、`metal` |
| `dtype` | string | `f32` | `f32`、`f16`、`bf16`；CPU 始终 f32 |
| `tf32` | bool | `false` | 仅 CUDA 生效 |
| `default_model` | string | `laya:latest` | 空模型名时的回退 |
| `max_loaded` | integer | `3` | 至少 1；超过后按 LRU 卸载 |

### serve

| 键 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `host` | string | `127.0.0.1` | 绑定地址 |
| `port` | integer | `11435` | 1–65535 |
| `api_key` | string | 未设置 | Bearer token |
| `allow_unauthenticated` | bool | `false` | 非 loopback 且无 key 时是否放行 |
| `max_batch` | integer | `32` | `1` 关闭合批 |
| `batch_wait_ms` | integer | `2` | 合批等待窗口 |
| `max_body_bytes` | integer | `8388608` | HTTP body 上限 |
| `max_queue` | integer | `64` | 批处理队列上限 |
| `request_timeout_secs` | integer | `600` | 单请求总超时，0 关闭，最大 86400 |
| `max_operations` | integer | `4` | pull/load 并发上限，1–1024 |
| `cors_origins` | string[] | `[]` | 允许的浏览器 Origin；空数组关闭 CORS |
| `allowed_hosts` | string[] | `[]` | 额外接受的 Host 名；空时 loopback 只接受 loopback 名 |

`max_batch`、`max_queue`、`max_body_bytes`、`max_operations` 必须至少为 1，`port` 不能为 0；
`api_key` 不能为空或含空白；`cors_origins` 中每项必须是 `*` 或 `http(s)://` Origin；否则启动时
直接报错。上限分别为 `max_batch` 65536、`max_queue` 1000000、`max_body_bytes` 1 GiB、
`batch_wait_ms` 3600000、`max_operations` 1024、`request_timeout_secs` 86400。

默认不发送任何 CORS 响应头。只有显式配置 `cors_origins` 后，浏览器跨域请求（含预检）才会被放行；
这一默认值避免本机无鉴权的 `olaya serve` 被任意网页调用。`cors_origins = ["*"]` 会放行所有 Origin。

服务还会校验 `Host` 头以阻止 DNS rebinding：绑定 loopback 时，未配置 `allowed_hosts` 只接受
`localhost`、`127.0.0.1` 或 `::1`；配置后只接受列出的主机名。非 loopback 绑定默认接受任意 Host，
建议在反向代理后显式设置 `allowed_hosts`。

### hub

| 键 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `models_dir` | path | `$OLAYA_HOME/models` | 物化后的模型目录 |
| `cache_dir` | path | `$OLAYA_HOME/cache` | 下载状态、catalog 缓存 |
| `tmp_dir` | path | `$OLAYA_HOME/tmp` | 下载 staging |
| `store_dir` | path | `$OLAYA_HOME/store` | OCI blob/manifest store |
| `offline` | bool | `false` | 禁止网络 |
| `hf_endpoint` | string | `https://hf-mirror.com` | Hugging Face 端点 |
| `modelscope_endpoint` | string | ModelScope 默认 | ModelScope 端点 |
| `proxy` | string | 未设置 | HTTP/SOCKS 代理；显式空字符串关闭环境变量代理 |
| `no_proxy` | string | 未设置 | 逗号分隔的免代理主机 |
| `allow_private_network` | bool | `false` | 允许 manifest 引用 loopback/内网地址 |
| `download_timeout_secs` | integer | `60` | 下载空闲/读取超时，1–86400 |
| `connect_timeout_secs` | integer | `15` | 下载连接超时，1–3600 |
| `download_retries` | integer | `5` | 失败 HTTP 传输的重试次数 |

`cache_dir` 和 `tmp_dir` 是可丢弃目录：删除它们不会影响已经物化的模型、`store_dir` 中的 blob 或
`config.toml`；下一次操作会按需重建。为了防止删除缓存误删权威数据，CLI 会拒绝以下布局：

- `cache_dir` / `tmp_dir` 与 `models_dir` / `store_dir` 相同或互相包含
- `cache_dir` / `tmp_dir` 与 `[registry].dir` 相同或互相包含
- 配置文件位于 `cache_dir`、`tmp_dir`、`models_dir`、`store_dir` 或 `registry.dir` 之下

`models_dir` 本身也是可重建的投影；权威内容在 `store_dir`。如果在一个正在进行的下载过程中删除
`cache_dir`/`tmp_dir`，该次下载可能失败或失去 resume 信息，重新执行即可；已经完成的模型文件和
store blob 不会被删除。

> 安全提示：模型 manifest 的 `source`、`revision` 和每个文件的 `url` 都会触发下载请求。`url`
> 只接受 `http://`/`https://`。默认会拒绝 loopback、私网、link-local、CGNAT 和文档地址，并对
> 主机名做解析检查；本地镜像或内网源需要显式设置 `[hub].allow_private_network = true`，或通过
> `--allow-private-network` / `OLAYA_ALLOW_PRIVATE_NETWORK=true` 开启。模型文件（safetensors、
> tokenizer、JSON 配置、PyTorch pickle）会在进程内解析。只应从可信 catalog/仓库拉取模型；不要对
> 不可信来源运行 `models pull` 或暴露无鉴权的 `/api/pull`。

Hugging Face 鉴权沿用 `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`，ModelScope 沿用其官方凭证文件；
这些凭据由 shaul 直接读取，不需要（也不会）写入 `olaya.toml`。

### registry

| 键 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `dir` | path | `./registry`（存在时）否则 `$OLAYA_HOME/registry` | 本地 manifest 目录 |
| `store` | bool | `false` | 以 content store 作为 catalog |
| `url` | string | latest release asset | 远端 catalog bundle |
| `timeout_secs` | integer | `30` | bundle HTTP 超时，1–86400 |

隐式 `$OLAYA_HOME/registry` 在远端/缓存可用时以远端 catalog 为准，但会把本地新增的、不与远端
名称/别名冲突的 manifest 合并进来（来源显示为 `network+dir`）。远端/缓存都不可用时使用内置 seed
和该目录。显式 `--registry-dir` / `OLAYA_REGISTRY_DIR` / `[registry].dir` 会关闭远端拉取，目录
不可用时直接报错，不会静默换源。

## 环境变量

| 变量 | 覆盖的配置 | 说明 |
| --- | --- | --- |
| `OLAYA_HOME` | 所有默认路径的根 | 默认 `~/.olaya` |
| `OLAYA_CONFIG` | 配置文件位置 | 等价 `--config` |
| `OLAYA_DEVICE` | `[engine].device` | |
| `OLAYA_DTYPE` | `[engine].dtype` | |
| `OLAYA_TF32` | `[engine].tf32` | `true/false/1/0/yes/no/on/off` |
| `OLAYA_DEFAULT_MODEL` | `[engine].default_model` | |
| `OLAYA_MAX_LOADED` | `[engine].max_loaded` | |
| `OLAYA_HOST` | `[serve].host` | 仅 `serve` |
| `OLAYA_PORT` | `[serve].port` | 仅 `serve` |
| `OLAYA_API_KEY` | `[serve].api_key` | 仅 `serve` |
| `OLAYA_ALLOW_UNAUTHENTICATED` | `[serve].allow_unauthenticated` | 仅 `serve` |
| `OLAYA_MAX_BATCH` | `[serve].max_batch` | 仅 `serve` |
| `OLAYA_BATCH_WAIT_MS` | `[serve].batch_wait_ms` | 仅 `serve` |
| `OLAYA_MAX_BODY_BYTES` | `[serve].max_body_bytes` | 仅 `serve` |
| `OLAYA_MAX_QUEUE` | `[serve].max_queue` | 仅 `serve` |
| `OLAYA_REQUEST_TIMEOUT_SECS` | `[serve].request_timeout_secs` | 仅 `serve` |
| `OLAYA_MAX_OPERATIONS` | `[serve].max_operations` | 仅 `serve` |
| `OLAYA_CORS_ORIGINS` | `[serve].cors_origins` | 仅 `serve`；逗号分隔 |
| `OLAYA_ALLOWED_HOSTS` | `[serve].allowed_hosts` | 仅 `serve`；逗号分隔 |
| `OLAYA_MODELS_DIR` | `[hub].models_dir` | |
| `OLAYA_CACHE_DIR` | `[hub].cache_dir` | |
| `OLAYA_TMP_DIR` | `[hub].tmp_dir` | |
| `OLAYA_STORE_DIR` | `[hub].store_dir` | |
| `OLAYA_OFFLINE` | `[hub].offline` | 布尔值 |
| `HF_ENDPOINT` | `[hub].hf_endpoint` | |
| `MODELSCOPE_ENDPOINT` | `[hub].modelscope_endpoint` | |
| `OLAYA_PROXY` | `[hub].proxy` | |
| `OLAYA_NO_PROXY` | `[hub].no_proxy` | 逗号分隔 |
| `OLAYA_ALLOW_PRIVATE_NETWORK` | `[hub].allow_private_network` | 布尔值 |
| `OLAYA_DOWNLOAD_TIMEOUT_SECS` | `[hub].download_timeout_secs` | 秒 |
| `OLAYA_CONNECT_TIMEOUT_SECS` | `[hub].connect_timeout_secs` | 秒 |
| `OLAYA_DOWNLOAD_RETRIES` | `[hub].download_retries` | |
| `OLAYA_REGISTRY_DIR` | `[registry].dir` | 显式目录 |
| `OLAYA_REGISTRY_STORE` | `[registry].store` | 布尔值 |
| `OLAYA_REGISTRY_URL` | `[registry].url` | |
| `OLAYA_REGISTRY_TIMEOUT_SECS` | `[registry].timeout_secs` | |
| `OLAYA_MODEL` | `ask --model` | 单次请求模型 |

布尔 CLI/env 值使用 `--flag`、`--flag=true`、`--flag=false` 三种形式。需要显式关闭文件中的
`true` 时使用后者，例如：

```sh
olaya --tf32=false ask --model laya --state "..." --questions questions.json
olaya --offline=false models list
olaya --registry-store=false models list
olaya serve --allow-unauthenticated=false
```

## 各命令的生效范围

| 配置 | `serve` | `ask` | `models` | `doctor` |
| --- | --- | --- | --- | --- |
| `[engine]` | ✅ | ✅ | —（catalog 命令不加载模型） | ✅ 诊断 |
| `[serve]` | ✅ | — | — | ✅ 诊断 |
| `[hub]` | ✅ | ✅ | ✅ | ✅ |
| `[registry]` | ✅ | ✅ | ✅ | ✅ |

`serve` 和 `ask` 会初始化 CUDA/Metal 设备；`models list/show/export` 只读取 catalog，不会加载模型或
选择 accelerator。

## 示例

### 本地 CPU 开发

```toml
[engine]
device = "cpu"
dtype = "f32"
default_model = "laya"

[serve]
host = "127.0.0.1"
port = 11436
max_batch = 8
```

```sh
olaya serve
olaya ask --model laya --state "This is great." --questions questions.json
```

### CUDA 服务器

```toml
[engine]
device = "cuda"
dtype = "f16"
tf32 = true

[serve]
host = "0.0.0.0"
port = 11435
api_key = "replace-me"
```

```sh
olaya doctor
olaya serve
```

`doctor` 会输出 `device_preference`、实际 `device`、`compiled_accelerators`、`dtype`、`tf32`、
`default_model` 和 `max_loaded`。

### 离线或内网 catalog

```toml
[hub]
offline = true

[registry]
dir = "D:/models/registry"
```

```sh
olaya --registry-dir ./registry models list
olaya --registry-dir ./registry models pull laya:latest
```

首次使用隐式目录时，CLI 会先把内置 manifest seed 写进去；没有网络也能 `models list`。

### 只使用 content store

```toml
[registry]
store = true
```

```sh
olaya models pull laya:latest
olaya --registry-store models list
```

### 自定义存储目录

```toml
[hub]
models_dir = "D:/olaya/models"
cache_dir = "D:/olaya/cache"
tmp_dir = "D:/olaya/tmp"
store_dir = "D:/olaya/store"
```

相对路径会按配置文件目录解析，适合把整个 `olaya` 数据目录随项目移动。

### 远端 catalog 与代理

```toml
[registry]
url = "https://internal.example/olaya/registry-catalog.json"
timeout_secs = 60

[hub]
proxy = "http://127.0.0.1:7890"
```

```sh
OLAYA_REGISTRY_URL=https://mirror.example/catalog.json \
HF_ENDPOINT=https://huggingface.co \
olaya models list
```

## 诊断

```sh
olaya doctor
```

输出包括：

- 实际加载的配置文件与基准目录
- 解析后的 device/dtype/tf32/default_model/max_loaded
- registry 目录、来源（`network`、`network+dir`、`cache`、`dir`、`dir-fallback`、`dir-offline`、`store`）、URL、超时
- `[hub]` 的目录存在性、offline、HF/ModelScope 端点、proxy（已脱敏）、`no_proxy`、超时与重试
- `[serve]` 的有效 host/port/鉴权/批量/请求体限制/CORS Origin/Host 允许列表
- content store 中已有的模型

`doctor` 不会连接远端来验证模型文件；`models verify <model>` 会重新计算 store 中每个 blob 的 sha256。

## 排错顺序

1. 端口被占用：检查 `[serve].port` 或 `--port`。
2. 非 loopback 启动被拒绝：设置 `--api-key`，或明确 `--allow-unauthenticated=true`。
3. `unknown field`：配置启用 `deny_unknown_fields`，拼错的键会直接报错。
4. 下载失败：先确认 `HF_ENDPOINT`、`OLAYA_PROXY` 和 `[hub].offline`，再运行 `olaya doctor`。
5. catalog 旧了：临时设置 `OLAYA_REGISTRY_URL` 或传入显式 `--registry-dir`，避免命中缓存。
