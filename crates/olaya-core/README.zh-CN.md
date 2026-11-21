# olaya-core

[English](README.md) | 简体中文

[olaya](https://github.com/hushink/olaya) 的协议与原语层。这个 crate 不包含模型运行时，不依赖
candle，也不提供 HTTP 服务。

## 包含内容

- System One 的 `Question`、`Answer`、`State`、`Content`、`Questions` 和 `Answers` 类型。
- 请求校验和 TypeSafe 兼容的请求解析器。
- 校准工具和与 Python 兼容的规范 JSON 渲染。
- 模型 manifest 数据结构，包括 router manifest。
- `ManifestRegistry` 和 `ManifestStore` 扩展 trait。
- `RuntimeMetrics`：原子计数、固定桶延迟直方图、可序列化快照和 Prometheus 文本渲染。

candle family SDK 见 [`olaya-candle`](https://github.com/hushink/olaya/blob/HEAD/crates/olaya-candle/README.zh-CN.md)；
engine 与服务见 [`olaya`](https://github.com/hushink/olaya/blob/HEAD/crates/olaya/README.zh-CN.md)；
命令行工具见 [`olaya-cli`](https://github.com/hushink/olaya/blob/HEAD/crates/olaya-cli/README.zh-CN.md)。

## 文档

- [工作区 README](https://github.com/hushink/olaya/blob/HEAD/README.zh-CN.md)
- [设计说明](https://github.com/hushink/olaya/blob/HEAD/docs/design.md)
