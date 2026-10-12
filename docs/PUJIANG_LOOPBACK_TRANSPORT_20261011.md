# 有界 HTTP 开发验证：仅本地 fixture

更新：2026-10-11。实现已在本地验证；当时未提交公共 PR，不接受远端服务地址。
后续授权 Draft 送审范围见 [五项控制第 10 节](PUJIANG_FIVE_DATASET_CONTROL_20261009.md)。

## 1. 本轮实际补齐的内容

[`pujiang_loopback_transport.py`](../scripts/pujiang_loopback_transport.py) 将一个合成
请求发送到开发 fixture，保存实际收到的 HTTP 应用层正文和收集审计，再交给既有
stored transport 解析器。它没有 CLI、生产入口或远端模式，不加载模型，也不注册
插件运行实现；发行入口仍是 `import_only`。

本地插件全量回归 **715 passed**，零失败、错误、跳过；其中新增 loopback 测试
**44 项**。另外独立运行并保留十个实际本地 HTTP 场景的原始记录，不以 pytest
临时文件替代持久回执。测试中注入的进程替身只用于半写入元数据/启动失败等控制流，
不计为实际 HTTP 场景。

十个场景为正常响应、302、503、断连、完全停滞、持续慢速小块响应、超大正文、
正文截断、错误 nonce、截断握手。前八个各发起一次合成 POST；后两个没有 POST。
正常 fixture 输出可解析；其余已发起请求保留为失败，不从错误或部分正文提取答案。
这不是 MMLU 真实作答或十次模型调用，模型请求为 **0**，无质量/性能成绩。

## 2. 边界与失败策略

| 项目 | 实现与验证范围 |
| --- | --- |
| 地址 | 字面量 `127.0.0.1`，调用者只能给端口；没有 DNS、远端 URL 或代理环境调用 |
| 服务握手 | 先 GET `/_aqkv_fixture`，校验固定 schema、调用者提供的 64 位 nonce 和 fixture 声明；失败不发 POST。这是合作式开发服务识别，不是防恶意服务的认证，也不是正式模型身份 |
| 请求 | 固定合成模型名 `aqkv-loopback-fixture`、非 streaming；单请求最多 1 MiB，拒绝 NaN；实际请求字节与 server 独立观测摘要对照 |
| 时限 | 0.25–10 秒绝对截止时间，包含子进程启动、握手与正文读取；子进程 POSIX alarm 与父进程有界 join 双重约束，持续小块到达不能刷新时限。停止/回收有额外清理宽限，不把它算成模型延迟 |
| 次数 | 握手后最多一个 POST，不重试、不跟随重定向；client attempt 标记不单独证明 server 已收到，十个 fixture 另有服务端计数 |
| 大小/完整性 | 正文上限 1 MiB，可下调；至多读取上限加一字节。超大响应记 `response-too-large`；Content-Length 未满足记 `connection-error`，即使部分内容是合法 JSON 也不评分 |
| 原始数据 | 普通响应完整正文保存；超大/截断正文只保存诊断前缀并标注不评分。HTTP 解析后的应用正文不是 TCP 抓包，不声称保留全部网络字节 |
| 元数据 | 读取上限 8 KiB；半写入、非法结构、缺结果或启动失败留下审计并拒绝生成可评分记录，不猜测请求成功 |
| 退出 | 只回收本函数创建的子进程；不扫描、停止其他进程。本地回执与测试确认本轮自有子进程已退出 |
| 存储 | 新目录排他创建、0700；记录 0600、排他创建；旧输出拒绝覆盖，保留 SHA/字节数 |

握手之前失败不生成 `stored-request.json`，但保留 `capture-audit.json`，调用者须将
批次视为采集未完成。握手后已发起的超时/网络/HTTP/大小失败，接入旧解析器后仍留
在完整分母。`response-too-large` 是 stored transport v1 的新增失败类型，原格式和
已有回执不改写。

## 3. 尚未完成的正式部分

本轮没有实现远端模型 executor、正式服务身份握手或服务端 token IDs parity，亦
没有放开 B0/B1 执行。候选 70 题的选题/scorer 审阅、V5.4 INT8 KV/B1 准入、签名
release/OCI/core/plugin/model/tokenizer/server/client 身份、Host/provider/profile/
Manager/真实正确性与资源清理条件，仍按现有控制文档核验。

下一步先取得负责人输入，再设计有正式身份与任务清单绑定的执行器；本地 fixture
实现不能简单放开 URL 就冒充该执行器。未取得真实结果前，不生成 task quality、
AQKV eligibility、性能收益或网站恢复依据。

控制入口：[五数据集控制](PUJIANG_FIVE_DATASET_CONTROL_20261009.md)、
[V5.4 对齐](PUJIANG_V54_ALIGNMENT_20261010.md)、
[MMLU 来源与评分连接](PUJIANG_MMLU_SOURCE_PLAN_20261010.md)。
