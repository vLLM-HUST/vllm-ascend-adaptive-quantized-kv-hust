# MMLU 已存完整 token IDs 对照与评分门禁

日期：2026-10-11。范围：来源绑定的离线准备工具；不是实际服务验证或执行许可。

## 1. 解决的缺口

此前 stored transport 比较 `usage.prompt_tokens` 与本地提示长度，不能证明完整
token IDs 或模板相同。本轮新增
[`pujiang_mmlu_token_parity.py`](../scripts/pujiang_mmlu_token_parity.py)，将每个来源绑定
请求的已存 `/tokenize` 完整响应逐 token 比较。任一题不匹配，阻断两臂质量评分。
检查通过只说明文件字节一致；不认证服务来源、权重、签名 release 或启动参数。

## 2. 固定接口来源

只读取得 vLLM 精确提交 `ced6857afa0ea7b2e3f0846a62e1394e90f15607` 的
[protocol](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/entrypoints/serve/tokenize/protocol.py)、
[serving](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/entrypoints/serve/tokenize/serving.py) 和
[router](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/entrypoints/serve/tokenize/api_router.py)。
API Git blob、长度、SHA 和 AST 字段已核验；未导入或启动该服务，也未探测正式
endpoint。该 pin 只是候选接口依据，不是已接受 Host runtime 身份或兼容性证明。

该版本为 `POST /tokenize`，不是 `/v1/tokenize`；响应包含 `count`、`max_model_len`、
完整 `tokens` 和可空 `token_strs`。对应源码 SHA：

| 文件 | SHA-256 |
| --- | --- |
| protocol.py | `0bb4ddf2c0b1412e88c9d853e2ef7163f62665df0b8bbef86df5bdbdd9a76e44` |
| serving.py | `fb9fd00cc71df5c052f420a959e39434ce5fd3c84556275eb9dc25d43ff59750` |
| api_router.py | `dac2a4d0cd6076d50b568f3bae65dea0c6a65547fda073217a7dab9a1e2d8ccc` |

text-only chat 投影保留原 `model`、完整 `messages` 与
`chat_template_kwargs={"enable_thinking": false}`。显式设置
`add_generation_prompt=true`、`continue_final_message=false`、
`add_special_tokens=false`、`return_token_strs=false`；`chat_template`、`tools`、
`media_io_kwargs`、`mm_processor_kwargs` 为 null。原生成请求摘要仍绑定完整生成
参数，temperature/max_tokens 等不伪装成 tokenize 参数；不支持的 token-affecting
字段拒绝，不静默删除。正式模板覆盖规则仍待 owner 冻结。

## 3. 回执接线

只允许从固定来源重建的 v2 scorer bundle。在
[来源绑定 scorer 命令](PUJIANG_MMLU_SOURCE_PLAN_20261010.md) 追加：

```bash
  --stored-token-parity-root /absolute/path/to/private-tokenize-records
```

可与 `--stored-transport-root` 同时使用。B0/B1 分别位于该目录的
`B0/receipt.json`、`B1/receipt.json`。envelope 新增 `stored_token_parity_sha256`，
必须且只包含两臂真实 receipt SHA；已绑定摘要却省略 CLI 参数时拒绝，不能绕过评分。

receipt 采用 `aqkv-mmlu-stored-token-parity/v1`，字段为 schema_version、arm、
evidence_kind、scorer_bundle_sha256、tokenize_reference_revision、request_method、
request_path、tokenizer_declaration、requests。method/path 固定为 POST、/tokenize。
每题需 case_id、原生成请求的 completion_request_sha256、request artifact、
attempts、result；artifact 使用相对 path/sha256/bytes，result 与 stored transport
共用 HTTP/body 或四类失败格式。每题完整投影与来源绑定原请求重新比较。

declaration 采用 `aqkv-mmlu-tokenizer-declaration/v1`：绑定 repository、revision、
tokenizer/配置/template 文件摘要、class、库版本、served_model_name、context_tokens、
server_configuration_sha256 与 release_manifest_sha256。**后两项目前只检查摘要
格式和记录绑定，没有读取原始配置/manifest 或验证签名；声明也不认证服务端来源。**
本地库版本匹配是保守候选要求，不代表 owner 已选择该依赖组合。

所有 artifact 复用相对路径、symlink、1 MiB、长度与 SHA 检查；拒绝漏题、重复题、
错臂、伪 measured 证据、变更请求、错误摘要和重试元数据。attempts 只约束存储声明，
不证明现实无重试。固定 receipt 文件及输出排他处理，旧报告不覆盖。

## 4. 失败与主张边界

- 比较全部有序 IDs；拒绝数量-only、布尔/浮点/负 token、错误 count、非空 token_strs
  和 context mismatch。timeout/connection-error/not-attempted/response-too-large/
  HTTP 错误/坏 JSON 全部保留，每题都进入诊断，不能删除失败题恢复通过。
- 有效输入中任一题失败，报告 `stored_token_parity_mismatch_scoring_blocked`，
  `offline_quality=null`，不调用共享 scorer；CLI 写权限 0600 的新诊断报告并退出 1。
- 来源/文件损坏、缺失记录或不合法声明属输入完整性错误，直接非零退出，不能补成匹配。
- 完整匹配仅为 `stored_token_ids_match_origin_unverified`。serving_tokenizer_verified、
  transport_verified、release_signature_verified、tokenizer_declaration_origin_verified、
  execution_authorized 始终为 false，不生成正式质量或性能主张。

## 5. 本轮验证

从固定官方资产/tokenizer/selection/scorer 重建 70 题，执行两组各 70 题/臂的
离线接线。第一组响应复制本地 IDs；第二组只改 B1 第一题索引 7 的 ID，数量不变。
第二组准确识别一条 token-ids-mismatch，完整保留两臂各 70 题并阻断评分。
**两组都是合成 fixture，不是服务端响应。** 模型和 tokenize 网络请求均为 0。
精简摘要见 [公开开发回执](evidence/PUJIANG_MMLU_TOKEN_PARITY_20261011.json)。

新增 parity 测试 64 项；loopback 另补一项拒绝数字 `0` 冒充布尔 false 的握手
回归。完整插件本地 **780 passed**，零失败/错误/跳过；真实来源/tokenizer opt-in
与固定源码 AST 检查实际执行。旧 715 项和十个 HTTP 场景原回执保留，新检查独立
记账。没有 Host CI、NPU、真实模型成绩；验证时点尚未提交公共 PR。
后续获用户授权的 Draft 送审范围见 [五项控制第 10 节](PUJIANG_FIVE_DATASET_CONTROL_20261009.md)。

## 6. 后续输入与顺序

1. 明确 V5.4 INT8 KV/B1 准入并取得签名 release/OCI/core/plugin/model/tokenizer/
   server/client 身份，不能以源代码中有 `/tokenize` 替代这些输入。
2. 确认 Host 最小测试补丁的公共提交/正常 CI 路线；已发确认请求，不绕过审批。
3. 冻结正式选题/scorer/许可/tokenization 规则，确认实际服务接口后，补签名身份
   校验与受限真实 transport/parity 采集，不把合作式合成服务作为评测端。
4. 量化臂另需 Host/provider/profile/Manager/设备正确性及资源/清理回执；闭合后
   才正式执行 B0/B1，并提交 benchmark、回链 Issue/网站。Slack 只拟稿。

本轮没有 SSH、模型/设备/沙箱调用或新自动任务。
