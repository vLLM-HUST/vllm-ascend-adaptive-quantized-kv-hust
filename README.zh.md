# vLLM Ascend 自适应量化 KV 插件

本仓库是 Adaptive Quantized KV 课题面向 vLLM-HUST 和 Ascend 的独立扩展。

当前处于 P1 打包阶段，只提供 Extension Manager 0.2 实验性静态清单、宿主合同兼容性回执、
ACL Graph reset/recapture 回执校验，以及下一阶段所需的通用宿主接口提案。

另提供需要显式导入的 CPU 分页 INT8 attention 参考实现，承接研究仓已关闭 PR #19
中的分块反量化与在线 softmax 思路。它支持 GQA、变长序列、per-channel scale
和因果后缀查询，并以独立 dense 计算做数值对照；不是 Ascend 融合算子，也未接入服务。
使用 Python 3.12 安装 `.[test,component]` 后运行 `pytest -q tests/test_reference.py`。
默认安装和发现不会加载 Torch。接管范围、来源和阶段门禁见
[`PR19_TAKEOVER_CONTROL_20260904.md`](docs/PR19_TAKEOVER_CONTROL_20260904.md)。

CPU 参考实现已通过插件 PR #3 合入，提交为
`8587845f29d70256103990ed6492837ddaf5a800`。当前宿主依赖、PR #35 状态和
责任边界以
[`ISSUE1_PR35_CONTROL_20260926.md`](docs/ISSUE1_PR35_CONTROL_20260926.md)
为准。

自 2026-09-27 起，本扩展已不再展示于公开 MOD Workshop。恢复公开展示前，
必须闭合与精确版本绑定的校准 profile、宿主已接纳且可达的 C8 路径、真实
Extension Manager 与 NPU serving 回执、回滚和退出清理证据，并为质量和性能
结论分别保留可追溯原始产物。仅能被发现或仅修改为 active manifest 不满足
恢复条件。逐项证据清单见
[`docs/PUBLIC_SURFACE_RESTORATION_MATRIX_20260927.md`](docs/PUBLIC_SURFACE_RESTORATION_MATRIX_20260927.md)。

当前版本不会自动激活运行时逻辑，不会 patch vLLM 或 vLLM Ascend，不会启动
模型服务，也不声明性能收益。现有宿主基线尚未提供本插件需要的已评审接口，
所以安装后保持惰性是有意设计的结果。
只要 manifest 仍标记为 `import_only`，Extension Manager 就会拒绝启用；必须先由
宿主接受接口合同，并在后续 manifest 中明确声明 active implementation。

本组织插件是课题实现和可见进展的主要载体。负责人后续允许提交最小、通用、
默认关闭的宿主合同候选，交由 CODEOWNERS/maintainer 审查；宿主 Draft PR #35
属于这一候选路线，但不代表宿主接口的交付、采纳和最终决策责任转移给本课题。
只有宿主负责人接受精确接口版本且真机正确性门禁通过后，才允许讨论运行时激活。
可复现实验 workload 继续由研究仓中的固定
`intellistream/llm-serving-workloads` gitlink 提供。

## 维护状态与当前交付目标

- 插件维护与证据负责人：[`@XilingGao`](https://github.com/XilingGao)。
- 宿主架构、接口交付与最终 go/no-go 负责人：
  [`@ShuhaoZhangTony`](https://github.com/ShuhaoZhangTony)，责任边界记录在
  [Issue #1](https://github.com/vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust/issues/1)。
- 当前集成工作由 Draft PR #4 承载；恢复矩阵闭合前，默认发行包继续保持
  `import_only`。
- 第一组预注册交付测试固定为 `Qwen3.5-35B-A3B`、TP2、APC、MTP2、async、
  `FULL_AND_PIECEWISE`，使用官方 benchmark 的
  `prefix-repetition-online-2chip` 场景；合同见
  [`docs/QWEN35_PREFIX_REPETITION_PILOT_20260928.md`](docs/QWEN35_PREFIX_REPETITION_PILOT_20260928.md)。

“持续维护”不等于“运行时已可用”。校准 profile、宿主已接纳且可达的
Host/provider 版本、真实 Manager 激活、NPU 正确性、图回放、回滚与清理证据
仍是性能测试和恢复公开展示前的强制门禁。

10 月前的分级施工顺序见
[`docs/PRE_OCTOBER_DELIVERY_EXECUTION_GUIDE_20260928.md`](docs/PRE_OCTOBER_DELIVERY_EXECUTION_GUIDE_20260928.md)。

研究合同、实验脚本、原始证据和结论继续保存在
[`intellistream/ascend-adaptive-quantized-kv`](https://github.com/intellistream/ascend-adaptive-quantized-kv)，
不会混入 PyPI 运行时包。

张老师指定保留的历史 Ascend PR #271/#279 及其合同证据边界记录在
[`docs/LEGACY_CONTRACT_EVIDENCE.md`](docs/LEGACY_CONTRACT_EVIDENCE.md)。这些
引用只用于接口设计，不能视为宿主 API、运行依赖或性能结果。

后续发布使用 PyPI Trusted Publishing。仓库和 CI 中不得保存密码、2FA 验证
码、恢复码或长期 API token。

CI 在 Python 3.10、3.12 和 3.14 上运行测试，并在 Python 3.12 上构建和校验
惰性扩展发行包。当前流水线只保留构建产物，不会上传到 PyPI。
