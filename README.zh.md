# vLLM Ascend 自适应量化 KV 插件

本仓库是 Adaptive Quantized KV 课题面向 vLLM-HUST 和 Ascend 的独立扩展。

当前处于 P1 打包阶段，只提供 Extension Manager 0.2 实验性静态清单、宿主合同兼容性回执、
ACL Graph reset/recapture 回执校验，以及下一阶段所需的通用宿主接口提案。

当前版本不会自动激活运行时逻辑，不会 patch vLLM 或 vLLM Ascend，不会启动
模型服务，也不声明性能收益。现有宿主基线尚未提供本插件需要的已评审接口，
所以安装后保持惰性是有意设计的结果。
只要 manifest 仍标记为 `import_only`，Extension Manager 就会拒绝启用；必须先由
宿主接受接口合同，并在后续 manifest 中明确声明 active implementation。

本组织插件是课题实现和可见进展的唯一载体。课题不再直接向 `vllm-hust` 或
`vllm-ascend-hust` 提交优化 commit/PR；缺少的通用宿主接口在本仓库提出，
由宿主负责人实现，或由负责人明确指定其他交付方式。可复现实验 workload
继续由研究仓中的固定 `intellistream/llm-serving-workloads` gitlink 提供。

研究合同、实验脚本、原始证据和结论继续保存在
[`intellistream/ascend-adaptive-quantized-kv`](https://github.com/intellistream/ascend-adaptive-quantized-kv)，
不会混入 PyPI 运行时包。

后续发布使用 PyPI Trusted Publishing。仓库和 CI 中不得保存密码、2FA 验证
码、恢复码或长期 API token。

CI 在 Python 3.10、3.12 和 3.14 上运行测试，并在 Python 3.12 上构建和校验
惰性扩展发行包。当前流水线只保留构建产物，不会上传到 PyPI。
