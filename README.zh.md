# vLLM Ascend 自适应量化 KV 插件

本仓库是 Adaptive Quantized KV 课题面向 vLLM-HUST 和 Ascend 的独立扩展
Bundle。

当前处于 P1 打包阶段，只提供 Bundle v1 静态清单、宿主合同兼容性回执、
ACL Graph reset/recapture 回执校验，以及下一阶段所需的通用宿主接口提案。

当前版本不会自动激活运行时逻辑，不会 patch vLLM 或 vLLM Ascend，不会启动
模型服务，也不声明性能收益。现有宿主基线尚未提供本插件需要的已评审接口，
所以安装后保持惰性是有意设计的结果。

研究合同、实验脚本、原始证据和结论继续保存在
[`intellistream/ascend-adaptive-quantized-kv`](https://github.com/intellistream/ascend-adaptive-quantized-kv)，
不会混入 PyPI 运行时包。

后续发布使用 PyPI Trusted Publishing。仓库和 CI 中不得保存密码、2FA 验证
码、恢复码或长期 API token。

