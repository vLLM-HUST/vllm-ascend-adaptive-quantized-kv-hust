# 10 月前公开贡献与 Qwen3.5 交付施工指导

日期：2026-09-28

## 1. 目标与口径

本轮必须同时满足两类要求，不能相互替代：

1. **贡献评价合规**：最终成果位于 `vLLM-HUST` 公开仓库，通过 PR 提交；
2. **项目交付闭环**：至少完成一组 `Qwen3.5-35B-A3B` baseline/treatment
   正向测试，保留原始数据并向负责人回报。

截至本文建立时：

- 插件仓 `vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust` 已公开；
- 插件 PR #3 已合并，是已经闭环的公开 PR 贡献；
- 插件 PR #4 为 Ready、mergeable，`b6c1fe4` 的四项 CI 全绿，等待
  `ShuhaoZhangTony` review；
- Host PR #35 仍为 Draft，等待 Host/profile 路线确认；
- Host PR #37 为 Ready，CPU baseline 已通过，但必需的 source-change gate
  仍未完整结束；
- 插件仍为 `import_only`，没有真实 NPU/service 性能结论；
- Qwen3.5 第一组测试已预注册，但尚未执行。

私有 `intellistream/ascend-adaptive-quantized-kv` 只作为内部研究和原始证据仓，
不计入公开贡献。需要计入评价的最终代码、测试和交付文档必须进入公开插件仓
或公开 Host PR。

## 2. 优先级总表

| 优先级 | 工作 | 当前状态 | 完成定义 |
| --- | --- | --- | --- |
| P0 | PR #4 评审与合并 | Ready、CI 全绿、已请求 review | 负责人 review，PR 合并，记录 merge SHA |
| P0 | Host/profile 路线决定 | PR #35 Draft；profile 路线未定 | Host 负责人记录接受路线、交付负责人和精确 revision |
| P0 | Qwen3.5 执行输入 | 模型、校准 profile、两卡窗口未就绪 | 模型/配置/数据 digest、profile provenance、资源租约齐全 |
| P1 | Host baseline 与 provider 闭环 | PR #37 未合并；PR #35 未通过完整门禁 | PR #37 基线闭环；PR #35 重放测试并获 Host 接纳 |
| P1 | Manager/NPU 正确性 | 未执行 | discover/check/plan/run、数值、图回放、fallback、清理全部通过 |
| P1 | 一组正向 baseline/treatment | 只完成预注册 | 满足固定成功率、TTFT、throughput 和正确性阈值 |
| P1 | 原始数据与回报 | 无可回报真机数据 | 原始产物、哈希、配置和结论发布到公开 PR/Issue 并通知负责人 |
| P2 | MOD 恢复公开展示 | 当前已下架 | 恢复矩阵通过，负责人同意并完成网站 PR |
| P3 | 重装后恢复验证 | 迁移前备份已完成 | 新集群恢复、哈希验证、环境重建和最小测试通过 |
| P4 | AgentX 榜单 | 未运行、非强制 | 已有合格 256k 服务时再运行 smoke；不得替代 P1 |

## 3. P0：立即处理的外部决策和贡献合规

### 3.1 推进插件 PR #4

负责人：`XilingGao` 跟进，`ShuhaoZhangTony` 评审。

已完成：

- 仓库公开并通过匿名 API 验证；
- 完整历史未发现私钥、常见服务 token、口令、大文件或模型权重；
- Apache-2.0 许可证存在；
- PR #4 已转 Ready 并请求 review；
- `b6c1fe4` 对应 CI run `36430203806` 四项通过。

剩余步骤：

1. 只处理 reviewer 提出的具体修改，不趁机扩展运行时范围；
2. 每次更新后保留精确 head 和 CI run；
3. review 通过后由有权限的维护者合并；
4. 在 Issue #1 记录 merge SHA、CI 和公开 URL。

验收：PR #4 状态为 `MERGED`。如果截止前只有公开 Ready PR，则只能报告为
“已提交待接纳”，不能报告为“已合并贡献”。

### 3.2 请求 Host/profile 最小决策

Host 负责人必须在 Issue #1 或对应 Host Issue/PR 明确：

1. 采用 ModelSlim-compatible BF16+C8 sidecar/checkpoint，还是 Host-owned
   external profile source；
2. profile 的生成/校准负责人、存放位置、精确模型 revision 和 provenance；
3. PR #35 的接口是接受、修改还是替换；
4. 最终 Host 交付负责人和精确 accepted revision；
5. 两张 910B2 的使用窗口，以及重装后模型所在位置。

没有上述记录时，不得自行虚构 scale、默认采用 `1/0` 参数、绕过 Host 选择逻辑
或把 W8A8 fixture 当作指定 BF16 target。

### 3.3 冻结执行输入

在启动任何两卡任务前，必须形成一份 revision-bound 执行清单：

- model：`Qwen/Qwen3.5-35B-A3B@59d61f3ce65a6d9863b86d2e96597125219dc754`；
- BF16 weight 和 tokenizer 的实际位置与 digest；
- TP2、APC、MTP2、async、`FULL_AND_PIECEWISE`；
- calibrated zero-offset C8 profile 的逻辑 tensor digest 和 TP2 分片规则；
- accepted Host、plugin、Manager、benchmark SHAs；
- 两张独占设备的租约、超时和清理责任；
- benchmark `vLLM-HUST/vllm-hust-benchmark@47c12e5`；
- scenario `prefix-repetition-online-2chip`。

缺少任一项即停止，不下载 35B 权重、不启动服务、不占卡等待。

## 4. P1：Host、正确性和一组正向测试

### 4.1 关闭 Host PR #37 基线门禁

1. 等待或推动 self-hosted `select-tests`/source-change gate 完成；
2. 确认最终 CI 没有 queued、failure 或被错误解释的 skipped 必需任务；
3. 由 Host maintainer review/merge；
4. 记录 merge SHA，不能只引用通过的 CPU 子任务。

### 4.2 重放 Host PR #35

PR #37 合并后：

1. 将 PR #35 rebase 到精确 Host main；
2. 重跑 pre-commit、mypy、CPU UT 和选定测试；
3. 验证 default-off 路径与 Host 原行为一致；
4. 验证 provider eligibility、mixed batch、MTP2、hybrid attention/Mamba、
   capture/replay、fallback 和 post-acceptance fail-closed；
5. 等待 Host owner 给出架构 review，不自行合并核心 Host 修改。

### 4.3 真实 Manager 与 NPU 正确性

必须在 exact Host/plugin heads 上依次保留：

1. Manager `discover`、`check`、`plan`、`run`；
2. baseline 和 treatment 的相同输入输出对照；
3. TP2 两 rank 的 profile 分片与 layer coverage；
4. continuing-prefill、混合批、MTP2 accepted-token rollback；
5. graph capture、首次 replay、后续 replay；
6. unsupported layer/config fail closed；
7. fallback、rollback/uninstall、进程退出和设备释放。

任何 provenance、correctness、graph 或 cleanup 失败都立即停止，不进入性能测试。

### 4.4 执行唯一一组预注册测试

固定条件：

- 200 requests；
- 4096 input tokens；
- 256 output tokens；
- baseline：native BF16 KV；
- treatment：calibrated zero-offset INT8 KV + reviewed paged-INT8
  continuing-prefill provider；
- 相同模型、数据顺序、seed、软件 SHA、运行模式和设备；
- 每个 arm 最长 30 分钟，arm 结束立即清理并释放设备。

正向结果必须同时满足：

- 200/200 完成；
- 无 correctness、graph、fallback、cleanup 错误；
- baseline/treatment 输出 token 数一致；
- treatment median TTFT 至少改善 5%；
- treatment throughput 退化不超过 3%。

只完成一对运行属于 bounded pilot，不得外推为跨模型或通用性能结论。

### 4.5 原始数据与汇报

至少保留并公开引用：

- 每个 arm 的完整配置、命令、环境和 Git SHAs；
- 请求级原始记录、错误、TTFT、TPOT/ITL、throughput、峰值内存；
- 模型/profile/数据文件 digest；
- 服务日志、Manager 回执、graph/correctness/cleanup 回执；
- 原始产物 SHA-256 和一页结论。

将公开 URL 回传到插件 Issue #1/PR #4，并通知张老师或指定工程师。失败结果也
必须保留，不能调整阈值后改写成正向结果。

## 5. P2：恢复 MOD 公开展示

公开仓与 PR 形式只解决贡献载体问题，不等于 MOD 已可用。恢复网站展示前仍需：

1. revision-bound calibrated profile；
2. Host 已接纳且可达的 C8 selection/loading 和 provider；
3. 真实 Manager lifecycle；
4. NPU serving、数值正确性和 graph replay；
5. fallback、rollback/uninstall 和退出清理；
6. 独立绑定的质量与性能原始证据；
7. 负责人批准恢复 `public_surface`；
8. 通过公开网站 PR 恢复条目。

不得仅因仓库公开、wheel 可 import、静态 schema 通过或 manifest 改为 active
就申请恢复展示。

## 6. P3：重装后的恢复验证

迁移前备份已完成并验证 SHA-256。新集群可用后：

1. 从公开组织仓 clone 插件和 Host；
2. 从备份恢复仅本项目不可再生的原始结果与 dirty checkout；
3. 校验 `SHA256SUMS`；
4. 按 manifest 重建环境，不恢复旧 venv/conda 目录；
5. 重新下载精确 revision 的公共模型，不从本地备份复制大权重；
6. 先运行 CPU/无设备测试，再申请 NPU 资源。

## 7. P4：AgentX 的正确定位

`vLLM-HUST/agentx-bench@e0c34de` 是可选的 256k serving 榜单工具，不是当前
Qwen3.5 交付的替代品。只有部署负责人已经提供以下条件时才运行 smoke：

- 可信且已运行的 OpenAI-compatible endpoint；
- 原生至少 262,144 token window；
- streaming usage 与 `ignore_eos` 支持；
- frozen local tokenizer；
- 明确的服务器配置、资源租约和清理责任。

不得为了测试客户端包装器单独启动或占用 NPU，也不得把 synthetic replay 写成
模型质量结果。

## 8. 9 月 30 日前的执行顺序

1. **立即**：保持 PR #4 Ready，处理 reviewer 反馈并争取合并；同步请求
   Host/profile/model/resource 的五项最小决策。
2. **Host 决策返回后**：关闭 PR #37，重放 PR #35，形成 accepted revision。
3. **正确性输入齐全后**：执行 Manager/NPU correctness matrix。
4. **正确性全绿后**：运行唯一一组 baseline/treatment pilot，发布原始数据。
5. **结果满足门槛后**：向负责人回报，并申请网站恢复；否则保留失败证据并提出
   下一轮修复，不制造性能结论。
6. **新集群启用后**：执行恢复验证；AgentX 始终排在强制交付之后。

## 9. 最终完成判定

10 月前的完整目标只有在以下条件全部成立时才能标记完成：

- 至少一个公开 `vLLM-HUST` PR 已合并；
- PR #4 已 review/merge，或负责人明确接受其公开提交状态作为本轮交付；
- Host/profile/model/resource 决策有公开记录；
- Qwen3.5 一组 baseline/treatment 通过预注册门槛；
- 原始数据、哈希和清理回执已发布并通知负责人；
- 没有把私有研究仓、dry-run、静态 schema 或失败/跳过的 CI 写成已完成贡献。

若外部决策未在截止前返回，必须公开记录阻塞责任、已完成证据和无法执行的确切
前置条件；这可以证明已履行学生侧工作，但不能替代真机正向结果。
