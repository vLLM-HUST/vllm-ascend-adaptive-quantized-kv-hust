# 2026-10-10 增量反馈记录与待提交草稿

**2026-10-12 覆盖说明：** 当前 authority 已切换到
[V5.5](PUJIANG_V55_ALIGNMENT_20261012.md)。下文 V5.4 留言/草稿保留历史时点；
其身份 owner 问题现已由新版明确为发布与评测工程角色，仍无实际材料或 AQKV/B1
准入回复。本轮没有发送新的 Issue、Host 或 Slack 消息。

以下为各轮发送前草稿，保留当时尚未推送代码的状态；后续获授权的公共 Draft
审阅范围见 [五项控制第 10 节](PUJIANG_FIVE_DATASET_CONTROL_20261009.md)。未发送稿
不能写成已发 Issue/Slack，公共 Draft 也不等于 CI/验收通过。长稿保留供审阅，
用户本轮明确授权后已发送两条精简 GitHub 留言，见文末实际回链；benchmark 长稿和
Slack 摘要仍未发送。Slack 未经用户明确许可不得发送。

本轮再次同步发现 benchmark 已发布 V5.4（精确提交 `b32fcbd`）：强制
Qwen3.5-35B-A3B / 2 张 910B2 / TP=2 / BF16，仅 B0/B1，E1/E2 不启用。
本地控制已对齐；历史资产和 C8 研究记录保留，不作为正式执行 authority。
请明确 BF16 约束的 KV 精度范围及本模块是否纳入最终 B1 签名 manifest；
我们不会自行认定 INT8 KV 已获准或已被禁止，也不为可选项排测量队列。

## 插件 Issue #10

已基于插件公共 main `9f9844ec7d73bdd627edf3930f3ec6cb5ef5f325`
（含 PR #4/#5/#11、manifest 0.3）准备最新五数据集适用性与阻塞矩阵。
本轮矩阵和接口增量仍在本地工作分支，尚未推送或提交公共 PR。

MMLU-Pro、HLE-Verified、SWE-bench-Pro、FrontierScience、Terminal-Bench 2.1
的源数据状态按 benchmark `e3a7183` 记录为 `asset-frozen`；本模块当前五项均为
`blocked`，不是永久不适用：Host/provider 接纳、精确 Qwen3.5 C8 profile、可启用
实现和真实正确性回执尚未闭合，各项评分合同/环境也未齐。

已审查共享 benchmark `8ed7ec9` 上合并的 MMLU、FrontierScience、Terminal 工具。
program JSON 与旧资产快照逐字节相同，旧总账缺口保留为历史；最新共享实现及当前
实际缺口单独入矩阵。不把 PyramidKV 的 70 题成绩或本地 API 请求记作 AQKV 成绩。

MMLU 本地离线接口直接复用固定源码的 score_results，保留失败与截断分母，核对
完整任务配对、token ID/输入与文件哈希，并移除 PyramidKV 专属的 4096 eligibility。
未继承其选题、直接激活或 transaction 路线。插件最新本地回归 671 passed；共享工具
相关测试 57 passed，V5.4 上游合同测试 4 passed。均无正式数据集执行，不是 CI/NPU/Host 运行验收。

随后实际核验 MMLU 官方固定 Parquet/evaluator/提示：test 12,032 行、validation
70 行、14 学科；生成各学科前五个 test ID 的 70 题候选五样例消息/请求计划。
原题/答案不进入公共仓库；选题/预算尚未审阅。固定 Qwen3.5 tokenizer/模板的本地
核验已完成，70 题长 913–2147 tokens，加候选生成预算后全部不越界，无截断/删题。
服务端 tokenizer/template/依赖与 token parity、transport 尚未核验，模型请求数为 0，
不是该子集已取得准确率。

源题/token plan 到 scorer 的连接已补齐：新增来源绑定模式重新读取固定资产、
selection 和 tokenizer，拒绝对题号/答案/顺序/token/完整生成请求的篡改；两臂逐题
摘要必须匹配。70 题实际接线检查只使用明确标记的合成失败 fixture，保留完整分母，
没有生成真实 B0/B1 作答、正式准确率或任何性能主张。

已补存储原始请求/响应到 scorer 的离线连接：核验真实字节与完整请求，完整记录的
超时/HTTP 错误/坏响应保留分母，缺失或损坏文件阻断报告。新 70 题两臂 fixture
覆盖合成可解析/length 响应和五类失败，模型调用仍为 0；usage 数量匹配不等于
服务端 token parity，记录的 attempts 不等于实际无重试。旧全失败 fixture 原样保留。

FrontierScience 离线接口已按赛道/源行号保留任务并绑定 judge/校准摘要和原始请求/
响应；外部结构绑定不算真实语义校准，不产生成绩，坏判词与未判分保留。
Terminal 候选计划已复用固定 job_config，检查任务/预算/重试/计划开关，实际服务
激活和后端仍未验证。三项离线接口不是正式执行合同，所有 claim gates 仍为 false。

知识/科学题并不自动证明发生 continuing-prefill；agent/tool 对话续接可能触发，
但也需真实 cache-hit / eligible-dispatch / fallback 记录。Qwen3.5 的
linear-attention/Mamba 状态不计入本模块普通 full-attention KV 优化范围。

任务质量按固定 scorer 的 MMLU-Pro/HLE 准确率、SWE-bench-Pro fresh-sandbox
resolved rate、FrontierScience 两赛道独立评分、Terminal-Bench 2.1 Harbor
task resolution 记录；吞吐和时延仅为辅助，历史 prefix pilot 不替代正式验收。

下一步推进本地适配与证据接口，同时需要确认：

- 请审阅五项矩阵与具体适用性判断。
- 请确认 Issue #254 中各评分合同/许可/镜像的交付 owner 和可用输入。
- Host 接纳与精确校准 profile 交付仍按 Issue #1 分工；不会以 Draft/CPU CI
  替代真实运行，也不会在未分配设备上启动正式实验。

PR #6 关闭未合并的四个新增文件已逐项对照，原分支仍完整保留；结果校验逻辑暂按
supplementary 复用候选处理，不把其 TTFT/吞吐阈值当作五项任务质量标准。

本次只有本地记录与测试，无正式数据集执行、任务质量数值或性能主张。

## benchmark Issue #254

本模块已按 `e3a7183` 准备五项矩阵，源资产均记 `asset-frozen`，正式执行均记
`blocked`；prepared 与 reviewed 分开管理。共同缺口为可达且接纳的量化路径、
校准 profile、真实 Manager/NPU correctness 及当前资源入口；逐项评分/许可/镜像
缺口保持与 canonical 总账一致。

已更新最新共享进展：MMLU scorer 可复用；FrontierScience 分赛道输入/候选 grader
已具备但实际 judge/语义校准未齐；HLE/SWE 来源与 Terminal digest/候选计划不再
笼统标记缺实现，仍缺权利、图像执行、真实 AMD64 后端与后端到模型的连接/认证，
以及 fresh-sandbox verifier。Harbor job 名称与实际 B0/B1 服务开关分开核验。

请确认实际 judge/校准、正式评分协议与许可审查，以及 SWE/Harbor 后端的 owner
和接入方式；Harbor 候选预算不能自动升为 AQKV 已批准合同。
V5.4 合同已同步到本模块控制；签名 B0/B1 manifest、OCI RepoDigest、完整服务/客户端
合同与运行回执仍未取得。不能从已有 BF16 测量或原 C8 研究推定本模块正式准入。
我们继续推进本模块适配；仅重取约 4.2 MB 的 MMLU 固定源文件做本地核验，
其余已有资产优先复用、不下载模型权重或约 40 GB 镜像，也不会
从历史 B0 serving telemetry 推定准确率。真实结果将先提交 benchmark，再回链
插件 Issue #10 和网站。

## Host PR #37

已核对老师指出的 source-change gate。`ready-all` run `36372359941` 中 CPU UT
与 pre-commit 通过，设备任务跳过、总 `ci-gate` 失败。
[select-tests job](https://github.com/vLLM-HUST/vllm-ascend-hust/actions/runs/36372359941/job/108775478306)
的 API 回执为 `cancelled`：label `linux-amd64-cpu-8-hk`，runner name 为空、
steps 为空，约 24 小时后取消，没有实际选择/执行测试的证据。

原 head 的 coverage 与 --all-tests/--skip-cpu-ut 选择器离线检查均 exit 0；15 个
设备分组、132 个目标出现次数，不是 CI/设备通过。最新 main `f939ad7` 已统一初始化
allocator、改变 GLM 缓存布局，并改为 CPU 双版本工作流；旧修复不能原样重放。

11 文件已逐项核验固定主线源码：7 个测试文件仍需维护，4 个旧改动已覆盖或不再适用。
本地最小候选只有测试变更，18 行新增、8 行删除，删除旧 GLM 专用产品分支的重放计划；
最新布局和已修的测试不倒退。首轮四处接口错误及随后五组 Mamba/Attention/Triton
对照已实际执行：原 fixture 失败、候选下通过。共九组源码片段/参数化对照，不是完整
Host UT；Triton 仅合成 package 导入机制，没有运行真实安装/Gluon、CI 或设备测试。
没有推送/更新 PR head。
请确认是更新本 PR 还是独立提交测试维护，及所需 runner/双版本 CPU 合法重放路线。
Host provider/产品差异仍需适用设备用例和正式接纳，测试修补不解除此要求。
设备正确性与总 gate 仍需实际通过，不会将 CPU 证据当作可合并或删减门禁。PR 保持 Draft。

## Slack 手动发送摘要

老师，我已根据共享 benchmark 的新进展更新五项矩阵，并审查 MMLU、FrontierScience
和 Terminal 工具。MMLU 离线配对评分接口已复用固定版本的公共 scorer，保留失败和
截断计分，不继承 PyramidKV 的资格门槛或激活方式。FrontierScience 分赛道/校准
绑定接口和 Terminal 候选计划配对校验也已完成；本地全量测试 671 项通过。
MMLU 官方源题与固定 Qwen3.5 tokenizer/模板已本地核验；70 题长 913–2147 tokens，
预算下无越界、截断或删题；已接通来源绑定离线评分，70 题全失败 fixture 检查
保留完整分母，尚未调用模型或取得真实作答成绩。
本轮已补 raw request/response 的离线解析和五类错误保留，仍不等于真实 transport。
同时按最新 V5.4 对齐 BF16 交付和未启用可选项；还请明确 INT8 KV 的正式准入范围
及本模块是否进入最终 B1 manifest，旧量化研究不自动作为正式执行许可。
源数据已冻结，但正式评分、Host/校准 profile、真实正确性与执行环境仍有缺口，
所以没有启动正式评测或写性能成绩。下一步补服务端 token parity/transport 与路径回执。
PR #37 的 CPU 测试通过，但选择测试 job 约 24 小时后取消且无执行步骤，需要确认
runner 与门禁重放路由。11 文件已核对，最小候选只剩 7 个测试文件的修正；四处接口
差异及新增五组 Mamba/Attention/Triton 对照都以片段/stub 复现并修正，共九组，
仍需真实新基线 CI；Triton 不是实际安装验证，Host/provider 的设备门禁保持不变。
待公共 PR 链接补齐后，会把具体矩阵和阻塞回链 Issue #10/#254。

## 本次拟发送的精简 GitHub 留言

以下保留发送前的精简稿；实际发送版本与链接见下一节，其中 Host 留言另明确更正了
此前不准确的“仅测试改动”描述。旧长稿保留为工作记录；本地代码没有公共链接，
不能写成已提交或可在远端审阅的附件。两条均不请求跳过 CI/设备门禁。

### Issue #10：确认 V5.4 准入和服务身份

已同步 benchmark V5.4（`b32fcbd`）：正式交付为 Qwen3.5-35B-A3B / 2×910B2 /
TP=2 / BF16，只启用 B0/B1；E1/E2 不排执行。历史 C8 研究和 70 题开发子集均不作为
当前正式执行依据。

本地已接通固定源题、tokenizer、stored request/response 与公共 MMLU scorer；
插件 671、固定共享测试 57、新上游合同测试 4 项通过。响应仅为合成 fixture，
模型调用为 0，不代表正式准确率、Host 接纳或性能结果。增量尚未提交公共 PR。

为避免错误启动量化臂，请确认：

1. V5.4 的 BF16 约束是否同时约束 KV cache 精度？AQKV/INT8 KV 是否纳入最终
   B1 签名 manifest？当前不会自行判断已准入或已禁止。
2. 正式 B0/B1 release/OCI RepoDigest、core/plugin/model/tokenizer 与 server/client
   冻结身份由谁交付？如果本模块暂不纳入 B1，请明确正式交付范围。

确认范围与身份后，继续按正常 Host/provider、profile、correctness 和资源门禁推进；
不以 Draft/CPU 通过代替真实运行验收。

### Host PR #37：确认新基线最小测试维护路线

重新核对 main `f939ad7` 与当前 PR head `3fb17f2`。主线已统一 allocator、更新 GLM
布局和双版本 CPU 工作流，因此不能原样重放旧产品分支。11 个原文件逐项核对后，
4 个旧改动已覆盖或不再适用，剩余最小候选只修改 7 个测试文件（+18/-8），不改
生产代码、工作流、选择器或 gate；尚未推送或更新本 PR head。

首轮四组加新增五组，共九组固定源码片段/显式 stub 对照：旧替身/断言失败，候选
通过；保留混合页大小不能整除时拒绝的现行行为。Triton 只验证合成 package 的导入
机制，未运行完整 Host collection、真实 Triton/Gluon 或两套 CI 依赖环境。

原 select-tests job 的 API 为 cancelled、runner name/steps 为空；离线 coverage
与选择器可运行，仍不能据此确认原调度根因或解除总 gate。

请确认是更新此 PR，还是另开仅测试维护 PR，以及可用的正常双版本 CPU runner/
CI 路线。Host provider/产品路径仍需适用设备回归和正式接纳，不请求绕过这些条件。

## 用户授权后的实际发送记录

- [插件 Issue #10 留言](https://github.com/vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust/issues/10#issuecomment-6097948895)：
  2026-10-10 13:23:08 UTC，由 `XilingGao` 发布；询问 V5.4 BF16/KV 精度范围、
  AQKV 是否纳入 B1 及签名服务身份的交付 owner。没有将 fixture 写成实际成绩。
- [Host PR #37 留言](https://github.com/vLLM-HUST/vllm-ascend-hust/pull/37#issuecomment-6097950276)：
  2026-10-10 13:23:19 UTC，由 `XilingGao` 发布；更正原 PR 产品改动的范围描述，
  汇报七文件最小候选及九组片段对照，请维护者确认提交/正常 CI 路线。

发布后通过 comment ID 重新读取两条正文，核对作者、链接、ID 与实际发送文本。
这只完成沟通回链，不代表准入批准、公共代码提交、CI 通过、合并或 Host 接纳。
没有发送 Slack、触发 CI、变更 PR head、预约设备或创建自动化。

## 10 月 11 日增量草稿：未发送

再次核对两条确认请求，尚无负责人新回复。benchmark 的新 StateAxis 准入审计
不属于 AQKV；V5.4 声明与三份复用源码原字节不变，既有来源 pin 保留。

本地补齐有界 HTTP 开发验证：仅字面量 loopback 与显式合作式合成服务，校验
总时限、无重试/重定向、响应上限/截断、元数据损坏审计和子进程退出。新增 44 项
测试，完整插件 715 项通过，另保留十个实际本地 HTTP 场景的原始回执；模型调用为 0。
这不是正式 runner、服务端 token parity、Host CI 或任务成绩；代码尚未提交公共 PR。

下一步仍需 V5.4 KV/B1 范围与签名身份、Host 提交/正常 CI 路线，以及真实评分
合同/运行输入。正式量化路径和设备门禁不变，不以本地测试通过要求跳过验收。

### 10 月 11 日后续增量：未发送

已补 MMLU 完整 token IDs 的已存回执对照，来源绑定 v2 的两臂逐题校验先于评分；
数量相同但 ID 不同仍阻断，保留所有题目诊断，不产生质量分数。70 题/臂对照使用
复制本地 IDs 的合成回执，模型与 tokenize 网络请求均为 0，不是实际服务验证。
新增 64 项 parity 与 1 项严格布尔握手测试，完整插件本地 780 项通过、无 skip。

实际 endpoint、签名身份、Host 正常 CI、准入与正式选题/scorer/许可仍需负责人输入。
公共代码仍未提交；本段只是草稿，没有新增 Issue/Slack 留言或自动任务。
