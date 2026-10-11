# 浦江五数据集：适用性、阻塞与施工控制

更新：2026-10-12。维护者：`XilingGao`。本记录已准备，尚未得到负责人审阅。

## 0. 最新控制覆盖：V5.5

benchmark PR #275 已合并到 `9e7f83388d72a37767fd9f24636219093ecbe662`，
当前执行 authority 是 [V5.5 对齐控制](PUJIANG_V55_ALIGNMENT_20261012.md)。正式
35B/TP2/BF16、B0/B1、五数据集及 E1/E2 不启用保持不变；新增明确 B0 回执归属
发布与评测工程，13 项实际工件/运行材料须同 manifest 绑定并验签，禁止推填。
校验器已切换为原字节 V5.5 声明，拒绝 V5.4 作为当前控制输入；旧快照原样保留。
INT8 KV/B1 正式准入仍未决；身份生产者已知不等于材料已交付或 AQKV 已接纳。

Host main 已更新为 `82f586f8dcb5f92f8f3d55727bc05b3e2887606d`。八个目标 C8/graph
函数 AST 未变，七组提取源码/非数值 stub 对照完成；[目标路径核验设计](PUJIANG_PATH_VERIFICATION_PLAN_20261012.md)
已准备，真实回执仍为空。新 Host 依赖与本地 tokenizer 环境分开管理，不自行重置 pin。

以下各日期章节保留验证当时的状态。10 月 11 日增量已公开于
[Draft PR #12](https://github.com/vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust/pull/12)；
四项成功 CI 对应旧 head `65395c0`，不是本次 V5.5 增量通过。最新验证/发布状态按
确切 head 记录，不能把下面历史“未提交”或旧测试数量当实时状态。

### 0.1 历史 V5.4 同步记录

本轮再次同步发现 benchmark `b32fcbdaf8d3237936b35f9f4ba441bf2d64b361` 已发布
V5.4：强制 Qwen3.5-35B-A3B / 2 张 910B2 / TP=2 / BF16，仅 B0/B1，
E1/E2-G/E2-D 不启用。正式 SLO、生命周期成本、签名 manifest/OCI RepoDigest 和
禁止现场调参等规则以
[`PUJIANG_V54_ALIGNMENT_20261010.md`](PUJIANG_V54_ALIGNMENT_20261010.md) 为最新控制。
不能由早期 C8 计划推定正式准入；BF16 的 KV 精度范围与 AQKV 是否进入最终 B1
manifest 尚待明确，不擅自解释为课题被取消。历史资产/研究记录保留，正式执行仍阻断。

## 1. 要求与证据等级

[插件 Issue #10](https://github.com/vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust/issues/10)
是本模块的适配工单；[benchmark Issue #254](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254)
是评分合同、资产和运行结果的总账。主指标为任务质量，吞吐与延迟只能作为辅助。

10 月 9 日在 benchmark 精确提交 `e3a7183206bd399447475b2d117fef897c0880b6`
读取了 `dataset_program_v1.json` 和资产审计报告。原 JSON 原样保存在
[`evidence/PUJIANG_DATASET_PROGRAM_SOURCE_20261009.json`](evidence/PUJIANG_DATASET_PROGRAM_SOURCE_20261009.json)，
字节 SHA-256 为 `de1347eb1557f7e82ea3314c118ee090df1d4a87dd18e474c801b4be3a8185a4`，
已与同一 GitHub 提交的原文件重新核对。

10 月 10 日早轮已固定共享实现到
`8ed7ec9a16075c5bce5644a68a712139a6e5d403`，并在本地逐字节确认其 program JSON
与上述快照相同。**总账 JSON 未变不等于没有进展**：PR #258-263 / #265 已补充
共享实现、PyramidKV 候选协议与回执。矩阵中的 `canonical_blockers` 保留冻结总账的
原描述；`shared_progress` 与 `current_blockers` 分别记录新成果与当前剩余条件，不能
改写旧快照，亦不能把其他 MOD 的结果当成 AQKV 成果。随后观察到 `b32fcbd` 的
program 增加 V5.4 声明并规范三个外部路径；旧快照不再是当前执行 authority。
三份复用评分/计划文件逐字节未变，来源绑定仍保留原精确实现 pin。

10 月 11 日只读刷新至 `402a5e41a1b2b0363858c0b77586e1aa92723396`：新增
PR #272/#274 只涉及其他 StateAxis MOD 的准入审计。逐文件重新取得 V5.4 声明及
三份复用文件，与原固定字节相同，不更换既有 pin；别的 MOD 准入不是 AQKV 许可。
插件 #10、Host #37 两条已发布请求仍无新的负责人回复；PR #35/#37 仍 Open/Draft。

本轮代码复用审查见
[`PUJIANG_SHARED_REUSE_REVIEW_20261010.md`](PUJIANG_SHARED_REUSE_REVIEW_20261010.md)。

这是公共总账的本地只读快照，不是我们在执行容器完成的 1,135 文件校验。
我们没有连接远端、验证当前容器入口或执行 NPU 任务。

结构化适用性矩阵：
[`evidence/PUJIANG_AQKV_APPLICABILITY_20261009.json`](evidence/PUJIANG_AQKV_APPLICABILITY_20261009.json)。
数据、评分合同、插件运行状态、实际测量四层分开记录；目前不含任务成绩或性能数值。

## 2. 为什么不能直接说“五项都支持”

本模块的目标是让 full-attention 路径消费分页 INT8 KV，并处理 cached multi-token
continuing-prefill。单纯使用 Qwen3.5 生成文本不证明进入了该路径：新请求可能是
all-new prefill，后续可能仅是 decode，缓存命中也未必产生符合资格的多 token query。

Qwen3.5 混合模型中的 linear-attention/Mamba 状态不是本模块宣称优化的普通 attention
KV。十个 full-attention 层及其校准范围按既有精确模型审计保留，但该审计仍需在新 Host
和实际执行环境核验。不能将所有模型层的状态都计入 INT8 KV 收益。

每项都必须保留 eligible dispatch、实际 cache hit、原生 fallback 与未作用请求的证据。
若 B1 未触发目标路径，结果应说明未作用，不能把服务噪声或其他机制的收益归因给插件。
不得为了制造命中而改变正式评测任务内容、工具预算或评分合同。

## 3. 五项矩阵

下表 `blocked` 是目前的正式执行状态，不是断言该数据集永久不适用。机制关联只是假设，
尚无该数据集上的运行回执；后续据实更新为老师要求的 `applicable` / `not-exercised` /
`blocked`，并保存变更依据。

| 数据集 | 源资产 / 本模块状态 | 与模块的潜在关系与需观测条件 | 正式评测缺口 | 首要指标 |
| --- | --- | --- | --- | --- |
| MMLU-Pro | asset-frozen / blocked；源题/token/scorer 与已存原始回执的本地连接已具备 | 知识题使用 full-attention KV；单题新请求不保证命中或 continuing-prefill，需实际路径观测；不继承 PyramidKV 4096 门槛 | V5.4 准入、正式选题/scorer 合同、服务端 token parity、实际有界 transport 与运行门禁；离线绑定不验证实际模型成绩 | 固定 evaluator 的 accuracy |
| HLE-Verified | asset-frozen / blocked；来源/模态审计已具备 | 目标模型作答与 judge 评分须分离；长推理本身不证明缓存复用；Gold 包含图像题 | 权利审查、full/Gold 任务范围、实际 judge、图像处理及执行环境；不得丢弃图像题冒充完整成绩 | 固定验证集与 grader 的 accuracy |
| SWE-bench-Pro | asset-frozen / blocked；642 个 V2 任务映射已审计 | tool-result / 对话续接可能产生 cached multi-token prefill；须记录真实 dispatch；保留 harness 原字节 | 权利审查、实际 AMD64 fresh-sandbox、镜像层、agent/tool budget、regrader 与服务配置 | 隔离复判 resolved rate |
| FrontierScience | asset-frozen / blocked；分赛道输入及候选 grader 已具备 | 两赛道均可能使用 KV，但不能假定命中；按赛道和源行号记录，不能用 group ID 去重 | 候选协议审阅、可用且固定的 judge、作答前绑定的真实语义校准及环境；不是官方 grader 复现 | 两赛道各自准确率或 rubric score |
| Terminal-Bench 2.1 | asset-frozen / blocked；digest 元数据与候选 Harbor 计划已具备 | agent/tool 历史可能续接 KV；需区分 target server 与 Harbor 成本；job-b0 文件不证明真实服务开关 | 权利/协议审阅、AMD64 后端、镜像层、实际后端到模型的连接/认证、fresh-sandbox verifier/reference/failure 检查 | task resolution / pass@1 |

HLE 源快照 2,500 行不等于已决定评测 2,500 个正式任务。FrontierScience 为 100 个
Olympiad 与 60 个 Research；Terminal-Bench 2.1 为 89 个任务定义，均来自冻结总账，
不是已完成测量的样本数。

PyramidKV 70 题各 55/70 的结果仅是该模块的命名子集兼容性证据；没有继承其任务
选择、压缩资格、运行激活检查或分数。MMLU `choose_tasks` 会加入超过压缩门槛的
题，`validate_activation` 要求直接启用与空 Manager bundle list，`analyze` 依赖
PyramidKV transaction 日志，这三段都不能用于 AQKV 验收。

## 4. 共享阻塞与分工

| 工作 | 责任与当前边界 | 可立即进行的工作 | 进入执行前必须获得的输入 |
| --- | --- | --- | --- |
| 模块矩阵、适配、证据格式 | XilingGao | 实现本地接口与拒绝错误输入的测试；准备 Issue #10 回执 | 负责人审阅，适用项的正式评分合同 |
| Host/provider 接纳 | ShuhaoZhangTony，依据 Issue #1；具体实现者另行确认 | 维护候选接口依赖、梳理当前源码差异 | 已接纳且可达的精确 Host revision，不以 Draft PR 代替 |
| 精确模型 C8 profile | 交付负责人待确认 | 核验字段/分片/zero-offset 要求，不编造校准值 | 模型 revision 绑定的校准 provenance、profile/shards 与加载路径 |
| 评分合同与环境 | 总账 Issue #254；各项具体 owner 待确认；不把 PyramidKV owner 指定为 AQKV owner | 复用已合并共享工具，单独审阅 AQKV 协议；候选 grader/Harbor 计划不等于正式合同 | task manifest、实际 judge/校准、许可记录、真实后端/连接/镜像、agent/tool budget |
| PR #37 CI | XilingGao 排查；runner 支持需 Host maintainer | 只读核对 job/日志，准备精准请求 | 所需 runner 可执行，设备任务与总 gate 完成 |
| 正确性与资源 | XilingGao 验证；资源分配者待确认 | 准备 opt-in、失败退出、回滚与清理回执 | 可启用 provider、Manager 生命周期、NPU correctness、当前租约/环境 |

没有为外部维护者猜测个人分工，也没有将 Host 最终交付责任转移给插件侧。
许可、环境或接纳输入未齐不妨碍继续做其他独立的本地开发。

## 5. 最新基线与 PR #6 去向

- 插件基线：`9f9844ec7d73bdd627edf3930f3ec6cb5ef5f325`，含合并的 PR #4、#5 及
  老师新增的 PR #11（manifest `0.3-experimental`）。默认仍为 `import_only`。
- Host main：`f939ad767b60d60253c3e20c7522defe69e4ca78`；只固定源码身份，不声称已运行。
- Host PR #35：Open/Draft，`084f70f50dfcdf2daf66b3a31813bc982c2d1d09`，
  base `fbe4911bb54ce493b3fcbbf6238b032b9dc07ec6`，不能当作新 main 上已接纳版本。
- Host PR #37：Open/Draft，`3fb17f2a9eb140cd785d390645a9c743344f9d29`，
  base `17f68177472b896facf3f0501d424d24e16e699a`，也未基于当前 main 接纳。
- 原主工作目录的未提交修改和 ahead commit 均原样保留；本轮使用原本干净的控制工作目录，
  新分支 `feature/pujiang-five-contract-20261009` 从上述公共 main 创建。

PR #6 已关闭且未合并，无关闭原因说明。其 head
`1de483cc64103215560246b97c0c17a4ad2ab084` 对 base
`8733e1b8f8c71c7962de8ac27382fa0a5ae563c4` 的四个新增文件均不在当前 main；
README 增加的对应链接也未进入主线：

| PR #6 内容 | 当前主线情况 | 处理决定 |
| --- | --- | --- |
| README 的 pilot-result/checklist 链接 | 对应四个新文件缺失，不能直接添加悬空链接 | 不复制旧链接；改为本次五数据集控制入口 |
| 节后执行 checklist | 新文件缺失 | 原分支保留；以本文件的新主线控制代替旧优先级，不将旧 pilot 视作 primary 验收 |
| prefix-pilot result validation 文档 | 新文件缺失 | 保留为 supplementary 方案，后续单独审阅 |
| `validate_qwen35_prefix_pilot_result.py` | 新文件缺失 | 原 521 行代码保留；哈希、配对、失败和清理校验可复用，200 请求 / 5% TTFT / 3% 吞吐阈值不可当作五项质量验收 |
| 对应 result-validator tests | 新文件缺失 | 原 358 行测试保留；未来伴随实际复用部分提交，不宣称已合并 |

“未进入主线”是源码对照结论，“为什么被关闭”仍未知。没有重新打开或重复提交该 PR。

## 6. Host CI 精准诊断与下一步

已读取 PR #37 `ready-all` run `36372359941` 的 job 回执：

- `pre-commit` 与 CPU UT 通过；总 `ci-gate` 失败，所选设备任务被跳过。
- [select-tests job 108775478306](https://github.com/vLLM-HUST/vllm-ascend-hust/actions/runs/36372359941/job/108775478306)
  的 API conclusion 为 `cancelled`，时间为 2026-09-28 03:28:01Z 至 09-29 03:28:02Z。
- runner label 为 `linux-amd64-cpu-8-hk`，`runner_name` 为空，`steps=[]`。
  这说明该回执没有选择测试步骤的执行证据；约 24 小时后取消。不能据此诊断成产品测试断言失败，
  也不能断言已找到 runner 不可用的最终原因。
- 老师仍要求 GLM-Next 产品改动的设备验证；CPU 绿灯不足以合并。未重新触发 CI 或用卡。

11 个文件已逐项核对：7 个测试文件仍需维护，4 个旧改动已覆盖或不适用；候选补丁
仅测试变更，18 行新增、8 行删除。四处接口差异用固定源码片段与显式 stub 实际复现，
候选下四处通过；随后另补五组 Mamba/Attention/Triton 对照，共九组片段/参数化复现，
不是完整 Host UT、真实 Triton 安装、CI 或 NPU 结果。原 Host worktree 与 PR head 保持不变。
下一步确认最小测试差异的公共提交路线和真实双版本 CI；Host 产品改动的设备门禁
仍保留，不因测试拆分绕过要求。用户授权后已将支持请求与变更范围发布到
[Host PR #37](https://github.com/vLLM-HUST/vllm-ascend-hust/pull/37#issuecomment-6097950276)，
不将留言成功记为维护者批准。

## 7. 本轮完成与后续门禁

- 五项矩阵、canonical snapshot、缺口/分工、PR #6 内容去向已本地准备；10 月 10 日
  增加最新共享进展及剩余阻塞，原快照与原始主张边界保留。
- 本地校验器检查固定五项、source SHA-256、版本、指标和缺口，拒绝伪造 runtime/成绩；
  输出 `planning_valid_execution_blocked` 只证明规划记录一致，不授权执行。
- MMLU 离线配对接口已复用精确版本的 `score_results`，绑定 supplied manifest / token IDs / raw
  outputs / 文件哈希，并拒绝缺任务、重复、请求不匹配和真值伪装。失败及截断保留在分母。
  删除 PyramidKV 专用 eligible/ineligible 字段，不生成 AQKV 资格或性能结论。
- 离线输入仅允许 development-fixture / unverified-stored-outputs；原 v1 不核验源题成员身份、
  正式任务选择、真实原始网络回执、模型/Host/激活身份或权限，所以不是完整 runner/验收器。
- FrontierScience 离线接口按赛道/源行号配对，绑定 judge 请求/原始响应、指令及校准
  文件；未评分/失败/截断保留。fixture 评分与真实 task_quality 分离，不凭结构校验
  宣称实际 judge 语义校准或作答前冻结。
- Terminal 候选计划已复用固定 job_config，约束任务/预算/重试与计划开关；仍不验证
  真实任务资产、Harbor schema、服务切换、后端或 verifier。两项细则见
  [`PUJIANG_FRONTIER_TERMINAL_OFFLINE_20261010.md`](PUJIANG_FRONTIER_TERMINAL_OFFLINE_20261010.md)。
- 尚未冻结正式任务清单、评分器、许可、镜像或 B0/B1；矩阵未审阅。用户授权后已在
  [Issue #10](https://github.com/vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust/issues/10#issuecomment-6097948895)
  请求确认 V5.4/B1 准入与身份 owner；代码未推送，未取得批准，Slack/benchmark 长稿未发送。
- 未执行正式评测、未确认当前远端资产可访问、未改变激活状态、未恢复网站展示。

```bash
python scripts/validate_pujiang_readiness.py
python -m pytest -q tests/test_pujiang_readiness.py
```

10 月 10 日随后完成 MMLU 官方源题/提示构建与 70 题候选请求计划，实际源数据
为 test 12,032 / validation 70 / 14 学科。已核验固定 Qwen3.5 tokenizer/模板字节，
70 题真实提示长度 913–2147，候选预算下全部不越界且无截断；请求/选题没有改变。
服务端 tokenizer/模板/依赖与 token parity、transport、正式选择合同与路径回执仍未齐。见
[`PUJIANG_MMLU_SOURCE_PLAN_20261010.md`](PUJIANG_MMLU_SOURCE_PLAN_20261010.md)。

Host #37 门禁源码已离线核查。最新 main `f939ad7` 已改变初始化、GLM 缓存布局和
CPU 双版本工作流，已完成 11 文件分类与最小测试候选，不重放旧产品分支。见
[`HOST_PR37_GATE_AUDIT_20261010.md`](HOST_PR37_GATE_AUDIT_20261010.md)。
随后已将真实源题和 token plan 接到 scorer cases：v2 模式重建固定来源并核对完整
bundle/生成参数及 B0/B1 请求摘要；旧 v1 不自动升级。70 题接线检查使用全失败的
development fixture，没有模型调用/成绩。公开只保留摘要，私有题目/答案不入仓。
该接线轮本地回归 603 passed，含真实官方源资产与 tokenizer 的 opt-in 集成；
仍无 CI/NPU/Host 接纳或正式任务成绩。

随后新增已存 request/response 解析：逐题核验原始字节、相对路径与 SHA，完整请求
必须等于来源绑定候选请求；超时/HTTP 错误/坏响应保留失败分母，缺失或损坏原始文件
则阻断整个报告。70 题两臂 fixture 覆盖 20 条可解析合成答案、10 条 length 和
50 条失败（length 属于前 20 条，不另外加分母），模型请求数为 0。细则见 MMLU 文档
第 7 节，不能把存储回执校验或 usage 数量匹配当成真实 transport/token parity。
V5.4 控制投影另有旧版本/C8/伪准入/可选排队拒绝测试；正式门禁不开放。
最终本地插件回归 **671 passed**，固定复用 benchmark 测试 **57 passed**，
新 V5.4 上游合同测试 **4 passed**；各自 XML 零失败/错误/跳过，
不是公共 CI、Host 接纳、设备验证或正式任务成绩。

接下来审阅 Host 最小测试补丁并安排合法 CI，同时推进 MMLU 服务端 token parity/
有界 transport 与路径回执、
Issue #10 矩阵审阅和 owner 输入；完整正式 runner 尚未完成。
只有评分合同、可达量化路径、真实正确性和当前资源门禁同时满足的项目才能进入 B0/B1。

## 8. 10 月 11 日本地有界 HTTP 增量

已完成 [loopback HTTP 开发验证](PUJIANG_LOOPBACK_TRANSPORT_20261011.md)：仅合作式
合成服务，44 项新增测试和十个持久 HTTP 场景；完整插件 **715 passed**，零失败/
错误/跳过。绝对时限、不重试、不跟重定向、正文大小/截断拒绝、坏元数据审计与
自有子进程回收均覆盖；旧结果保留，正式远端 executor 尚未实现。

源文件、原始 request/response、服务端计数与 SHA 已本地记录，未推送公共代码、
发布新留言或发送 Slack。没有启动模型/NPU/沙箱、变更运行实现或创建自动任务。
当前待办仍是 V5.4 准入/B1 身份、Host 正常提交/CI 路线、真实服务 parity 和正式合同。
证据先进入公共 benchmark，再回链 Issue 和同步网站。Slack 仅拟稿，不自动发送。

## 9. 10 月 11 日完整 token IDs 增量

完成 [stored token parity 评分门禁](PUJIANG_MMLU_TOKEN_PARITY_20261011.md)：
v2 scorer 先重建来源，再核验两臂逐题已存完整 IDs；任一失败不调用 scorer，保留
全部题目诊断并非零退出。仅数量匹配不能通过。两组 70 题/臂实际来源绑定检查使用
复制本地 IDs 的合成响应，不是服务验证；网络 tokenize/模型请求均为 0。

完整插件本地 **780 passed**，零失败/错误/跳过；新增 64 项 parity 与 1 项严格
布尔握手回归。旧回执保留，公开精简摘要不含原题/答案/tokens。实际服务身份、
签名与正式 transport 仍缺，五项正式执行继续 blocked，V5.4 准入规则未放宽。

## 10. 公开 Draft PR 的审阅范围

用户于 10 月 11 日明确授权将这些准备工具、测试和文档作为插件仓库的公共
Draft PR 送审。各轮章节中的“未提交/未推送”保留其验证时点，不代表本 PR 的
实时发布状态；发布链接、精确 head 和 CI 以 GitHub PR/API 回执为准。

本 PR 只交付 source/token/scorer、stored transport/parity、本地 HTTP fixture
与五项控制文档。没有运行时包、生产 Host、工作流、选择器或设备 gate 改动，
没有原题/答案/tokenizer 权重/私有回执。默认发行入口仍为 import_only。
Host 七文件测试维护候选不在本 PR 内，PR #35/#37 及其合法 CI/设备要求保持不变。

780 项是本地显式 opt-in 验证，不是公共 CI 结果。默认 CI 没有私有源资产/
tokenizer/上游源码目录时，相应 opt-in 测试会明确 skip；这些未运行集成不能记为
服务或设备验收。Draft 发布不等于审阅/接纳，也不解除 V5.4、运行或网站门禁。
