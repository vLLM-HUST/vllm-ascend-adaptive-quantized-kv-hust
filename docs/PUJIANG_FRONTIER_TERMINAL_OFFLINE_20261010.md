# FrontierScience / Terminal：离线证据接入

日期：2026-10-10。状态：本地开发完成，尚未提交或得到合同审阅。
两个工具均不调用模型、judge API、Harbor、Docker 或设备，不下载任务资产或镜像。
只读取用户指定的已有 JSON；新报表拒绝覆盖旧文件。

## 1. 固定源码与完成边界

共享 benchmark 固定为 `8ed7ec9a16075c5bce5644a68a712139a6e5d403`，不能传浮动 main。
共享加载器 [`../scripts/pujiang_shared_source.py`](../scripts/pujiang_shared_source.py)
先核对 Git 根与 HEAD，再验证全部目标/依赖源码的 SHA-256，直接执行已校验字节。
临时依赖注册在成功或失败后恢复；不依赖环境里另一个 benchmark 安装。
MMLU 离线工具也复用该加载器，原评分语义和资格边界未改变。

| 接口 | 固定共享来源 | 本轮已完成 | 没有证明的内容 |
| --- | --- | --- | --- |
| FrontierScience | `frontierscience_inputs.py` / `frontierscience_pyramidkv.py`，哈希见复用审查文档 | 分赛道源行身份、配对、预算、judge/校准摘要及原始 judge 请求/响应校验；候选 parser/aggregate 复用 | 官方源题成员身份、实际 tokenizer 未泄漏答案、真实 judge 语义校准、真实作答前冻结、运行接纳或成绩 |
| Terminal-Bench 2.1 | `audit_pyramidkv_terminal.py` / `prepare_pyramidkv_terminal.py`，哈希见复用审查文档 | 固定候选 `job_config`、任务与预算配对、摘要、计划开关与实际服务身份分离 | 真实任务资产/文件权限、镜像层、Harbor schema/后端、模型连接、服务切换、沙箱/verifier 或任务成绩 |

源码和依赖哈希完整列表保留在
[`PUJIANG_SHARED_REUSE_REVIEW_20261010.md`](PUJIANG_SHARED_REUSE_REVIEW_20261010.md)。
JSON 采用严格解析，拒绝重复键和非有限常量；文件只读取一次，已校验字节直接解析。
所有 `claim_boundary` 均为 false，不能用本工具解除执行门禁。

## 2. FrontierScience 离线输入

工具：[`../scripts/validate_pujiang_frontier_pair.py`](../scripts/validate_pujiang_frontier_pair.py)。
本接口是 offline envelope，不是获批准的正式 solver/judge execution contract。

- envelope：`schema_version=aqkv-frontier-offline-envelope/v1`、
  `scope=supplied-manifest-only`、`evidence_kind=development-fixture` 或
  `unverified-stored-outputs`；dataset/revision/两份源码哈希固定。
- dataset revision：`25ed67db7da8f4591484e764008ff585544f5a30`。
- generation：显式 `max_tokens`、`temperature`、`seed`、`trials=1`、`retries=0`；
  这是本离线格式的单次配对约束，不是对正式协议的审批。
- inventory：两赛道都存在，每题用 `track/test/{source_row:03d}` 作为身份。
  源行限于 Olympiad 0-99、Research 0-59；允许多个题具有同一 group ID。
  包含 source-row 摘要、独立 problem/reference 文本、真实 prompt token IDs、
  token 数/摘要、generation 摘要。摘要字段不证明它是官方源题。
- B0/B1：各包含 `collection` 和完整 `outcomes`，不允许漏题或重复；每条匹配
  task/track/prompt/generation。solver 失败须留错误；成功尝试须留原始文本及
  stop/length。截断不丢题，不依据是否答对调整分母。
- `files_sha256`：固定 inventory、B0、B1、judge_contract、calibration 的实际文件
  摘要。后两项必须同时提供或同时为 null。

judge 合同固定 candidate scope、model/runtime、共享 scorer 摘要、分赛道指令摘要、
生成预算及 calibration 文件摘要。校准记录必须绑定同一个 model/runtime/指令，
保留两个赛道及严格 bool 的 passed 状态；不接受单赛道、重复赛道或伪造 bool。
fixture 输入只能配 fixture 校准；存储输出只能配 `unverified-external-receipt`。
即使外部记录声明 passed，本工具也不验证其语义真实性。

collection 记录绑定相同 judge/calibration 摘要和“作答前冻结”的声明；B0/B1 使用
同一 judge。该声明没有独立时间/运行审计，因此输出明确保留
`pre_generation_binding_verified=false`，不能称为真实预注册证明。

已判分的条目需保留 judge runtime、合同/校准摘要、完整请求和原始响应。请求中的
problem/reference/attempt 必须匹配对应题及原始尝试，不能换题、换 rubric 或事后换答案。
响应 model 必须匹配；从响应重新提取 verdict，忽略缓存 correct/grade 字段。
格式错误、多个矛盾 verdict 或 judge 截断会保留为 judge_error，而非丢弃样本。
候选 Research parser 的 0-10 / >=7 语义来自固定共享源码，不是本模块确认的官方协议。

输出规则：

| judge_binding | 输出与边界 |
| --- | --- |
| missing | 缺少 judge/校准；已有缓存判分也不产生成绩 |
| not_passed | 校准声明未通过；不产生 fixture 或真实成绩 |
| fixture_only | 只在开发 fixture 上调用固定 parser/aggregate；结果仅在 fixture_scores 下，两赛道分别记录 |
| bound_unverified | 外部记录结构绑定完成；不生成新的准确率，不升级为真实校准 |

所有情况下 `task_quality=null`、`semantic_calibration_verified=false`、
`aqkv_path_observations=null`。逐赛道记录任务数、solver 错误、未评分和 length-limited；
不合并成一项跨赛道准确率。fixture 中 judge_error 使该赛道聚合分数为 null；
solver_error 仍由公共 aggregate 保留在分母中。

```bash
python scripts/validate_pujiang_frontier_pair.py \
  --benchmark-checkout /path/to/pinned/vllm-hust-benchmark \
  --envelope /path/to/offline-envelope.json \
  --inventory /path/to/inventory.json \
  --b0 /path/to/B0.json --b1 /path/to/B1.json \
  --judge-contract /path/to/candidate-judge.json \
  --calibration /path/to/calibration.json \
  --output /path/to/new-offline-report.json
```

无 judge 时省略最后两项输入，并将对应摘要与 collection 绑定置 null/false。
这些占位路径不代表我们已有正式题目或校准资产；本轮实际输入只有测试临时目录中的
synthetic 样本。正式输入需源题/提示审计、许可、实际 judge 与语义校准、预注册审阅、
真实 raw request/response、Host/profile/资源与正确性证据。

## 3. Terminal 候选计划输入

工具：[`../scripts/validate_pujiang_terminal_plan.py`](../scripts/validate_pujiang_terminal_plan.py)。
本轮只调用固定共享 `job_config`，不调用 prepare、audit、validate_harbor 或 task runner。

2026-10-12 后续增加**显式实际资产绑定**，不改变未提供资产输入的旧 v1 语义。
附加 `--published-root`、`--snapshot`、`--staged-tasks` 必须三项齐全：重验全部固定
源码、公开报告及给定任务副本，只允许 image pin、保留全部其他字节/政策/Git 权限，
并将 inventory 摘要与实际文件比对。成功为 `candidate_plan_assets_bound_execution_blocked`；
只提升 source-membership/task-assets/local-staged-modes。不修改实际服务/Harbor/沙箱/
权利/执行门禁，不生成正式选题或 model endpoints。使用与摘要口径见
[Terminal 第 5 节](PUJIANG_TERMINAL_SOURCE_AUDIT_20261012.md)。

- envelope：`schema_version=aqkv-terminal-candidate-envelope/v1`、
  `scope=supplied-task-plan-only`；kind 为 development-fixture 或
  unverified-candidate-plan。Terminal 源 revision 固定为
  `7131e4375048a0e408a8fb404b5f499d726b695b`。
- Harbor 候选 version/revision 必须与固定共享代码一致；目前是 0.24.0 /
  `b53b8134e1241686dca7759af188f987ecc48e8b`，不是项目组已批准的正式协议。
- inventory：唯一 safe task IDs、按 digest 的 image 引用，以及 source-row、
  staged-files、file-modes、task-policy 摘要。允许明确命名的子集，不声称已核对
  所有 89 个真实任务；没有读取实际文件/镜像时，摘要只是声明。
- `serving_intent`：显式 model alias、两臂 credential-free `/v1` URL 与 bool
  `B0=false/B1=true` 计划。`runtime_receipts` 两臂必须为 null。
- B0/B1 JSON 必须完整匹配固定 generator 的候选配置，只允许约定的 AQKV job label、
  共同 model alias 与明确登记的 API 地址；目录标签、URL 和 model alias 都不证明激活。
- `files_sha256` 绑定 inventory/B0/B1 的实际字节。

保留候选的完整任务列表、一次 trial、并发 1、turn/token/context 预算、无 summarization、
verifier 开启、退出删除 sandbox、task policy 不覆盖等配置。两臂同时降低预算也会拒绝，
不能只靠 B0/B1 彼此相同绕过固定候选协议。

重试声明独立保留：whole-trial 和 LiteLLM SDK retries 为 0，但 Harbor LLM 和
Terminus query 两层各有 3 次 attempts。**不能称为零模型请求重试**，也不能据此推导
所有故障情况下的精确请求总数。真实环境与依赖还需检查兼容 fallback 等行为。

```bash
python scripts/validate_pujiang_terminal_plan.py \
  --benchmark-checkout /path/to/pinned/vllm-hust-benchmark \
  --envelope /path/to/candidate-envelope.json \
  --inventory /path/to/task-inventory.json \
  --b0 /path/to/job-b0.json --b1 /path/to/job-b1.json \
  --output /path/to/new-plan-validation.json
```

输出为 `candidate_plan_valid_execution_blocked`。schema 本地实例化、真实资产、连接、
实际 B0/B1 开关全部未验证；task_quality=null、model_calls/tasks_executed=0，
ready_for_scored_execution=false。若两臂使用同一 endpoint，也不证明真实顺序切换。

尚需正式合同/权利审阅、真实任务与权限、digest 对应的镜像层、实际 AMD64 后端/
依赖与资源限制、后端到模型的连接/认证、匹配服务配置与 Host/profile/Manager/
正确性证据、fresh-sandbox reference/failure verifier 检查和完整任务执行回执。

## 4. 本轮验证与后续

- 插件完整本地回归：507 passed，比第一轮增加 148；0 skipped。
- 新 FrontierScience 测试 79 项、Terminal 64 项、共享加载器 5 项。
- 指定固定 checkout 后，MMLU 3、FrontierScience 4、Terminal 9 项真实源码离线
  集成测试实际执行；未配置时这 16 项明确 skip，不用 skip 冒充通过。
- 共享 benchmark MMLU / FrontierScience / Terminal-plan 57 项测试仍单独核验。
- 本轮评分样本均为 synthetic fixture，不是官方源题、模型/judge 输出或成绩。

接下来补正式 MMLU 源题/任务/transport 接口与路径回执，并继续 Host #37 runner/gate
排查；各项实际 judge、Host/profile、许可、后端与设备输入必须另行闭合。
三项离线接口完成不等于五项正式评测完成。本轮不提交 GitHub、不发送 Slack。
