# V5.4 交付线与 AQKV 研究线对齐

更新：2026-10-11。本文件记录已公开的新合同及本地控制调整，不是执行许可。

## 1. 新权威与固定来源

benchmark 在精确提交 `b32fcbdaf8d3237936b35f9f4ba441bf2d64b361` 已发布
[V5.4 合同](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/b32fcbdaf8d3237936b35f9f4ba441bf2d64b361/docs/ACCEPTANCE_V5_4.md)。
权威声明的原始字节保存在
[`PUJIANG_V54_CONTRACT_SOURCE_20261010.json`](evidence/PUJIANG_V54_CONTRACT_SOURCE_20261010.json)，
SHA-256 为 `0e4e14a2ad0b6a2fe351b817c03519ecdb638464152bb4e429933be9fcee4f11`。
既有 V4.x 文档及早期 C8 研究计划不再生成当前正式 target/config ID/测量队列。

仓库 DOCX 是 semantic mirror，571,064 字节、SHA-256
`2423d611fce0cd74be47efcac8e3fb79a472e8304bbfe663dd2127974e1ddd46`。
本地已核验该镜像字节、schema 和上游四项合同测试，不声称它与未提供的最终本地文档
字节相同，也未验证最终排版或 87 页缓存属性。

## 2. 必须调整的施工规则

| 项目 | V5.4 固定规则 | 对本模块的约束 |
| --- | --- | --- |
| 必选模型/设备 | Qwen3.5-35B-A3B，2 张 910B2，TP=2，BF16/bfloat16 | 不能把历史 C8/其他模型实验直接升为正式交付配置 |
| 两臂 | 仅 B0/B1；B1 为测试开始前冻结的最终交付引擎 | B1 不自动等同“本插件 ON”；须有签名 release manifest、OCI RepoDigest 与精确 core/plugin/model/tokenizer 身份 |
| A3 | 30720 输入 + 2048 输出，32768 上下文，30 分钟；TTFT mean/p95/p99 为 30/45/60 秒，TPOT 为 40/60/80 ms/token | 固定 SLO；prefill 吞吐只作诊断，不能生成阈值或改变 verdict |
| A4 | 完整启动至停止生命周期成本 / 30 分钟窗口内满足冻结 SLO 的成功输出 tokens | 不能只计算 kernel/稳态局部成本，失败输出不能进入成功分母 |
| 可选项 | E1/E2-G/E2-D 不启用，NOT_EXECUTED_OPTIONAL | 不注册 target/config ID，不排测量；激活须新计划版本及完整冻结合同 |
| 配置 | 执行前明确并冻结；禁止现场调参、自动 fallback、best-of 参数搜索和事后改 verdict | 既有模块内兼容 fallback 与正式运行的允许策略必须事先明确，不能靠运行中换配置过门禁 |
| 五项质量 | MMLU-Pro、HLE-Verified、SWE-bench-Pro、FrontierScience、TerminalBench2.1 | 保留原题、失败/截断、评分和真实后端要求，不用其他 MOD 的成绩替代 |

BF16/dtype 声明并未单独给出本模块 INT8 KV 的准入结论。**不能擅自解释为已允许 C8，
也不能解释为彻底禁止 KV 量化或取消课题**。需要负责人明确 BF16 精度约束的 KV 范围，
以及本模块是否属于最终 B1 引擎的签名 manifest；在确认前不启动 C8 正式测量。
E1/E2 禁止排队是原文规则，但不擅自把 AQKV 自动归类为其中某项。

70 题 MMLU 的 2048 候选生成预算只是本地开发合同，不是 A3 已执行，亦不是正式
选题获审。共享任务质量结果与 A3/A4 的服务 SLO/成本分开核验，不能互相替代。

## 3. 保留旧证据，不混淆版本

- 原 `e3a7183` 资产 JSON 与 `8ed7ec9` 复用实现不覆盖；旧 canonical_blockers 是历史
  总账的原文，不是当前执行 authority。所有正式决策采用上述 V5.4 声明。
- 新 program JSON SHA-256 为
  `4d9d73fa55c6e15db036fd5f65588f2be476642816db4a9651cfad9e5b2b53ca`。
  对照只发现更新时间、V5.4 test_plan 和三个 MMLU 外部交接路径变更；五项源数据
  身份、数量和原阻塞文本未变。路径改名不证明已取得那些外部材料。
- MMLU scorer、FrontierScience scorer、Terminal plan 三份复用文件在新提交与旧
  `8ed7ec9` 的字节相同。继续使用原固定 checkout，避免污染既有来源/token/scorer 回执。
- `validate_pujiang_readiness.py` 同时核验旧资产快照和新 V5.4 声明原字节；拒绝把
  control 改为 C8、旧版本、第三臂、可选队列或已接纳。输出仍为 planning-only。

## 4. 当前施工顺序

1. P0：控制与 TODO 已对齐 V5.4；用户授权后已在插件 Issue #10 请求确认未决精度
   范围和 B1 身份 owner。10 月 11 日刷新仍无新回复，不以留言成功视作批准。
2. P1：完成已存 MMLU 原始回执到 scorer 的离线连接和拒绝错误输入的测试，收束包与
   文档；无模型请求，不以 fixture 声称成绩或 transport 可用。
3. P1：审阅 Host 最小测试补丁的公共提交路线，获得合法双版本 CI/设备回执；保持
   PR #35/#37 的 Draft/未接纳状态，不复制旧产品分支。
4. P1：取得签名 B0/B1 身份、实际服务 token parity、正式选题/许可/scorer、执行环境
   与资源/清理回执。AQKV 量化臂另需已接纳 provider、校准 profile 和真实正确性。
5. P2：条件满足后执行被冻结的适用任务，再将原始证据提交 benchmark、回链 Issue
   和网站。未审阅的候选、其他 MOD 成绩、CPU CI 均不能升级成该步骤完成。

10 月 10 日已授权发布
[Issue #10 准入请求](https://github.com/vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust/issues/10#issuecomment-6097948895) 和
[Host #37 提交/CI 路线请求](https://github.com/vLLM-HUST/vllm-ascend-hust/pull/37#issuecomment-6097950276)，
不再沿用发送前的“未发布 Issue”状态。公共代码尚未提交；没有发送 Slack、触发
模型/NPU/Harbor 任务或创建自动化。正式执行继续 blocked。

10 月 11 日再次从 benchmark 精确提交 `402a5e41a1b2b0363858c0b77586e1aa92723396`
取得 V5.4 JSON 与三份复用源码，原字节均未变；新增 StateAxis 审计不属于本模块准入。
本地新增有界 HTTP fixture 验证，但不代替真实服务身份/远端 transport，见
[本地 HTTP 边界](PUJIANG_LOOPBACK_TRANSPORT_20261011.md)。

同日继续补 [已存完整 token IDs 对照](PUJIANG_MMLU_TOKEN_PARITY_20261011.md)，
不一致时阻断离线评分；780 项本地测试通过。tokenizer/config/release 摘要声明不
认证服务来源或签名，合成回执不升级正式运行准入，P0/P1 外部输入与执行门禁不变。
