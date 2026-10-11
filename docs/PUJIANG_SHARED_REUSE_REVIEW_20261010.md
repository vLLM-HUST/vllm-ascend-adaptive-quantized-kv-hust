# 五数据集共享工具：复用审查与离线接入

日期：2026-10-10。本文件为本地开发记录，未提交、未获合同审阅；无模型请求、
沙箱任务或 NPU 运行。共享 benchmark 固定为
`8ed7ec9a16075c5bce5644a68a712139a6e5d403`，本地 sparse reference 约 8.7 MB。
这一初始复用审查没有复制模型权重、评测源题数据或约 40 GB 镜像层。
同日后续 MMLU 固定源文件/tokenizer 的小文件核验与来源绑定增量见
[`PUJIANG_MMLU_SOURCE_PLAN_20261010.md`](PUJIANG_MMLU_SOURCE_PLAN_20261010.md)。

## 1. 审查结论

| 入口 | 可复用部分 | 不可直接继承的部分 |
| --- | --- | --- |
| `mmlu_pro_pyramidkv.py` | `score_results` / `extract_answer` 的固定提取、范围校验、失败计分、截断记录及逐题输出 | `choose_tasks` 添加 >4096 的压缩资格题；`eligible/ineligible` 与该门槛绑定；prepare/run/analyze 的协议、transport、激活和 transaction 均属 PyramidKV |
| `frontierscience_inputs.py` / `frontierscience_pyramidkv.py` | 来源摘要、赛道/源行号身份；候选 verdict parser / aggregate 的分赛道接口；未完成评分保持 null | grader 明示非官方复现；Research >=7 的候选成功阈值和具体 judge prompt 需协议审阅；parser 单测不是 judge 语义校准；不得继承压缩 not-exercised 判定 |
| `prepare_pyramidkv_terminal.py` | 镜像 digest pin 后的 TOML 语义核验、原 Git 文件权限恢复、任务政策保留及 schema 检查 | agent/model 名、job 标签不切换服务；Harbor 0.24.0 / Terminus-2 预算是候选合同；不能把 schema 或本地 transport 回执当成 sandbox/verifier/任务证据 |
| HLE / SWE 来源审计 | 模态、源行/patch/harness/repo/image 摘要和权利缺口的记录方式 | 本轮只审查入口、provenance/权利边界及部分相关实现，未在本地重跑真实数据审计；root license 不清除第三方任务权利；不能泄漏 rationale 到 solver |
| Terminal fetch / metadata audit | digest 校验、source/image metadata 与镜像层分离的流程 | 未重新下载或校验所有 89 个真实任务资产，也未拉镜像层；actual backend/连接/认证与 verifier 仍须验证 |

共享 program JSON SHA-256 仍为
`de1347eb1557f7e82ea3314c118ee090df1d4a87dd18e474c801b4be3a8185a4`，与旧本地
canonical snapshot 逐字节相同。故保留 `sources.benchmark_revision=e3a7183...` 的
资产快照身份，另记录 `shared_implementation_revision=8ed7ec9...`，不伪造更新后的总账。

## 2. 复用版本与源码摘要

以下路径相对于上述公共 benchmark 提交。代码读取与摘要核对完成，不代表协议已审阅。

| 文件 | SHA-256 |
| --- | --- |
| `src/vllm_hust_benchmark/mmlu_pro_pyramidkv.py` | `c2237c69f848e2358abe71c68f54143b5c090a81bd4f7915088b19b8ae157801` |
| `src/vllm_hust_benchmark/frontierscience_inputs.py` | `3dcc94f282f055776f129688469320978c1dae7474284ce41e5e7e100519c8ff` |
| `src/vllm_hust_benchmark/frontierscience_pyramidkv.py` | `1e3b6a4e4eade0144f2372c6ed16292ae143bb4243bb672e1a374baa26560560` |
| `scripts/prepare_pyramidkv_terminal.py` | `1a36a171be3db57964284bc2ff9fa26cd218ef9f766ed3bdaafa548e55780a95` |
| `scripts/audit_pyramidkv_hle.py` | `2190113809026f35e847de9ec460b77e294a51d5f1a42a8c79d78bc3822d99e2` |
| `scripts/audit_pyramidkv_swe.py` | `64296056ab78423ee7e5a0d9fb243dc905ba4f904b923a65661771eedc18ebf2` |
| `scripts/audit_pyramidkv_terminal.py` | `9dcfbec30b1a75df2fca3e46b23b3aee12779912c1db8b15c6d0ae9e18a5951c` |
| `scripts/fetch_pyramidkv_terminal.py` | `1a92e265cbd696bdd50f6fc98d2381d89a20137baf4be3d54f8a2aa989d02c52` |

来源：[不可变源码](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/8ed7ec9a16075c5bce5644a68a712139a6e5d403)
与本次 5 封邮件对应的 [Issue #254](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254)。

## 3. 本次 MMLU 离线接口

[`../scripts/validate_pujiang_mmlu_pair.py`](../scripts/validate_pujiang_mmlu_pair.py)
仅离线读取文件，直接调用经过精确 HEAD 与源码摘要验证的公共 `score_results`，
没有复制/重写答案提取算法，也没有调用上游 prepare/run/analyze。

输入是离线证据 envelope，不是作答前已审批的正式 execution contract：

- `schema_version=aqkv-mmlu-offline-envelope/v1`；`scope=supplied-manifest-only`。
- `evidence_kind` 仅允许 `development-fixture` 或 `unverified-stored-outputs`。
- dataset 固定为 MMLU-Pro 与源 revision；benchmark/scorer 固定到第 2 节的身份。
- `task_count`、`generation.max_tokens`、`generation.ignore_eos` 与所有任务匹配。
- `files_sha256` 固定 cases/B0/B1 的实际字节；每个文件只读取一次，已校验字节直接解析。
- cases 逐项包含唯一 `case_id`、category、answer、option_count、真实 token IDs 及摘要。
- B0/B1 必须各有一个与每题匹配的原始输出或失败记录；success 必须是真正 bool。

接口保留所有任务的质量计算、失败与 length-limited 标志，删除上游专属
eligible/ineligible 字段。输入题目、答案与模型来源尚未验证，不能把报表解释为官方
或真实 AQKV 成绩。输出固定 `offline_scored_runtime_unverified`，所有 claim gates
为 false，`aqkv_path_observations=null`；拒绝覆盖已有输出文件。

```bash
python scripts/validate_pujiang_mmlu_pair.py \
  --benchmark-checkout /path/to/pinned/vllm-hust-benchmark \
  --envelope /path/to/offline-envelope.json \
  --cases /path/to/cases.json \
  --b0 /path/to/B0.json \
  --b1 /path/to/B1.json \
  --output /path/to/new-offline-report.json
```

上面的命令格式可用，但当前没有本模块真实数据集输入；本轮只有测试临时目录中的
synthetic-unit-test 样本。没有生成或保存虚假的正式数据集报表。

## 4. 第一轮 MMLU 验证回执

- 固定 benchmark 的 MMLU / FrontierScience / Terminal-plan 相关测试：57 passed。
- 插件全量测试：359 passed，较 10 月 9 日增加 66 项；其中包含固定公共 scorer 的
  实际离线调用、错误计分、截断分母、超 4096 不自动认定 AQKV eligible、篡改/重复拒绝。
- 公共 scorer 集成测试通过 `AQKV_TEST_BENCHMARK_CHECKOUT` 提供明确的固定 checkout；
  未配置时这 3 项会明确 skip，其他边界测试不依赖下载或环境中的 benchmark 安装。
- wheel/sdist 构建及发行包校验通过；新增脚本、测试与控制文档已进入源码包。
  wheel 继续只暴露静态 extension entry point，manifest 为 `import_only`，没有增加
  模型执行或激活入口。打包通过不是正式发布、Host 接纳或评测验收。
- 上述均为本地 CPU/离线开发证据，不能解释为 GitHub CI、Host 接纳、NPU correctness
  或真实任务成绩。

```bash
AQKV_TEST_BENCHMARK_CHECKOUT=/path/to/pinned/vllm-hust-benchmark \
PYTHONPATH=src python -m pytest -q
python scripts/validate_pujiang_readiness.py
```

以上保留第一轮 359 项测试回执，不覆盖旧记录。第二轮已完成 FrontierScience
离线分赛道/校准绑定和 Terminal 候选计划校验；完整回归为 507 passed。
具体输入、反例、证据边界及剩余条件见
[`PUJIANG_FRONTIER_TERMINAL_OFFLINE_20261010.md`](PUJIANG_FRONTIER_TERMINAL_OFFLINE_20261010.md)。
这两项均不启动 judge/模型/Harbor。MMLU 正式数据/选择合同、实际请求 transport、
Manager/Host 激活与 runtime correctness 仍待完成；正式 FrontierScience
校准和 Terminal 后端/verifier 也仍缺。保持反馈稿未发送，后续按授权提交公共 PR/Issue。

本节以上记录最初 v1 adapter 的边界。随后新增的来源绑定 v2 模式已经在本地重新
核验官方题目、答案、固定 tokenizer 和完整 scorer bundle；该功能不改变旧 v1 的
声明范围，也不证明实际服务端分词或请求发送。70 题全失败 development fixture
仅验证连接与计分分母，不是模型准确率；五项正式合同/执行验收仍未完成。
