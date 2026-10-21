# Host PR #37 门禁及最新主线对照

## 1. 当前证据

10 月 10 日重新读取 GitHub：PR #37 仍 Open/Draft，head 为
`3fb17f2a9eb140cd785d390645a9c743344f9d29`，PR base 为
`17f68177472b896facf3f0501d424d24e16e699a`。老师 10 月 9 日要求未变：产品改动
必须满足设备/source-change gate，不能凭 CPU UT/pre-commit 合并。

原 run `36372359941` 的 select-tests job `108775478306`：cancelled、
runner ID 0、runner name 空、steps 空，标签为 `linux-amd64-cpu-8-hk`，
约 24 小时后取消。没有其任何测试选择步骤实际执行的证据。
这支持先调查 runner/调度，但不足以确诊根因或断言产品断言失败。

- [老师要求](https://github.com/vLLM-HUST/vllm-ascend-hust/pull/37#issuecomment-6073434214)
- [原始 job](https://github.com/vLLM-HUST/vllm-ascend-hust/actions/runs/36372359941/job/108775478306)

## 2. 无设备的源码检查

在原 head 的干净 worktree，只运行原版 `coverage.py` 与
`select_tests.py --all-tests --skip-cpu-ut`，未更改工作流或触发 CI：

- coverage 配置检查 exit 0；有过渡期未标记测试 warning，不等于无 warning。
- 选择器 exit 0，has_tests=true，15 个设备分组、132 个目标出现次数、CPU 分组 0。
- 标签分布：310p-1 一组、a2b3-1 五组、a3-800i-2 四组、310p-4 一组、
  a3-800i-4 四组。这是配置输出，不是资源预约、设备执行或精确测试通过数。
- 现有 GLM pooled-cache UT 明确使用 CPU 和构造的 runner。nightly 中有 GLM
  state/kpool kernel 测试，但不能自动作为本 PR initialization/allocation-route
  的完整回归或数值正确性证据；仍需明确针对该路径的设备用例和 oracle。

这些检查缩小了“选择器本身不能运行”的猜测，不能替代原 CI 的 checkout/rebase/
container/runner，也没有解除设备门禁。

## 3. 最新主线已经改变，先核对再重放

本次 live main 为 `f939ad767b60d60253c3e20c7522defe69e4ca78`。
GitHub compare 显示它相对 PR head 已 diverged：ahead 237、behind 6；这是提交图，
不是这 237 个提交全部已审阅。compare 文件列表可能截断，故另行按精确 main SHA
读取相关源码，没有根据“列表没出现某文件”认定文件未变。

确认以下实际变化：

- 最新 `initialize_kv_cache_tensors` 在 allocation context 内统一调用
  `self._allocate_kv_cache_tensors`。旧 PR 的 model_type 专用分支已不是当前结构，
  不能原样追加，更不能据此宣称本 PR 已合并。
- 最新 reshape 调用 `view_glm5_next_cache`；GLM pooled-cache test 的 tail
  从旧“连续尾部”期望改为按 block/page stride 定位。这是布局合同变化，不能
  把旧测试机械移植成兼容性修复。
- 最新工作流 CPU UT 对 verified vLLM main 与 matched release tag 分别运行，
  镜像标签也已更新。旧 run 的单版本 CPU 通过不能代表新工作流两版本通过。
- `select-tests` 的专用 CPU label/后续设备门禁仍存在，未找到允许绕过门禁的依据。

源码回链：
[当前初始化/分配代码](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/f939ad767b60d60253c3e20c7522defe69e4ca78/vllm_ascend/worker/model_runner_v1.py#L4745)，
[当前工作流](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/f939ad767b60d60253c3e20c7522defe69e4ca78/.github/workflows/pr_test.yaml)，
[当前 GLM UT](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/f939ad767b60d60253c3e20c7522defe69e4ca78/tests/ut/worker/test_glm5next_pooled_cache.py)。

## 4. 全部 11 个文件的改动意图对照

再次 live 检查，main 与 PR head 均未变化。11 个 main 文件分别通过 contents API
按精确 SHA 获取，并核验 Git blob SHA；不是以 compare 截断列表推定文件未变。

| 原 PR 文件 | 当前 main 的对应事实 | 本轮决定 |
| --- | --- | --- |
| `tests/ut/conftest.py` | 仍用非 package 的 MagicMock 替换 `triton.runtime`；现代 Gluon 导入仍需要 runtime.jit | 保留仅 mock driver 的测试环境修正；完整 Triton/Host collection 待真实 CI |
| `tests/ut/models/test_engram.py` | 原 CLI parsing 用例不在当前文件，文件已改为 embedding/checkpoint 用例 | 不移植旧 `get_model_path` monkeypatch；不声称其他位置的 CLI 覆盖已审阅 |
| `tests/ut/ops/test_fused_moe.py` | apply 的测试替身没有 moe_config，当前 production apply 无条件读取 num_experts | 补测试替身 `moe_config.num_experts=4`，不改 production |
| `tests/ut/ops/test_vocab_parallel_embedding.py` | 生产 normal route 传第四参数 skip_gather=false，原断言仍只有三个参数 | 更新断言，保留 normal/lmheadtp 分支检查 |
| `tests/ut/patch/worker/test_mamba_utils_contract.py` | fixture 的两个 block_size 不同；当前两条 vLLM CI 依赖线均拒绝这种 UniformType 分组 | 同 block_size、等 page bytes、不同 state shape；不改 Mamba 分组机制 |
| `tests/ut/worker/test_attn_utils_v2.py` | indexer/cache/model config 字段已存在 | 原改动已覆盖，不重复追加 |
| `tests/ut/worker/test_glm5next_pooled_cache.py` | cache_config 已不被当前 allocation helper 使用；最新 initializer 仍传 kv_cache_groups，UT 断言漏传 | 不恢复旧 runner.cache_config；只补绑定断言，保留最新 page-stride 期望 |
| `tests/ut/worker/test_kv_cache_allocation_scope.py` | initializer 读取 kv_cache_groups，scope fixture 仍是空 SimpleNamespace | 只补实际需要的 kv_cache_groups=[]，不复活旧 kv_cache_tensors 分支 |
| `tests/ut/worker/test_model_runner_v1.py` | V41 annotation 与 backend fixture 已有；graph 测试按新 enable_sp 路线重写 | 保留主线版本，不用旧 graph mocks 覆盖新行为 |
| `tests/ut/worker/test_model_runner_v2_mamba.py` | 两个 FakeAttention 仍缺 get_attn_backend；第二个测试已改为验证不能整除时拒绝 | 为现有两个测试补 identity customize_spec，不恢复旧“接受不能整除”期望 |
| `vllm_ascend/worker/model_runner_v1.py` | 初始化已统一使用 Host 自有 allocator | 不重放 GLM model_type 专用产品分支；不宣称原 PR 产品改动已被合并 |

结论：**7 个测试文件仍有可保留差异，4 个文件的旧改动已覆盖或不再适用**。
本轮候选补丁只有 18 行新增、8 行删除，没有任何 production/workflow 改动。
这只是缩小修复范围，不意味着这 7 项修正足以保证整个新主线 CI 通过。

当前 CI 固定 vLLM main 为 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，
matched release 为 `v0.30.0`；两份实际 `kv_cache_interface.py` 字节相同，
SHA-256 为 `1cf202f1a44d5bc5c3832b70b41507687ecceea3784e67a4f003bc0210d6eecb`。
其中 `is_uniform_type` 明确先拒绝不同 block_size，再检查 Mamba 的相同 page bytes 等属性。

## 5. 实际复现与补丁边界

本地执行固定主线的 initializer、logits、MoE apply 函数片段及相关测试函数，显式
stub 导入、allocation/reshape/scale fill 和 fused kernel 等依赖。复现了四处差异：

- GLM bind：原断言缺 kv_cache_groups，产生 AssertionError；候选断言通过。
- allocation scope：原配置没有 kv_cache_groups，产生 AttributeError；候选配置通过。
- logits：生产调用四参数，原三参数断言产生 AssertionError；候选断言通过。
- MoE：原替身没有 moe_config，产生 AttributeError；候选替身通过。

这是 **pinned-source-fragments-with-stubbed-dependencies**，不是完整 Host UT、真实
GLM cache allocation/layout、Triton 安装验证或设备数值正确性。首轮其余三类差异仅为
源码对照；随后增量复现见第 7 节。7 个文件通过原 Host 配置的 Ruff format 检查，补丁通过
`git apply --check -p2`，原 Host worktree 与旧 PR 分支保持不变。

候选补丁、11 文件原始摘要、四处复现脚本/回执在本地受限目录：
`/Users/anon/Documents/面向KV异构分布的量化维度设计/output/pr37_reconciliation_20261010.ggO1rB`。
`candidate-tests.patch` SHA-256 为
`e9e6b188980d76e05789e10f972460c2271891a2230b66fd89077b1b177d9701`；
在相同精确主线 checkout 中可先运行 `git apply --check -p2 /absolute/path/candidate-tests.patch`。
执行复现脚本需要本地 torch，不需要 vLLM 安装或设备；它刻意不代表完整依赖链。

## 6. 后续执行顺序

1. 将当前最小测试补丁交由维护者确认是更新 PR #37 还是拆为独立测试维护 PR；
   新提交必须注明基线与删去旧产品分支的理由，不能沿用旧“仅一行 annotation”的描述。
2. 在真实 CI 环境验证 Triton collection、两条 vLLM CPU UT 依赖线及 pre-commit。
   由正常工作流决定所需 gate，不改标签/选择器来绕过要求。
3. Host PR #35/provider 等产品改动仍需真实适用设备验证、总 gate 及正式接纳；
   测试维护补丁通过不能解除这些门禁。
4. 有完整结果并审阅/接纳后才进入插件运行验证。当前仍 Draft/blocked。

上述诊断阶段未推送、评论、改标签、重跑 CI、预约资源或连接服务器。
随后用户授权的沟通记录见第 8 节。

## 7. 后续增量：剩余三类差异的实际片段复现

再次只读核对 GitHub，Host main、PR #35/#37 head 和负责人回复均未变化。
最小补丁字节也未改变。新结果保存在独立目录，首轮四项回执没有覆盖。

本轮加载原字节 vLLM 数据类 AST，保留 dataclass/property/classmethod 装饰器；
两条依赖线的来源仍为 `ced6857` 和 `v0.30.0`，KV 数据类文件字节相同。
Host Mamba helper 另按精确 `f939ad7` 获取并验证 Git blob 与 SHA-256。
仅去除测试函数的 patch/parametrize 装饰器，显式传入对应 mock/参数。
依赖 stub 和源码摘要完整列在新回执中，没有替换被测的分组/对齐函数体。

| 对照用例 | 原主线 fixture | 候选 fixture |
| --- | --- | --- |
| Uniform Mamba 分组 | 不同 block_size 导致 from_specs 返回 None，断言失败 | 同 block_size、等页字节、不同 shape，实际 helper 分组断言通过 |
| Attention/Mamba 页对齐 | FakeAttention 缺 get_attn_backend，AttributeError | identity customize_spec 后，20-byte padded page 与 Attention 先于 Mamba 的顺序断言通过 |
| 48/80-byte 页、不含 Mamba | 同一缺接口错误 | 保留原两个 Attention spec，不强制混合对齐，通过 |
| 48/80-byte 页、含 Mamba | 同一缺接口错误，尚未到达预期拒绝点 | 实际 alignment helper 抛出指定 ValueError，测试通过；没有恢复旧接受逻辑 |
| Triton package 保留 | MagicMock 替换 runtime 后，runtime.jit 导入报 not a package | 仅 mock driver，保留原 package，子模块导入与 mock 属性断言通过 |

**五组对照全部符合预期**。与首轮四组共覆盖七个测试文件的九个片段/参数化对照。
这不是九个完整 Host UT；Triton 使用临时合成 package，仅证明 Python 导入机制，
未安装或测试真实 Triton/Gluon。既未执行完整 Host collection，也未跑两套实际 CI 环境。
运行有 Torch 的 NumPy 缺失 warning，本轮没有使用 NumPy，也没有将其描述成无 warning。

新私有目录：
`/Users/anon/Documents/面向KV异构分布的量化维度设计/output/pr37_contract_reproduction_20261010.KHpo6I`。
`remaining-contract-reproduction.json` SHA-256：
`43b6162060e5cf4389d66201599ee54945e738fdcc72a3aa4b9112300e46dc9b`。
`reproduce_remaining_contracts.py` SHA-256：
`ff603a54249fc40930cd2ee6f896a3ed99ef29fe15f5b086c99d1ee0f88b2670`。
取得的 `patch_mamba_utils.py` SHA-256：
`5dd0a15e141bb92a83715ac5ae98cf5b072fc8989d6b5225decbf088c4f6617b`。

当前可以交维护者评审的是七文件最小测试补丁及复现依据；正式提交路线、真实双版本
CPU/Triton CI、正常总 gate 和产品路径适用设备回执仍待完成。没有修改原 Host worktree
或 PR head；复现结束时没有推送、发消息、触发 CI、启动模型/设备任务。

## 8. 用户授权后的维护者沟通

已于 2026-10-10 13:23:19 UTC 以 `XilingGao` 发布
[PR #37 进展和路线请求](https://github.com/vLLM-HUST/vllm-ascend-hust/pull/37#issuecomment-6097950276)。
实际正文先更正旧留言的范围描述：现有 PR 包含 GLM allocation-route 产品改动，
不能称为“仅一行测试注解、无 production 改动”，也不再据 CPU 证据请求合并。

留言说明精确新基线、七文件测试候选、九组片段对照及真实 CI 的未闭合部分，
请求维护者选择更新当前 PR 或独立测试维护 PR，并确认正常双版本 CPU runner 路线。
发布后按 ID 重新读取校验正文/作者/链接；不据“留言成功”认定负责人已批准。

当前只有反馈已发布；代码未提交推送，PR head 未改，真实双版本/Triton CI、
总 gate、适用设备回归与 Host 接纳仍待完成。Slack 未发送，未启动模型/设备任务。

## 9. 10 月 12 日新 main 漂移核验

Host main 已推进到 `82f586f8dcb5f92f8f3d55727bc05b3e2887606d`（PR #46）。
本轮从 GitHub 逐文件读取该提交与原 `f939ad7` 的七个候选测试文件，验证长度、
Git blob 和 SHA-256：七文件原字节全部相同。原补丁 SHA-256 保持
`e9e6b188980d76e05789e10f972460c2271891a2230b66fd89077b1b177d9701`，
在独立的新源码副本运行 `git apply --check -p2` 通过；没有实际应用补丁。

这只解除“候选是否因新 main 失效”的源码上下文疑问，不证明完整 Host 测试可收集、
正常双版本/Triton CI 或产品设备正确性。新 main 的其他文件及依赖确有变动，不能把
七文件相同解释为整个环境相同。提交路线请求仍未回复，原 Host worktree、PR #35/#37
head 和资源状态没有变更。本轮回执独立保留，不覆盖 10 月 10 日九组 stub 记录。
