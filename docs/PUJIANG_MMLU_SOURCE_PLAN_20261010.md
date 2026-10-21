# MMLU-Pro 源题核验与候选请求计划

## 1. 已完成的具体工作

新增 [`prepare_pujiang_mmlu_source.py`](../scripts/prepare_pujiang_mmlu_source.py)，
在 macOS 上实际读取官方固定版本的两份 Parquet：test 12,032 行、validation
70 行，共 14 个学科。两份文件共 4,187,042 字节，SHA-256 均与固定共享实现一致。
官方 `evaluate_from_local.py` 和 `initial_prompt.txt` 也完成逐字节摘要核验；
没有执行官方 runner、加载模型或调用 NPU。

数据版本为 `b189ec765aa7ed75c8acfea42df31fdae71f97be`，参考 evaluator 版本为
`f418b116db00b065c2aea046518d8fcf74d39872`。格式化直接复用 benchmark `8ed7ec9`
的校验后源码，不重新实现答案过滤/提示模板。源码摘要仍为
`c2237c69f848e2358abe71c68f54143b5c090a81bd4f7915088b19b8ae157801`。

源码及摘要入口：

- [官方固定数据版本](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/tree/b189ec765aa7ed75c8acfea42df31fdae71f97be/data)
- [官方固定提示](https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/f418b116db00b065c2aea046518d8fcf74d39872/cot_prompt_lib/initial_prompt.txt)
- [固定共享实现](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/8ed7ec9a16075c5bce5644a68a712139a6e5d403/src/vllm_hust_benchmark/mmlu_pro_pyramidkv.py)

## 2. 生成内容与范围

- 显式清单保留不可变 question ID 和顺序，只允许 test 成员；0–69 属于 validation，
  本次发现并拒绝了把这组 ID 当测试题的假设错误。
- 验证所有源行的身份、类别、答案/选项一致性；每个学科使用原顺序全部五个
  validation CoT 样例，目标题的答案/解释不进入模型消息。
- 本轮候选子集：各学科按 test ID 升序取前五题，共 70 题；事先生成，不按模型输出、
  提示长度、缓存命中或 AQKV 资格筛选。它只是待审阅子集，不是正式全量评测清单。
- 固定候选请求的 messages、模型别名、temperature=0、seed=17、max_tokens=2048、
  ignore_eos=false、n=1、stop=null、enable_thinking=false、非流式；逐题保存源行、
  提示正文及请求摘要。预算是明确的候选值，不是已获批准的正式合同。
- 执行意图为 POST `/v1/chat/completions`、并发 1、重试 0、上下文上限 32768。
  当前工具没有网络发送功能；这些请求不能直接作为已验证 transport 使用。

原始题目、答案和 70 题计划仅保存在本地受限权限目录，不进入公共代码仓库。
公开的精简回执为
[`PUJIANG_MMLU_SOURCE_PREPARATION_20261010.json`](evidence/PUJIANG_MMLU_SOURCE_PREPARATION_20261010.json)。
私有原始目录：
`/Users/anon/Documents/面向KV异构分布的量化维度设计/output/pujiang_source_gate_20261010.D4RWW1`。

## 3. 复现入口

`selection.json` 使用 `aqkv-mmlu-source-selection/v1`，包含固定 dataset revision、
`split=test`、`review_status=candidate-unreviewed`、非空 selection_reason/model_alias
和有序唯一整数 question_ids。工具不接受“accepted”作为来源准备状态。

```bash
python scripts/prepare_pujiang_mmlu_source.py \
  --assets /absolute/path/to/pinned-parquet \
  --reference /absolute/path/to/pinned-reference \
  --selection /absolute/path/to/selection.json \
  --benchmark-checkout /absolute/path/to/benchmark-at-8ed7ec9 \
  --output /absolute/path/to/new-source-plan.json
```

读取 Parquet 需要 pyarrow；本次在独立验证环境使用 pyarrow 22.0.0，未改变插件运行依赖。
输出使用排他创建，不覆盖旧回执，文件权限 0600。pytest 通过 opt-in 的
`AQKV_TEST_MMLU_ASSETS` / `AQKV_TEST_MMLU_REFERENCE` 实际核验固定资产，不混称 fixture。

## 4. 来源准备之后的剩余条件

输出明确为 `source_verified_candidate_plan_execution_blocked`。这次关闭的是
**本地源题身份与提示构建缺口**，不是正式评测或运行验收：

- 仍需审阅源数据权利、选题、提示、生成预算和最终评分合同。
- 精确模型/tokenizer/chat template 未核验，token IDs/count 为 null；尚不能证明
  未越过上下文预算，也未绑定原离线评分接口中的 token-ID cases。
- 没有 HTTP 原始回执、实际服务身份、Manager 激活、C8 profile 或真实正确性证据。
- 没有 B0/B1 作答、任务准确率、缓存命中/目标路径观测或性能结果。

以上是 source-only 产物边界。下面的增量已经完成本地 tokenizer 阶段，不能倒写成
source-only 文件本身已有 token IDs 或正式执行许可。

## 5. 已完成的本地精确 token 审计

新增 [`tokenize_pujiang_mmlu_source.py`](../scripts/tokenize_pujiang_mmlu_source.py)，
从同一固定源题和 selection 重新构建请求，然后读取
`Qwen/Qwen3.5-35B-A3B@59d61f3ce65a6d9863b86d2e96597125219dc754`
的三份 tokenizer 文件。只下载 12,832,448 字节的小文件，没有下载模型权重。
tokenizer.json 的 LFS SHA-256 及另外两份文件的 Git blob SHA 都与固定版本 API 一致。

加载前逐文件核验固定 SHA-256，再将已读的三份字节放入隔离临时目录，
`local_files_only=true`、`trust_remote_code=false`；不让额外 config 或下载中的文件
参与加载。实际使用 Qwen2TokenizerFast、Transformers 4.57.1、tokenizers 0.22.2、
Jinja2 3.1.6。这是离线工具环境，**不是服务端已确认的依赖版本**。

逐题复用共享 `encode_prompt`，以 add_generation_prompt=true、enable_thinking=false
渲染，再与 apply_chat_template(tokenize=true) 的 token IDs 完全比对。
模板后分开编码使用 add_special_tokens=false，遵循
[Hugging Face 官方模板说明](https://huggingface.co/docs/transformers/main/chat_templating)。
提示正文、请求、源题、选题和预算均没有为通过审计而改变。

实际 70 题结果：

- prompt length：最短 913，最长 2147 tokens。
- 候选生成预算 2048；最大总预算 4195，低于候选 context limit 32768。
- 70 题全部保留，无超预算、截断或删题；逐题保存真实 token IDs、token 摘要、
  渲染提示摘要及 context headroom。局部测试还覆盖精确边界、越界保留/阻断及摘要篡改。
- 不复用 PyramidKV 的 4096 筛选资格；提示长短也不证明真实 cache hit 或目标路径作用。
- `model_requests_sent=0`，`task_quality=null`，全部运行/质量/性能主张仍为 false。

Token plan 状态为 `local_tokens_verified_candidate_plan_execution_blocked`，
其 source_tasks_sha256 与前一轮源题计划一致。
公开摘要：[`PUJIANG_MMLU_TOKEN_AUDIT_20261010.json`](evidence/PUJIANG_MMLU_TOKEN_AUDIT_20261010.json)。
原题/token plan 留在私有 `pr37_reconciliation_20261010.ggO1rB` 目录。

```bash
python scripts/tokenize_pujiang_mmlu_source.py \
  --assets /absolute/path/to/pinned-parquet \
  --reference /absolute/path/to/pinned-reference \
  --selection /absolute/path/to/selection.json \
  --benchmark-checkout /absolute/path/to/benchmark-at-8ed7ec9 \
  --tokenizer-directory /absolute/path/to/pinned-qwen35-tokenizer \
  --output /absolute/path/to/new-private-token-plan.json
```

输出排他创建且权限 0600；超预算时仍保存所有题目并非零退出。对真实 tokenizer 的
pytest 集成通过 `AQKV_TEST_MMLU_TOKENIZER` opt-in，本轮实际运行，未跳过。
词表大小只取一次，避免 Rust tokenizer 在逐 token 验证中反复复制完整词表；
这属于本地准备工具效率修正，不是模型性能收益。

下一步补**服务端精确模型/tokenizer/template/依赖及 token parity**与有界 transport；
本地 tokenizer 核验不证明权重或实际服务身份。评分合同、Host/Manager/profile/
correctness/资源与清理门禁仍需闭合，不得为制造命中改变正式任务提示。

## 6. 源题/token 到离线 scorer 的连接

同一个 token 工具现在可通过 `--scoring-output` 额外生成私有
`aqkv-mmlu-scoring-bundle/v1`。它包含全部有序题目、标准答案、真实 token IDs、
完整候选消息/生成请求，以及 `aqkv-mmlu-source-bound-envelope/v2` 的合同字段。
**没有生成 B0/B1 作答，也没有发送请求**；源题、答案和 tokens 不进入公开仓库。

```bash
python scripts/tokenize_pujiang_mmlu_source.py \
  --assets /absolute/path/to/pinned-parquet \
  --reference /absolute/path/to/pinned-reference \
  --selection /absolute/path/to/selection.json \
  --benchmark-checkout /absolute/path/to/benchmark-at-8ed7ec9 \
  --tokenizer-directory /absolute/path/to/pinned-qwen35-tokenizer \
  --output /absolute/path/to/new-private-token-plan.json \
  --scoring-output /absolute/path/to/new-private-scorer-bundle.json
```

两个输出都排他创建、权限 0600。上下文超预算时保存完整 token plan 并非零退出，
但不导出可评分 bundle，不截断或删题来通过检查。

`validate_pujiang_mmlu_pair.py` 的旧 v1 模式继续只核验 supplied manifest。
新增 v2 模式必须同时提供固定 assets/reference/selection/tokenizer-directory，
从已核验源字节重新构建整份 bundle，再逐项比较：

- 题号、顺序、子集、答案、选项、validation 示例、源行及文本摘要。
- 真实 token IDs/count/hash、渲染模板摘要和完整候选请求摘要。
- temperature、seed、max_tokens、ignore_eos、n、stop、thinking 和 stream 的完整参数。
- 数据/benchmark/参考 scorer、tokenizer 文件与库版本、context 和选题摘要。
- B0/B1 必须各有每题一条结果，并匹配完整请求/渲染/token 摘要；失败仍保留计分分母。

仅修改存储文件及其配套哈希、声称 `tokenizer_verified=true`，无法跳过源文件重建。
v2 的 `--cases` 接收整份 scorer bundle，而非旧 v1 的题目列表。
envelope 从 bundle 的 `envelope_contract` 复制，再添加
`evidence_kind=development-fixture` 或 `unverified-stored-outputs` 与
`files_sha256` 的 cases/B0/B1 实际字节摘要。B0/B1 的逐题记录除原 v1 字段外，
必须提供 `request_sha256` 和 `rendered_prompt_sha256`。

```bash
python scripts/validate_pujiang_mmlu_pair.py \
  --envelope /absolute/path/to/source-bound-envelope.json \
  --cases /absolute/path/to/new-private-scorer-bundle.json \
  --b0 /absolute/path/to/stored-B0.json \
  --b1 /absolute/path/to/stored-B1.json \
  --benchmark-checkout /absolute/path/to/benchmark-at-8ed7ec9 \
  --assets /absolute/path/to/pinned-parquet \
  --reference /absolute/path/to/pinned-reference \
  --selection /absolute/path/to/selection.json \
  --tokenizer-directory /absolute/path/to/pinned-qwen35-tokenizer \
  --output /absolute/path/to/new-private-offline-report.json
```

本地实际完成 70 题接线检查：重新核验真实来源和 tokenizer，但作答明确为每臂
70 条**合成失败 fixture**，检查共享 scorer 的完整分母和错误保留；不是 70 次
模型调用或该模型准确率。公开精简回执见
[`PUJIANG_MMLU_SCORER_BINDING_20261010.json`](evidence/PUJIANG_MMLU_SCORER_BINDING_20261010.json)。

输出 `offline_scored_source_bound_runtime_unverified` 只增加本地来源绑定。
存储结果的哈希仍不是 HTTP 发送/服务端分词证据；真实服务身份、权重、token parity、
transport、Host/Manager/profile/correctness/资源、正式合同与路径观测仍未核验。
全部运行/质量/性能主张保持 false，不生成 AQKV eligibility 或正式五数据集成绩。

## 7. 已存原始 request/response 到 scorer 的连接

新增 [`pujiang_mmlu_transport.py`](../scripts/pujiang_mmlu_transport.py)。这只是离线
解析器，没有 HTTP 客户端、请求发送、重试或超时执行器，不能称为正式 runner 完成。
`validate_pujiang_mmlu_pair.py --stored-transport-root` 只允许来源绑定 v2 模式，
`--b0/--b1` 此时接收原始回执 JSON，raw bodies 分别放在 root/B0、root/B1。

回执使用 `aqkv-mmlu-stored-transport/v1`，字段为 schema_version、arm、evidence_kind、
scorer_bundle_sha256、request_method=POST、request_path=/v1/chat/completions、requests。
每题记录为：

```json
{
  "case_id": "source-bound-case-id",
  "request": {"path": "request.json", "sha256": "<64-hex>", "bytes": 123},
  "attempts": 1,
  "result": {
    "kind": "http",
    "status": 200,
    "body": {"path": "response.json", "sha256": "<64-hex>", "bytes": 456}
  }
}
```

非 HTTP result 使用 kind=timeout / connection-error / not-attempted / response-too-large，不添加不存在的
response body。not-attempted 的 attempts 为 0，其余为 1；这是存储元数据约束，
**不证明现实中没有重试或真正执行了超时控制**。上述伪哈希示例不能作为合法输入。

- 文件必须位于所属臂目录，拒绝绝对路径、越界和 symlink；单文件读取上限 1 MiB，
  比较真实长度与 SHA。该本地读取上限不是网络/进程运行的时限。
- 原始请求语义须逐项等于完整候选请求，不能只提供自称匹配的 request_sha256。
- 非 200、API/JSON/choices/content/usage 错误保留失败；仅取唯一 assistant choice，
  拒绝 unexpected thinking、提示 token 数不匹配、超预算/不一致 usage，不从错误正文提答案。
- stop/length 的原始文本不改写后交共享 scorer；length 留在完整分母与截断统计中。
  空答案不删题。usage 的提示数量匹配不证明服务端 token IDs 或模板一致。
- 文件身份缺失/损坏、漏题/重复、请求变更是输入完整性失败，阻断整份报告；完整记录的
  请求失败/坏模型响应则保留为失败题。共享 `transport_failures` 是规范化失败项数量，
  不等于全部都是网络故障；具体 error_kind 和原始字节摘要另行保留。

在第 6 节命令追加：

```bash
  --stored-transport-root /absolute/path/to/private-raw-records
```

envelope 的 B0/B1 文件哈希绑定 raw receipt，而非转换后的 outcomes；cases 继续绑定
整份 scorer bundle。私有输出额外保留规范化结果和逐题 raw 摘要，权限 0600，
排他创建，不覆盖旧报告。旧 v1/v2 的非 raw 模式保持原输出结构，不改写旧回执。

真实官方源题和 tokenizer 的 70 题两臂接线检查已完成，但响应是 **development fixture**：
每臂 20 条可解析合成答案（含 10 条 length），另有 HTTP/timeout/connection-error/
not-attempted/坏 JSON 各 10 条失败。模型调用为 0；合成答案按源答案构造，
因此不发布其“准确率”。公开摘要为
[`PUJIANG_MMLU_STORED_TRANSPORT_20261010.json`](evidence/PUJIANG_MMLU_STORED_TRANSPORT_20261010.json)。

最新正式控制为 [V5.4 对齐](PUJIANG_V54_ALIGNMENT_20261010.md)，与固定复用源码版本
分开管理。仍需实际服务身份/token parity、有界请求执行、正式任务/scorer 审阅、
签名 B0/B1、AQKV 准入及真实 Host/profile/正确性/资源证据，不能从离线解析推出已满足。

## 8. 10 月 11 日：实际本地 HTTP 的有界验证

新增独立的 [loopback fixture collector](PUJIANG_LOOPBACK_TRANSPORT_20261011.md)，
只连接字面量 `127.0.0.1` 的显式合成服务；上述 stored parser 本身仍没有网络行为。
已验证绝对总时限、无重试/重定向、正文大小与完整性、子进程回收，以及坏元数据
留下审计但阻断评分。新增 44 项测试，完整插件 715 项通过；另有十个持久化实际
本地 HTTP 场景，模型请求为 0，不运行源题、不宣称准确率。

这一步关闭开发 fixture 的有界采集缺口，**没有关闭正式远端 transport、服务身份
和 token parity 缺口**，不开放 CLI/生产入口或修改运行门禁。

## 9. 10 月 11 日：完整 token IDs 的离线评分门禁

新增 [已存 token parity 审计](PUJIANG_MMLU_TOKEN_PARITY_20261011.md)，将来源绑定
v2 请求投影接到固定 vLLM `/tokenize` 候选协议。两臂完整 IDs 比较及原始请求、
tokenizer 声明、context 与文件身份校验已接入 scorer；任何题不匹配时保留全部
诊断、`offline_quality=null`、CLI 非零退出，不接受仅数量相同。

从真实源资产/tokenizer 重建的 70 题用两组合成回执验证：复制本地 IDs 与只改
一个 ID 的同长度对照。它们不是服务端输出，模型/tokenize 网络请求均为 0。
新增 64 项 parity 与 1 项严格布尔握手测试，完整本地 **780 passed**，无 skip。
这只关闭已存对照工具缺口，不关闭真实服务/签名来源/正式 transport 或评分合同。
