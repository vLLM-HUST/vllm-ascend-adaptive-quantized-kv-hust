# 五数据集目标路径核验设计

日期：2026-10-12。状态：prepared-unreviewed；没有真实路径回执、Host API 接纳或启用。
本文件落实 TODO P0 的路径范围及核验设计，不改变题目、调度策略、评测合同或运行时。

## 1. 最新源码事实

来源固定为 Host `82f586f8dcb5f92f8f3d55727bc05b3e2887606d` 的
[attention_v1.py](https://github.com/vLLM-HUST/vllm-ascend-hust/blob/82f586f8dcb5f92f8f3d55727bc05b3e2887606d/vllm_ascend/attention/attention_v1.py)。
原字节 SHA-256：`d5ba5d0e32be90397f82241977d5fc58a59d1fcc4413273bf2333e731de56352`。
与 10 月 10 日 `f939ad7` 对照的八个 C8/graph 函数 AST 相同；kv_c8.py 原字节也相同。
八函数范围与七组控制流复现见
[精简来源回执](evidence/PUJIANG_NATIVE_PATH_CONTROL_20261012.json)。
该回执执行的是从源码提取的 chunked 方法，tensor、materializer、FIA 都为显式非数值
stub。它没有导入 Host/Torch/NPU 模块，不证明算子数值、真实调度可达、图安全或性能。

| 实际分支 | 源码条件与行为 | 不能混淆的状态 |
| --- | --- | --- |
| 新鲜 prefill | 所有 prefill 的 `seq_lens_list[i]` 等于该请求当前 Q 长度，且 float K/V 都存在；直接使用新 K/V、TND、无 block table | 即使 cache 的存储是 INT8，也不代表读了历史分页 INT8 KV |
| continuing/chunked prefill | 任一 prefill 的总 KV 长度大于当前 Q，或新 float K/V 不完整；prefill 分组 gather、NZ→ND、去 padding、反量化，最后 dense TND | 单 token 或多 token 只是 shape；provider eligibility 不能仅凭 Q 长度推断 |
| decode | DecodeOnly 或 mixed 的 decode 前缀；BNSD、paged INT8、block table、per-channel scale | paged INT8 decode 成功不证明多 token prefill 路径可用；MTP decode 不按 Q=1 硬编码识别 |
| capture | C8 forward 在正常 ChunkedPrefill 分派前先进入 `full_graph_fia` | capture 本身不是用户请求执行证据；必须匹配随后 replay 与 graph 身份 |

混合请求中，回退影响的是同一次 chunked 调用的 **prefill 分组**：例如新鲜请求和
continuing 请求合组时，新鲜请求也进入 materializer；decode 前缀仍单独走 paged BNSD。
因此不写成“整个混合批次都回退/全局雪崩”。另一触发因素是 float K 或 V 缺失，不能
将所有 dense 调用都归因为 prefix-cache 命中。

写 cache 时存在 offset，但该 paged dense materializer 仍只有 scale 反量化；精确 C8
候选的零 offset correctness 门禁继续保留。没有据此直接修 Host 产品代码。

## 2. 需要关联的证据

下列是插件侧待审阅的证据需求，不是 Host 已公布的事件 schema；不私自打补丁/读私有
注册表或将 proposed observer 当作已可用 API。先由接受的 Host 接口提供相应语义与 provenance。

| 粒度 | 必需证据及核验 |
| --- | --- |
| 整轮/两臂 | 正式 task manifest、冻结输入/到达序列、签名 runtime/model/tokenizer 身份、B0/B1 含义；与 V5.5 实际发布回执绑定 |
| 任务/请求 | 不可变 task ID、请求 ID、状态、原始输出；客户端 cache usage 不能独自证明 C8 层的执行路径 |
| TP rank / 层 / dispatch | Host 来源、rank、layer、调用/批次身份、模型 attention 类型；分别关联十个 full-attention 层，其他 30 个 linear-attention/Mamba 状态不计入该 KV 机制 |
| 实际 shape | Host phase、累计 Q 边界、总 KV 长度、有效/补齐行、decode/prefill 分组、block table/布局/dtype、profile 身份；拒绝用全批 num_tokens 推定逐请求 continuing prefill |
| 路径结果 | eligibility 判定及拒绝原因、provider 接受与实际调用、输出提交、native 分支、materializer 调用/影响请求及触发原因；安装/启用/eligible 三者均不等于 executed |
| graph | eager/capture/replay 明确区分；Host 生成的 graph/pool/shape 身份、执行关联、错误及清理；capture 样本不混入真实执行计数 |
| 完整性/开销 | 应有调用数、事件数、drop/采样/未知数、观察范围与打点配置；遗漏不能按零处理，不能把打点耗时当优化收益 |

先保留安全的标量身份、长度和路径结果；不为遥测额外复制完整 KV/tokens 到主机，也不
引入热路径同步以补齐计数。默认关闭路径必须无 callback、同步、分配或标签新增，沿用
[Host 合同提案](HOST_CONTRACT_PROPOSAL.md) 的边界。输出摘要必须能回溯原始事件字节与哈希。

## 3. 判定顺序

1. 正式合同、来源、真实执行环境、Host/profile/correctness 或覆盖完整性缺失：`blocked`。
   不能因“计数为零”跳过缺失门禁，也不把旧源码控制流复现放入 path_observations。
2. 在满足合同和可信观察条件的实际任务里，目标路径确实没有被执行：`not-exercised`。
   保留新鲜 prefill、decode、linear attention、ineligible、native fallback 的各自计数与原因。
   有 continuing 形状却始终回退时，记录“目标机制未作用”，不能抹去该负向结果。
3. 路径来源、资格、provider 执行/完成、两 rank 与层范围及 correctness 均可追溯时，
   才将项目列为 `applicable` 并评估匹配 B0/B1 的任务质量。`applicable` 本身不证明正收益。

五项都使用这一判定顺序，但 judge 的推理与 target solver 分开，agent/tool 后端与
目标服务分开，图像题与 linear-attention 状态不归入普通 C8 KV。实际采样或工具预算
保持正式合同不动；不用预热/复制长前缀/删短题来人为制造命中，不继承 PyramidKV 的
4096-token 条件。所有正、零、负、blocked、not-exercised 结果均保留。

## 4. 未闭合项

本轮只完成新源码复核与核验设计。五项矩阵仍 blocked、path_observations 为空；
observer/provider 接纳、真实 graph/replay、两 rank 路径完整性和任务质量都未运行。
若后续接口不能提供上述关键证据，须先明确该依赖而不是给缺失字段填默认值。
