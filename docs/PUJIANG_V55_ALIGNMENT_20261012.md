# V5.5 控制基线与下一阶段顺序

核验日期：2026-10-12。这里只交付离线控制与源码复核，不授权正式执行。

## 1. 当前 authority

benchmark [PR #275](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/275)
已合并，当前基线为 `9e7f83388d72a37767fd9f24636219093ecbe662`。
[V5.5 原文说明](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/9e7f83388d72a37767fd9f24636219093ecbe662/docs/ACCEPTANCE_V5_5.md)
明确其为 active delivery test plan。V5.4 及更早文件保留为历史，不是现行执行依据。

原字节声明保存在
[V5.5 快照](evidence/PUJIANG_V55_CONTRACT_SOURCE_20261012.json)，
SHA-256 为 `9e64b0d19f84f9c4285fbdd9da12e2b6608ed02a8e9982bae9ec5c4f4ede538c`。
源码及文档均通过 Git blob SHA-1、长度和 SHA-256 核验；上游 schema、声明、镜像校验
与确定性 DOCX 重建的六项测试已在独立本地快照中全部通过。

两种文档身份必须分开：

| 文件身份 | 字节 / 页数声明 | SHA-256 | 本轮状态 |
| --- | --- | --- | --- |
| 仓库 semantic mirror | 571121 字节；cached 87 页 | `4f28cc44af069e3f2debd1f703926ba39bb3acee5ea64af899e05192c1a5bb41` | 已读取、核验摘要并确定性重建；cached 页数不是本轮渲染页数 |
| 单独声明的 canonical final | 637379 字节；83 页 | `bc61b65162952c28cde066ee3829aa58f15bf9ed0843d485e8b1a832410489d2` | 只有上游身份声明，未取得或校验该二进制 |

不能把前一文件的校验提升为后一文件已验证。

## 2. 未改变的正式规则

Qwen3.5-35B-A3B、2 张 910B2、TP2、BF16/bfloat16，以及 B0/B1 保持固定。
B1 仍指测试开始前签名冻结的最终引擎，不自动等于 AQKV ON；INT8 KV 是否属于该
B1 的允许精度范围及 manifest，仍无明确回复。E1/E2-G/E2-D 不启用、不排队。

A3 保留 30720 输入 / 2048 输出 / 32768 总上下文、30 分钟窗口；TTFT mean/p95/p99
限值 30/45/60 秒，TPOT 为 40/60/80 毫秒。Prefill throughput 只作诊断，不生成阈值。
A4 采用启动到停止的完整生命周期成本，分母只含测量窗口内成功且满足冻结 SLO 的
输出 tokens。五数据集与 task-quality-first 原则不变。不得现场调参、自动降级配置、
best-of 或事后改判。

Host correctness 中验证原生 fallback 是测试安全边界，不授权正式评测静默切换配置。

## 3. 新明确的 B0 责任与材料

`b0_receipt_gate.owner = release-and-evaluation-engineering`，即课题组发布与评测工程；
生产者也是该工程流程。**这是责任角色，不是已指定个人、已交付材料或 AQKV 准入决定。**

声明共要求以下 13 项实际材料：

| 范围 | 必需项 |
| --- | --- |
| 发布身份 | `signed_release_manifest`、`b0_oci_repo_digest`、`b1_oci_repo_digest` |
| 命令与运行 | `complete_server_argv`、`complete_client_contract`、`effective_runtime_receipt`、`startup_receipt` |
| 模型与数据 | `model_manifest`、`tokenizer_manifest`、`dataset_identity`、`arrival_sequence` |
| 输出证据 | `result_table`、`raw_requests` |

这些值必须来自实际发布工件及实际运行，不可预填或推断。材料须进入同一不可变
manifest、验签通过，并验证字段可生成唯一命令且与实际生效值一致。
声明没有提供我们可直接采用的签名格式、公钥或上述实际值；本轮不发明信任根或签名
验收协议。现有 token-parity 工具记录的配置/manifest 摘要仍不能验证原件及服务来源。

## 4. 当前独立施工与执行依赖

1. 本轮先完成 V5.5 的确切声明/文档身份同步，校验器拒绝旧 V5.4 回退、删除任一必需
   材料、单侧 OCI 冒充两侧、角色替换、推填实际值和原文/镜像混淆。保持所有执行主张为 false。
2. 复核新 Host main `82f586f8dcb5f92f8f3d55727bc05b3e2887606d`：目标 C8 分派及
   materialization 的八个函数 AST 相同；七组非数值 stub 复现只证明分支控制。
   [路径核验设计](PUJIANG_PATH_VERIFICATION_PLAN_20261012.md) 完成，实际路径回执仍未获得。
3. 继续送审当前插件 Draft PR #12；保持 Host #35/#37 原 head 不动。等待 #37 七文件
   测试候选的提交/正常 CI 路线答复，不能在旧产品分支上绕过设备和总 gate。
   七个候选文件在新 main 原字节不变，独立副本 patch apply-check 通过；这不是 CI。
4. 发布与评测工程交付正式身份材料；Host 接纳、C8 profile、正式选题/scorer、权利、
   judge/沙箱、资源与 correctness 分别核验。缺项不互相替代，也不由别的 MOD 回执自动解除。
5. 只有适用项目全部满足依赖，才连接真实有界 transport/token parity、目标路径及
   B0/B1。原始结果先入公共 benchmark，再回链 Issue 和同步网站。

截至本轮只读刷新：Issue #10/#1 与 Host #35/#37 没有新的负责人回复；#12 尚无评论
或审阅。公共 Plugin CI 的四项成功来自此前提交，不证明本次增量或 Host/NPU 验收。
PyramidKV 新增服务恢复及外部依赖催办不属于 AQKV 运行证据，不连接或复用其占卡服务。

新 Host `requirements.txt` 将 transformers 从 5.14.1 改为 5.19.0。本地 tokenizer 使用
4.57.1 的历史回执原样保留，不能将其视为新 Host 依赖或实际服务 token parity；未升级
本地/远端运行环境，也未重置 tokenizer/source pins。

```bash
python scripts/validate_pujiang_readiness.py
python -m pytest -q tests/test_pujiang_readiness.py
```

输出 `planning_valid_execution_blocked` 仅证明当前规划一致。99 项相关本地测试通过；
完整显式 opt-in 回归为 807 passed、零失败/错误/skip，默认为 783 passed / 24 skip。
24 项需外部源码/资产/tokenizer 的集成没有在默认环境执行；完整本地回执不能替代
公共 CI、Host/NPU 或正式成绩。构建与公共 CI 按各自确切提交单独记录。
