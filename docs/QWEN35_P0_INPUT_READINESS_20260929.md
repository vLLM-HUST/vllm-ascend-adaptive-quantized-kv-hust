# Qwen3.5 P0 输入就绪回执

日期：2026-09-29

机器可读记录：
[`evidence/QWEN35_P0_INPUT_READINESS_20260929.json`](evidence/QWEN35_P0_INPUT_READINESS_20260929.json)

## 结论

P0 的学生侧准备已完成到可公开复核的边界，但真机执行仍被外部输入阻塞，当前
`execution_authorized=false`。本次只做了公开 PR 状态核验和分配执行环境的只读
资产盘点，没有下载模型、启动服务或占用 NPU。

## 已就绪

- 插件位于公开组织仓 `vLLM-HUST/vllm-ascend-adaptive-quantized-kv-hust`；
- 插件 PR #4 为 Ready、mergeable，已请求 `ShuhaoZhangTony` review；
- PR #4 在 `1b2592f` 上的 CI run `36442127172` 四项全绿；
- 目标模型 revision、TP2、APC、MTP2、async、编译模式、benchmark commit、
  scenario 和 baseline/treatment 门槛已经预注册；
- 迁移前仅本项目不可再生资产的定向备份已经完成并校验。

## 尚未就绪

1. 当前分配执行环境中没有找到指定 revision 的
   `Qwen/Qwen3.5-35B-A3B` BF16 权重与 tokenizer；
2. 没有找到与该 revision 绑定、完成校准和审计的 zero-offset C8 profile 及 TP2
   分片；
3. Host PR #35 仍是 Draft，当前 head 的 pre-commit/CI gate 未通过，也没有
   accepted Host revision；
4. Host PR #37 的 CPU UT 已通过，但必需的 source-change/select-tests 门禁仍未
   完整结束；
5. 两张独占 910B2 的窗口和清理负责人尚未分配；
6. 插件 PR #4 尚未完成 reviewer 接纳与合并。

现有公开 Host 源码中可见的 `Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp` 测试夹具是
W8A8 目标，不能替代本项目预注册的 BF16 baseline 与 calibrated INT8-KV
treatment。

## 负责人需一次性确认

1. profile 采用 ModelSlim-compatible BF16+C8 sidecar/checkpoint，还是
   Host-owned external profile source；
2. profile 生成/校准负责人、artifact revision、provenance 和 TP2 分片规则；
3. PR #35 是接受、修改还是替换，并给出最终 accepted Host revision；
4. 基础设施迁移后精确 BF16 模型与 tokenizer 的位置；
5. 两张独占 910B2 的执行窗口与清理负责人。

上述输入齐全后，先执行 Manager/NPU correctness matrix；全部通过后，才运行唯一
一组预注册 baseline/treatment pilot。任何门禁失败都保留原始证据并停止，不产生
性能结论。
