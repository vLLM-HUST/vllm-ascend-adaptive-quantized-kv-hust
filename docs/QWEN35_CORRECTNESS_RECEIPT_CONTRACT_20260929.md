# Qwen3.5 P1 正确性回执合同

日期：2026-09-29

本合同把既有 H4/P1 八阶段 correctness matrix 转成机器可校验的执行与验收边界。
它不实现 Host provider，不生成校准 profile，也不授权当前环境启动服务或使用 NPU。

## 当前状态

[`evidence/QWEN35_CORRECTNESS_MATRIX_CONTRACT_20260929.json`](evidence/QWEN35_CORRECTNESS_MATRIX_CONTRACT_20260929.json)
保持：

- `status=preregistered_blocked_on_h3`；
- `execution_authorized=false`；
- Host/plugin/Manager revision、模型与 tokenizer digest、profile digest/TP2 分片、
  资源租约和 numerical oracle 阈值均为空。

任一输入为空时，校验器拒绝授权后的 correctness receipt。不得用当前模板启动任务。
对应的 Manager 命令映射、隔离配置、dry-run 和清理顺序见
[`QWEN35_MANAGER_CORRECTNESS_RUNBOOK_20260929.md`](QWEN35_MANAGER_CORRECTNESS_RUNBOOK_20260929.md)。

## 固定执行顺序

1. provider disabled，验证原生路径完全等价；
2. provider enabled 但请求不符合 eligibility，验证原生 fallback 完全等价；
3. APC continuing prefill eager；
4. `FULL_AND_PIECEWISE` capture、首次 replay 与后续 replay；
5. 加入 MTP2，保持同步调度；
6. 最后加入 async scheduling；
7. mixed decode 与 continuing-prefill batch；
8. unsupported case fail closed，并完成 cleanup/restart。

每一步必须绑定同一份授权合同、原始回执 SHA-256 和 numerical oracle；顺序、覆盖或
身份发生漂移即失败。

## 强制回执

完成后的单一 correctness receipt 必须同时证明：

- Manager `discover/check/plan/run` 四阶段全部通过并选择预期 extension ID；
- TP2 rank 0/1 都加载合同指定的 profile shard，只覆盖十个 full-attention layer，
  拒绝三十个 linear-attention layer；
- 八个 case 全部通过且输出 token IDs 与 dense baseline 一致；
- capture、首次 replay、后续 replay 全部通过，provider identity 不漂移；
- native fallback、rollback、uninstall、service exit 和 device release 全部通过；
- 退出后本项目进程数为零，设备内存已释放；
- 所有原始回执均给出 SHA-256。

运行：

```bash
python scripts/validate_qwen35_correctness.py \
  /path/to/authorized-contract.json \
  --receipt /path/to/completed-correctness-receipt.json
```

校验通过只表示 P1 correctness 闭合；输出固定
`performance_authorized=false`。性能测试仍需按独立预注册合同和资源门禁执行，不能从
正确性回执推导加速、质量或 public-surface 结论。
