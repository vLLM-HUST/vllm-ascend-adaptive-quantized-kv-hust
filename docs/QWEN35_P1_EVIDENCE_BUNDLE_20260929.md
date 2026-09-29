# Qwen3.5 P1 原始证据归档合同

日期：2026-09-29

## 目的

本合同补足 P1 correctness 的原始证据边界。它不执行模型、服务或 NPU，
只规定未来一次获批执行必须保留哪些文件，以及这些文件如何与 correctness
回执逐项绑定。

模板位于
[`evidence/QWEN35_P1_EVIDENCE_BUNDLE_CONTRACT_20260929.json`](evidence/QWEN35_P1_EVIDENCE_BUNDLE_CONTRACT_20260929.json)。
当前状态固定为 `preregistered_blocked_on_h3`，所有路径和哈希为空，不能被解释为
已完成的运行回执。

## 必须保留的原始文件

一个完整 bundle 共 26 项，覆盖：

- 授权后的 correctness 合同及最终 correctness 回执；
- 模型、tokenizer、profile、Host/plugin/Manager revision 和租约输入；
- 运行环境与实际启动命令；
- 两次资源快照、服务 stdout/stderr；
- TP2 两个 worker 的原始回执；
- Manager `discover/check/plan/run` 四阶段原始回执；
- 八个 correctness case 的逐项原始回执；
- graph lifecycle、native fallback 和 rollback/cleanup 回执。

校验器会把 Manager、worker、case 和 lifecycle 文件的实际 SHA256，与最终
correctness 回执中声明的 `raw_receipt_sha256` 逐项对照。仅在文件真实存在、非空、
大小与哈希一致、且 correctness 回执本身通过时接受 bundle。

## 路径与安全边界

所有路径必须是 bundle 根目录下唯一的 POSIX 相对路径。校验器拒绝：

- 绝对路径、`..`、反斜杠路径；
- 任一层符号链接或根目录逃逸；
- 缺失、空文件、目录、重复路径；
- 大小或 SHA256 不匹配；
- 与 correctness 回执未逐项绑定的替换文件。

校验器只读取显式列出的文件，不扫描主目录，不上传文件，也不修改原始证据。
运行结束后应将完整 bundle 放入负责人批准的证据载体；公开 Issue/PR 只引用已脱敏的
索引、哈希和获批 URL，不把密钥、内部路径或敏感环境变量提交到公开仓库。

## 使用方式

当前模板的静态检查：

```bash
python scripts/validate_qwen35_evidence_bundle.py \
  docs/evidence/QWEN35_P1_EVIDENCE_BUNDLE_CONTRACT_20260929.json
```

当前模板必须在完整性门禁下失败关闭：

```bash
python scripts/validate_qwen35_evidence_bundle.py \
  docs/evidence/QWEN35_P1_EVIDENCE_BUNDLE_CONTRACT_20260929.json \
  --require-complete
```

获批运行后，另存完整 manifest，不覆盖预注册模板，再执行：

```bash
python scripts/validate_qwen35_evidence_bundle.py \
  /path/to/completed-evidence-manifest.json \
  --root /path/to/immutable-run-root \
  --require-complete
```

PASS 只代表 P1 correctness 原始证据闭合。输出仍固定
`performance_authorized=false`、`quality_claim_authorized=false`、
`public_surface_authorized=false`；性能 pilot、质量结论和网站恢复各自需要独立合同。
