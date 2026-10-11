# HLE / SWE 固定来源离线核验

日期：2026-10-12。范围是源文件身份与公开 inventory/harness 的离线对应，
不是模型执行、官方评分、完整上游审计重跑或权利批准。当前控制仍为
[V5.5](PUJIANG_V55_ALIGNMENT_20261012.md)，插件仍 `import_only`。

## 1. 固定输入与复用边界

共享源码及报告固定到 benchmark
`8ed7ec9a16075c5bce5644a68a712139a6e5d403`。报告从新 main
`9e7f83388d72a37767fd9f24636219093ecbe662` 取得，但逐文件 Git blob 与旧 pin
完全一致，未改 reference HEAD。复用其 HLE source pins 和 SWE canonical、
test-list、Git blob 校验函数；不执行上游 task runner、solver、judge 或任务脚本。

| 输入 | 固定身份 | 本轮读取 |
| --- | --- | --- |
| HLE 原库 | GitHub `SKYLENAGE-AI/HLE-Verified@b705e0fb541c025a1532ce0d60d70ae2f53b00e0` | README、完整树身份、三个 LFS pointer 与公开清单；不是同名 HF 库 |
| SWE 数据 | HF `ScaleAI/SWE-bench_Pro@2d52cb3df914a3fcf80c7f66738b3a88ae37fc50` | `data/default/test-00000-of-00001.parquet`，6511689 字节 |
| SWE harness | GitHub `scaleapi/SWE-bench_Pro-os@66f92766bba642462d4bbe5479e83f91f9211862` | 完整 16077 项 Git tree 元数据与一条原始 test patch |

SWE Parquet SHA-256：
`1067f72c1f43bbc578276888625db3b836cebb535546c9eaf824c5d5dd104a7f`。
树校验重建每一级 Git tree 的二进制对象，根分别匹配
`471981d04bf02180e3a4c53da8da2f12fa989c4d` 和
`beb15336e37995a2dc2e5908e2bde931d59cef1e`；不是只相信 API 顶层 `sha`。
各文件的 Git blob 仍须与固定报告对应；树元数据不等于取得所有文件内容。

公开审计报告/相关元数据共 1558522 字节。HLE 原始三个 JSONL 声明合计
343137709 字节，本轮没有下载；SWE 的源数据和一条 32446 字节原始补丁留在
本地私有证据目录，不进入公开仓库或发行包。没有下载模型权重或镜像层。

## 2. HLE：核验清单，不冒充原始数据重跑

公开清单 2500 个唯一 ID、连续源行号、清单摘要及三份 LFS 身份均通过。

| 子集 | 全部任务 | 含题目图像 | 回答值类型 |
| --- | --- | --- | --- |
| Gold | 668 | 93 | 668 str |
| Revision | 1143 | 118 | 999 str / 109 int / 21 bool / 14 float |
| Uncertain | 689 | 131 | 689 str |

Gold 的 575 道纯文本题不等于完整 Gold。题目 `image`/`image_preview` 决定图像
输入需求；rationale 图像不能作为 solver 输入。必须保留回答的原始类型与 false/0，
不能用 truthiness 把它们算作缺答。本工具只检查清单中的类型标签与统计，
**没有重新解析真实 HLE 问题/答案，不能证明原始 false/0 值已重验**。

固定树未含 LICENSE/COPYING，README 未给出明确使用许可，只能记录 provenance
缺口，不推导法律结论或批准。full/Gold 范围、judge、提示、评分与图像后端仍未冻结；
PyramidKV 的 image-limit=0 不移植为 AQKV 环境事实。

## 3. SWE：真实源行与 harness 身份对应

实际解析固定 Parquet 的全部 642 行，核验 source-row canonical SHA、唯一 ID、
release 2.0.0、11 个 repo、base commit、hard 类型、问题/patch/test-patch 摘要与
测试列表数量。5778 项 harness blob 映射与固定完整树对应；gold patch 全部对上。
这不包含 task.toml/Dockerfile 等每个文件的内容重读及完整 descriptor/许可审计。

641 条 test patch 原字节相同；一条 Ansible 的 HF 字符串是 LF，固定 harness 是
CRLF。重新取得原 32446 字节补丁并核对 Git blob
`7e4e1cddeae4766e07dc5a44169d4e5e9009a0f2`；仅将 CRLF->LF 比较用于解释差异，
verifier 权威仍是原 harness 字节，未改 patch。

上游公开清单声明 5943 个唯一镜像层、压缩层合计 544122719405 字节。本轮只复算
这些元数据的去重/合计，不重验 registry manifest/layer bytes；不是解包存储、显存
或新的镜像测量。没有 Docker pull、fresh-sandbox、agent、测试执行或 solved rate。

## 4. 工具、测试与后续

[离线工具](../scripts/audit_pujiang_hle_swe_sources.py) 拒绝 pin/blob/树/清单摘要漂移、
重复 ID、缺 harness 映射、错误类型、改 verifier 权威和伪 runtime 标志；限制压缩
清单解压/输入大小，拒绝 symlink，排他创建输出。报告只有元数据，不复制问题、
答案、gold patch 或运行脚本内容。

```bash
python scripts/audit_pujiang_hle_swe_sources.py \
  --benchmark-checkout /path/to/pinned/benchmark \
  --published-root /path/to/verified-public-report-tree \
  --hle-pointer-root /path/to/hle-readme-and-lfs-pointers \
  --swe-data /path/to/fixed-default-test.parquet \
  --harness-tree /path/to/complete-harness-tree.json \
  --original-test-patch /path/to/original-crlf-patch \
  --output /path/to/new-source-linkage.json
```

需要明确提供固定 checkout；实际 SWE Parquet 解析另需 PyArrow，本地使用 22.0.0。
不把这个离线依赖加进零依赖的 runtime wheel。58 项新增本地测试通过，其中一项
显式使用本轮真实输入；默认 CI 不具备该私有源数据，必须明确 skip，不能冒充执行。
完整显式 opt-in 本地回归为 865 passed / 0 skipped；默认环境为 840 passed /
25 skipped。上游 HLE/SWE 工具相关 29 项另行通过。以上是本地验证，不是公共 CI
或 Host/NPU 验收，后续公共 CI 必须另按精确提交核验。
精简身份与边界见 [核验摘要](evidence/PUJIANG_HLE_SWE_SOURCE_LINKAGE_20261012.json)。

下一阶段仍需正式选题/scorer/许可、HLE 图像/judge、SWE AMD64 后端/固定 agent 与
budget/fresh-sandbox/verifier，加上 V5.5 签名发布身份、Host 接纳和实际服务
transport/parity。当前 `execution_authorized`、runtime/质量/性能主张均为 false，
本项缩小来源核验缺口，**不解除任何模型或设备执行门禁**。

来源：[固定 HLE 报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/8ed7ec9a16075c5bce5644a68a712139a6e5d403/reports/pyramidkv-hle-source-audit-20261009)、
[固定 SWE 报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/8ed7ec9a16075c5bce5644a68a712139a6e5d403/reports/pyramidkv-swe-source-audit-20261009)。
