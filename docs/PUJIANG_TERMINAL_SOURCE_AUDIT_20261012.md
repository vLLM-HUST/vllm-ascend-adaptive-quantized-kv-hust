# Terminal 固定来源与任务政策核验

日期：2026-10-12。离线工具只读文件、计算摘要和解析配置，不执行任务代码，
不调用 Harbor/Docker/模型或 registry。规划权威仍为 [V5.5](PUJIANG_V55_ALIGNMENT_20261012.md)，
本项不能解除 AQKV/B1、Host 或设备门禁。

## 1. 固定来源与方法

共享 benchmark 固定 `8ed7ec9a16075c5bce5644a68a712139a6e5d403`。复用已核验
摘要的 `audit_pyramidkv_terminal.verify_snapshot` 和
`prepare_pyramidkv_terminal.pin_task_image`，不重写核心字节核验或 pin 语义。
任务库固定 `harbor-framework/terminal-bench-2-1@7131e4375048a0e408a8fb404b5f499d726b695b`，
独立读取该 commit 的 tree 根为 `ddbd9031e59804a04e24019fc408d51b56a4e773`。

[本模块工具](../scripts/audit_pujiang_terminal_source.py) 有限执行以下步骤：

1. 检查 checkout 根/HEAD 和依赖源码摘要；公开报告逐文件核对固定 Git blob。
2. 重建来源的每一级 Git tree 与固定根，不能只信 API 的顶层 commit alias。
3. 限制 snapshot 总计 64 MiB/1101 文件，拒绝 symlink/额外文件，核验每个原始
   文件的字节、Git blob、大小和 SHA-256。
4. 逐任务对齐真实 TOML 与公开清单/映射：required assets、环境、agent、verifier、
   资源、超时、MCP 数量、源修订、镜像 tag/digest 和层元数据。
5. 只在内存预览 tag -> manifest digest，复用上游函数验证其他 TOML 语义不变。
   默认不写 staged 任务/job/执行合同；输出绑定摘要，不输出题目、解答或脚本内容。
   显式 supplied staged 输入的后续比对见第 5 节。

完整源码包曾传输截断，服务端不支持 Range。本轮保留有限重试回执，随后改用
固定 Git blob 获取源码，不声称完成 archive SHA 或上游 fetch 全流程重跑。
没有取得镜像层或服务模型权重；任务自带测试资产仅作不执行的字节输入。

## 2. 实际结果与声明边界

首次来源轮实际读取并核验全部 **1101 文件 / 59252655 字节 / 89 任务**；其中任务
目录 1035 文件，Git 模式为 1023 个 `100644`、12 个 `100755`。89 个完整 TOML
政策与 tag -> digest 内存预览均完成对应，未改原始输入、未产生 staged 执行目录。

真实来源核验和集成以本轮排他创建的 `terminal-source-linkage.json` 为准，
精简公开结果见 [来源摘要](evidence/PUJIANG_TERMINAL_SOURCE_LINKAGE_20261012.json)。
原始输入、逐任务绑定和邮件/GitHub/失败传输回执仅在本地私有目录，不入公开仓库
或发行包；公开摘要只含来源身份、计数、资源/时限和证据边界。

- Git `100644/100755` 元数据与原始字节核验不代表 staged 执行位已恢复。
  本地输入保持 `0600`，不让任务脚本变为可执行任务。
- linux/amd64 image mapping 可与原 TOML 对齐，但未重新取得 registry 原始
  manifest/config 或层。因此不是新的 registry/image 内容审计。
  只复算公开元数据的 319 唯一层 / 16699830105 压缩字节；两个任务的 descriptor
  声明总和分别比去重字典高 32/93 字节。按上游两种口径保留差异，未重验原 manifest
  的重复 descriptor，不改上游数值，也不把字典合计等同 descriptor 总和。
- 根 LICENSE 与来源对应，不批准第三方文件、镜像或下载资产的使用权。
- 独立来源审计不自动升级旧 v1 candidate-plan 的 `source_membership_verified=false`。
  显式提供完整 source/staged 输入、重新核验成功的单次回执才可绑定资产，见第 5 节；
  历史 v1 回执、单纯声明摘要或 separate source report 不能继承此状态。
- `task_quality=null`，模型/任务执行数为 0；ready/授权、runtime、质量与性能主张
  均为 false。Harbor schema、沙箱/verifier 和真实服务未验证。

## 3. 后端必须保留的配置

固定任务清单声明 CPU 1/2/4、内存 2048/4096/8192 MB、存储 10240 MB、GPU 0；
`allow_internet=true`、build timeout 600 秒。这些不是后端有效资源或每题必然
出网的运行观察；实际源码取得后逐任务核对，后端交付时还需验证有效值。

agent timeout 为 600–12000 秒，verifier 为 360–12000 秒。完整清单的时限之和
只是静态预算，不是实际耗时或正式子集工期。不能为了旧 45 分钟诊断 runner
截短正式任务、跳 verifier、丢失败题或改成绩分母。资源不足须在运行前报告，
由负责人确认后端/时段与正式协议，不共租对方运行中的卡或服务。

Harbor 0.24.0/Terminus-2 的 tool/context/token/retry 仍是待审阅候选协议。
源码可读不等于批准，还需模型连通性/认证、实际版本和有效资源、fresh-sandbox
生命周期及 reference/failure verifier。这些本轮均未执行。

## 4. 使用与验证

工具要求 Python 3.11+ 的 `tomllib`，不增加 Python 3.10+ 零依赖 runtime wheel
的依赖。输入可由固定 source-only 下载或 Git blob 路线取得：

```bash
python scripts/audit_pujiang_terminal_source.py \
  --benchmark-checkout /path/to/pinned/benchmark \
  --published-root /path/to/pinned-public-report-tree \
  --snapshot /path/to/original-terminal-source-files \
  --output /path/to/new-private-source-linkage.json
```

输出拒绝覆盖旧回执，源码/压缩元数据受大小限制。真实集成显式要求
`AQKV_TEST_TERMINAL_SOURCE_ROOT` 与 `AQKV_TEST_BENCHMARK_CHECKOUT`；缺真实输入或
Python 3.10 无 `tomllib` 时相关用例明确 skip，不伪装为实际执行。
本轮测试和打包计数见公开摘要/私有原始回执；公共 CI 必须按新提交另核验，
不能继承旧 head 绿灯。Draft/CPU 测试不等于 Host 接纳或已合并贡献。

来源：[固定公共报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/8ed7ec9a16075c5bce5644a68a712139a6e5d403/reports/pyramidkv-terminal-source-audit-20261009)、
[固定任务仓库](https://github.com/harbor-framework/terminal-bench-2-1/tree/7131e4375048a0e408a8fb404b5f499d726b695b)。

## 5. 后续：真实待运行副本绑定

2026-10-12 后续补齐原 v1 的实际输入比对缺口，不创建新的执行协议，不调用上游
`prepare` 的 registry 全审计，不以已发布的镜像 metadata 冒充原始 registry 回执。
同一个 source audit 可显式接收 `--staged-tasks` 和 `--candidate-inventory`；必须同时
提供，先重验完整固定原始 source 和公开报告，再核对给定任务副本：

- Task IDs 必须是唯一真实 source members；副本目录只允许这些任务的精确文件和
  目录集合。拒绝缺文件、多文件/空目录/任务、symlink、超量输入。
- 除 `task.toml` 中一次 tag -> 已核对的 manifest digest 变更外，题目、解答、
  verifier、Dockerfile 和其他资产全部保留原始字节/Git blob/大小。TOML 必须
  逐字节等于固定共享 pin 函数输出，不能顺便修改预算、MCP、资源或任务时限。
- 本地权限严格等于 Git `100644/100755` 的 `0644/0755`；缺执行位、额外写权限、
  setuid 等均拒绝。这里只读取副本权限，不 chmod 或执行任何脚本。
- inventory 摘要口径为 `encoded` 的 sorted/indented JSON + newline：source-row
  是原公开清单行；staged-files 是逐文件 bytes/sha256/git_blob_sha；file-modes
  是任务相对路径到 Git mode 字符串；task-policy 是原始 environment/agent/verifier。
  pin 单独绑定 `pinned_image`。自己重算被篡改副本的摘要不能绕过原字节核对。

本轮使用已有下载，只在私有目录创建全部 **89 任务 / 1035 文件** 的副本用于审计，
原 snapshot 保持 `0600`；无新资产下载、镜像层、服务/job 创建或脚本执行。所有
源码任务只是 copy-audit inventory，不是经批准的正式执行集。回执包含副本摘要与
计数，不公开题目/解答/资产。原始来源历史回执不覆盖。

```bash
python scripts/audit_pujiang_terminal_source.py \
  --benchmark-checkout /path/to/pinned/benchmark \
  --published-root /path/to/pinned-public-report-tree \
  --snapshot /path/to/original-terminal-source-files \
  --staged-tasks /path/to/private-candidate/tasks \
  --candidate-inventory /path/to/private-candidate/inventory.json \
  --output /path/to/new-private-staged-linkage.json
```

成功状态 `terminal_staged_assets_bound_execution_blocked`，只提升实际 source-member/
staged-byte/local-mode 对应。candidate-plan validator 可附加同样的 `--published-root`、
`--snapshot`、`--staged-tasks` 三项，将同一次计划校验绑定这些实际输入；不提供
三项时保持旧 v1 的不绑定语义。少任一项、任一文件/摘要不符都失败，不输出成功。

`task_assets_verified=true` **不含** registry layer/Harbor schema/真实 backend/服务
激活/授权/权利/完整 verifier 与任务质量；这些仍 false/null。正式任务集合及 V5.5
范围、实际镜像和依赖、连接/认证、Host/profile/correctness 等仍待交付。
