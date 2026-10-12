# Draft PR #12：离线数据流审查与修正

日期：2026-10-12。起点为插件 `920b8ba3427d7aed61345f692721c64e353a30d2`。
本次重点复核 Pujiang 输入、配对、评分、来源加载、transport/token parity 和控制入口；
不是整仓安全审计、外部合同审阅、Host CI 或设备验收。

## 已复现并修复

| 问题 | 旧行为 / 影响 | 修复及反例 |
| --- | --- | --- |
| Frontier judge 响应校验不完整 | error/refusal/wrong-role/wrong-index 响应只要含可解析 verdict，就可能记成 graded；即使 fixture 也不能把失败当分数 | 拒绝非 null API error、非整数零 index、非 assistant role、拒答和非空 tool calls；保留 judge_error、完整任务分母及截断标志。真实固定 aggregate 集成证明该赛道分数为 null、不丢题 |
| Frontier 报告权限依赖 umask | 缺少其他离线 CLI 的私有权限设置；宽松 umask 下可成为 0666 | O_CREAT/O_EXCL 以 0600 原子创建，保留拒绝覆盖；umask=0 的反例验证权限 |
| 源码包内 README 链接失效 | MOD_METADATA.json 存在于仓库却未进入 sdist；旧链接检查只核对 checkout，旧发行校验未检查源码包内目标 | 将公开 metadata 加入 sdist；发行校验直接遍历 archive 的本地 inline Markdown 链接，不解包、不依赖 checkout，并拒绝缺失/越界/链接成员。旧包真实反例已拒绝 |

修复前新增的 10 个反例全部失败；修复后 Frontier 的 91 项针对性测试通过。
判分检查不阻止 judge 自身生成 reasoning：这是不同于 solver 的合同，不能擅自套用
solver 的非 thinking 限制。仍不证明 judge 语义校准、实际源题或真实作答前冻结。

## 已核对的边界

- MMLU 的 B0/B1 完整题目集合、请求/预算绑定、失败/截断保留与原固定 scorer。
- source-bound 模式重建源题、selection、tokenizer 与评分 bundle，不由存储字段自称已验证。
- 原始 transport 文件按摘要/字节数绑定，正文最多 1 MiB；拒绝路径穿越与 symlink。
- token parity 缺项、类型错误或不匹配会阻断判分；声明的 tokenizer/来源不自动成为真实服务证明。
- loopback transport 只连带 nonce handshake 的 127.0.0.1 fixture，有硬截止和自有进程清理；不连接模型或未知服务。
- 固定共享源码及依赖全部验摘要后才执行，并恢复临时 module 注册。
- HLE/SWE/Terminal 的来源、任务/权限绑定与实际执行、权利、judge/沙箱分别记录。
- V5.5 是当前 authority；旧 V5.4 控制入口已修正，历史日志及分阶段计数不覆写。

未新增真实 HTTP、Host/provider/kernel、工作流或运行开关；manifest 仍 import_only。
上述核对没有发现需要改动的实际配对/失败分母逻辑，不为了增加代码量重写已有 scorer。

## 剩余限制

部分顶层 operator 指定 JSON/源文件仍直接 read_bytes，没有统一资源上限；上述
1 MiB 是 transport 正文上限，不能称为整个 CLI 的内存上限。当前工具只接受操作人
提供的离线输入，不能当作公开的不可信文件处理服务。后续若引入统一输入上限，必须
先评估全量任务/token bundle 规模并给出显式合同，不能用任意小上限偷偷拒绝正式任务。

其他历史 CLI 有“创建后 chmod”模式；本轮修正的是 Frontier 缺少私有模式的问题，
没有宣称所有 CLI 的创建窗口均已审计。无外部 judge/运行验收，无法在此解除任务门禁。

## 验证和成本复核

判分修复与成本测试阶段完整显式回归为 959 passed / 0 skip；新增发行链接测试后，
最终完整显式本地回归为 **969 passed / 0 skip**，默认本地为 941 passed / 28 skip。
完整回归包含外部固定源码/资产/tokenizer
的 opt-in 集成。公共 CI 必须另按新 head 核验，不能继承 `920b8ba` 的绿灯。
八项新增 CPU 操作计数与独立 dense oracle 测试证明当前 reference 的重复反量化与
浮点计算行为，详见 [tile/page 成本复核](REFERENCE_TILE_COST_REVIEW_20261012.md)。
没有把 CPU 计数升级为 HBM 流量、峰值显存、TTFT、NPU 正确性或正式贡献评价。
