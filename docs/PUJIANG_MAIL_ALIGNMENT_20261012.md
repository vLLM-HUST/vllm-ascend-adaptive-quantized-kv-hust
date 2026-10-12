# 10 月 12 日邮件对齐与执行边界

本轮收到五封 GitHub 邮件通知，并回读 benchmark #254 的原始留言。
它们是 LatchMoE/PyramidKV 的进展与公共依赖请求，不是张老师给 AQKV 的新批准。
当前 benchmark main 仍为 `9e7f83388d72a37767fd9f24636219093ecbe662`，
[V5.5 控制](PUJIANG_V55_ALIGNMENT_20261012.md) 不变。

## 1. 通知逐项归类

| 原帖时间 (UTC) | 发件人对应作者 | 内容与本模块处理 |
| --- | --- | --- |
| 10-10 10:07 | Li-changwu | [LatchMoE 来源/审计进展](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254#issuecomment-6096408466)：其单卡环境不能完成指定 TP2；五数据集仍缺评分、权利、工具和发布身份。保留这些缺口，不移植其环境事实或降低正式配置。 |
| 10-10 15:36 | Li-changwu | [LatchMoE PR #120 阶段进展](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254#issuecomment-6099168342)：文本/图像和小图回执只证明局部恢复，当时数值测试仍有失败。不记作五数据集结果或 AQKV 可执行条件。 |
| 10-10 16:37 | Li-changwu | [后续 wheel/路由与启动进展](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254#issuecomment-6099772125)：多次启动、短请求不等于五项基准测试；不把其他 MOD 的单卡局部验证当成我们的 TP2 正式评测。 |
| 10-11 04:48 | Irisuko | [PyramidKV 服务恢复](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254#issuecomment-6105567880)：该 MOD 的 PR/服务及少量请求已恢复，但其 ARM64 容器仍无 AMD64 沙箱。服务、卡与资源归对方所有，本模块不连接、借用或停止。 |
| 10-11 04:59 | Irisuko | [公共依赖交付请求](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254#issuecomment-6105641130)：要求 AMD64 fresh-sandbox、到模型服务的连通性/认证、Harbor/agent 合同、judge/校准、HLE 与 SWE 范围及权利。是待交付请求，不是已交付事实或执行授权。 |

以 GitHub 原帖时间为上表权威；邮件到达时间与其略有差异。完整邮件、Message-ID、
邮件/原帖摘要和只读抓取回执仅保留在本地私有目录，不发布邮箱内容或凭据。
本轮只读刷新插件 #1/#10/#12 与 Host #35/#37，未发现新增负责人回复或 review。
该判断是本轮抓取时点，不是对未来状态的承诺。

## 2. 优先级与可独立推进项

1. 保留最高优先级的真实交付门禁：V5.5 AQKV/B1 范围、签名发布材料、合法 Host
   接纳/CI 路线、量化 profile/correctness 与设备资源。没有新材料时不推填、不启动。
2. 独立补齐 Terminal 的固定任务源码/政策与公开 inventory 的对应核验，复用
   项目组已固定的审计函数；与镜像层、沙箱、模型及评分执行分离。进度与实际
   验证范围记录在 [Terminal 来源核验](PUJIANG_TERMINAL_SOURCE_AUDIT_20261012.md)。
3. 后端、评分/权利负责人需提供“事项 -> 负责人 -> 交付入口/尚缺内容 -> 预计时间”。
   这是下一次交接所需字段，不臆定具体人、连接地址、认证方式或交付日期。
4. 后端真正交付后先审查权限、版本、有效资源、出网和到模型的连接；正式合同与
   当前服务门禁同时满足后，才能做 fresh-sandbox reference/failure verifier，
   然后匹配 B0/B1。本轮没有执行这些步骤。

不同项目角色需要协作的部分不能用我们的 CPU 测试替代。FrontierScience judge
必须保留分赛道合同与语义校准；HLE full/Gold 与图像输入不能自行缩减；SWE 的
agent/tools/budget/regrader 与任务权利仍需明确。之前已发布的确认请求仍开放，
本轮不重复催办、不发送 Slack、不创建自动任务，也不把 Draft PR 当作已合并贡献。
