# 当前 CPU reference 的 tile/page 成本复核

日期：2026-10-12。回应两个问题：tile 反量化是否真的消除了原开销；为什么不让
attention 直接消费量化 KV。仅分析已交付的 CPU 数学参考，不评价尚未接通的 NPU kernel。

## 1. 固定对象与可重跑证据

对象为 [`reference.py`](../src/vllm_ascend_adaptive_quantized_kv/reference.py)，
源字节 SHA-256：`c441aec1080a86152703e12b62e0f7f6a562356c76d724ddcd6cf41a347585ae`。
本轮没有修改该文件、Host、provider 或 kernel。新增
[`test_source_cost_counts_and_floating_attention`](../tests/test_reference.py)
拦截现有 `_dequantize_page` / `einsum`，统计调用次数、元素数、参与乘加的逻辑维度及
实际输入 dtype；同时与独立 dense CPU oracle 核对输出。

```bash
PYTHONPATH=src python -m pytest -q tests/test_reference.py \
  -k source_cost_counts_and_floating_attention
```

八个用例覆盖一/多 query tile、尾页、全未来页、混合请求、非连续布局、BF16/FP16
输入与 FP64 输入。私有离线追踪亦核对输入不变；没有计时、读取 HBM 计数器或启动模型。
这些是 CPU 操作计数与源码公式，不是 NPU 测量、峰值显存或性能提升。

## 2. 当前实现到底做了什么

每个 request 的每个 query tile 都遍历该 request 的全部已用 KV 页，分别进行 K/V
的 NZ 视图转换、浮点转换和 scale 乘法，再做浮点 QK、online softmax 和浮点 PV。
causal mask 在反量化与 QK **之后**应用，因此全未来页也已经读入并反量化。

BF16/FP16/FP32 query 的中间数学 dtype 均为 **FP32**；FP64 query 使用 FP64。
量化输入并不意味着 INT8 矩阵乘。GQA 共享 KV head 避免了完整 head 扩展副本，但不
改变重复分页读取的循环顺序。NZ `reshape` 是否复制受 stride 影响，不能一概称为零拷贝。

设一个 request 的新 query 长度为 Q、已有完整 KV 长度为 L、query tile 大小为 T、
page 大小为 B、KV head 数为 Hkv、query head 数为 Hq、head dim 为 D：

| 源码量 | 当前实现公式 |
| --- | --- |
| query tile 数 R | ceil(Q / T) |
| 有效 page 数 P | ceil(L / B) |
| K/V 反量化函数调用 | 2 R P |
| 反量化元素数 / scale 乘法数 | 2 R L Hkv D |
| 单遍反量化元素数 | 2 L Hkv D |
| INT8 源元素逻辑读取字节 | 2 R L Hkv D；不含 scale/索引/布局复制，也不是实测 HBM |
| QK、PV 各自的逻辑 MAC 数 | Q L Hq D；含之后被 mask 的位置，不是硬件指令计数 |
| 一对页的 FP32 输出载荷 | 8 min(B, L) Hkv D 字节 |
| 完整 BF16 KV 载荷 | 4 L Hkv D 字节；仅用于比较参考，不是本轮实际 fallback 测量 |

混合 request 按各自长度累加；共享物理页仍按每个 request 的循环重新处理。
一对页的载荷不等于峰值内存：转换输入/输出、旧页的暂时存活、scores、weights、
accumulator 和 allocator 都未包含。短 KV 时页级 FP32 载荷甚至可大于完整 BF16 KV。

## 3. 本轮小规模实际计数

以下 Hkv=2、Hq=4、D=64；实际调用与上述公式全部一致，输出核对通过。

| L / Q / T / B | K/V 调用数 | 反量化元素数 | 单遍元素数 | 重复倍数 |
| --- | --- | --- | --- | --- |
| 257 / 63 / 64 / 128 | 6 | 65,792 | 65,792 | 1 |
| 257 / 129 / 64 / 128 | 18 | 197,376 | 65,792 | 3 |
| 129 / 129 / 32 / 32 | 50 | 165,120 | 33,024 | 5 |
| (81,35) / (40,5) / 3 / 32 | 92 | 308,224 | 29,696 | 各 request 分别 14 / 2 |

第三例中，源码循环还包含 20 次全未来 K/V 页调用、50,176 个反量化元素。这是依据
query 绝对位置与 page 起点逐项判断的无效工作量，不是优化后的运行结果。

仅作规模说明：若假设 L=32768、Q=2048、T=64、B=128、Hkv=8、D=128，
R=32，则完整 INT8 K/V 为 64 MiB；当前循环反量化元素逻辑读取累计为 **2 GiB**，
对应 16,384 次 K/V 页调用。完整 BF16 KV 载荷为 128 MiB，一对 FP32 页载荷为 1 MiB。
**这不是 Qwen 的已核实配置、不是实际大规模运行、更不是 HBM 流量或 TTFT 结论。**

## 4. 对老师两点意见的结论

**tile 反量化不是免除反量化。** 当前实现以有限页级中间载荷替代完整浮点 KV，
但仍需要浮点转换和 scale 乘法；query 分块会重复这些工作，且当前还处理全未来页。
不能由“无完整 BF16 materialization”推导更少计算、更低总流量或更快 TTFT。
八例未发现数值溢出，不能把额外计算开销写成已经发生 numerical overflow。
这也不证明 tile 路线不可取：cache/register 复用、融合、并行度与实际 kernel 实现
可能改变成本，必须在接受的执行环境中分别测量。

**直接读量化 KV 要区分两层。** 第一层是 fused attention 按 block table 直接读取
INT8 NZ KV，在算子内部按需恢复浮点并计算 QK/PV；它消除了全局 dense 中间链路，
但不等于低精度计算。第二层是原生低精度 QK/PV；本 CPU reference 未实现这一层。
后一层需要另外明确 Q 和 softmax 概率的量化、累加类型、per-channel scale 的合法
处理、误差与输出兼容性，不能把现有浮点 einsum 改名后算作完成。

## 5. 下一步及停止条件

1. 先将这些成本约束纳入已接受的 provider/kernel 设计：不要逐 query tile 盲目重做
   同一页反量化；评估 query/KV loop 排布、页复用和全未来页裁剪，但不能用无限缓存
   重新引入完整浮点 KV。当前 CPU oracle 保持不变，避免混淆参照和 treatment。
2. Host/provider 及 B1 精度范围明确后，再对 fused 浮点消费路线做正确性先行的最小
   对照，分别记录布局转换、反量化、attention、workspace、graph 和端到端指标；
   正式任务合同不为制造缓存命中而改题或改预算。
3. 原生低精度计算作为独立候选，不预设可行或优胜。先取得实际算子契约/支持证据，
   再验证 scale、GQA、causal、混合批次及误差；不能静默改变正式 BF16/B0/B1 合同。

本轮不解除现有 V5.5、Host 接纳、signed receipts、任务/judge/沙箱、资源、路径及
correctness 门禁。任何后续数值、来源、graph 或资源失败均保留原始证据并停止。
