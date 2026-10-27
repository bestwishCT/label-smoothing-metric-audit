# 困惑度更差但 BLEU 更好：标签平滑的评价方向

论文报告epsilon_ls=0.1，perplexity更差，accuracy/BLEU更好。 这篇围绕一个新阅读问题，将D1–D4逐项核对。相同论文可以提出不同问题；本次调用与旧表格读数任务独立，不复用旧回答。

## 问题、方法与实测

完整问题见[task.txt](task.txt)，运行前标准见[criteria.md](criteria.md)，冻结哈希见[protocol-lock.json](protocol-lock.json)。只给所选论文片段，逐行编号，要求D1–D4 JSON；服务无历史无工具，先dry-run逐字核对message后真实调用一次。

模型：chenkun/gpt-5-mini；completion id：chatcmpl-EdW4ARQjQDxIWrJa3RVgBAKkpH3V5；开始时间UTC：2026-10-27T07:39:21.492596+00:00；总耗时41.448秒；prompt/completion/total tokens：542/5431/5973。费用null，服务未返回逐次货币扣费。响应中的账户余额脱敏，模型content逐字保留；未伪造费用或实验。

机械验收：ranges_valid，4段合法范围，错误=[]。语义判定见每题核查；不将合法范围当作支持性证明。

输入[request.txt](request.txt)、预检[preflight.json](preflight.json)、响应[response.json](response.json)、原答[output.json](output.json)、元数据[metadata.json](metadata.json)、检查[validation.json](validation.json)均公开。

## 四道子题：原文、原答与作者核查

### D1：逐字核对label smoothing的值及对perplexity、accuracy、BLEU的方向；是否存在逻辑矛盾？这三个指标是否同义？

**结论**：论文报告epsilon_ls=0.1，perplexity更差，accuracy/BLEU更好。

**论文原文**：物理页8，P8:5–6。保留前后句如下：

```text
Label Smoothing During training, we employed label smoothing of value ϵls = 0 .1 [36]. This
hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.
```

**模型原答（原样）**：

```text
基于给定证据（P8 第5–6行）：论文在训练中使用了标签平滑值 ϵ_ls = 0.1，并明确写道“这会使困惑度(perplexity)变差，因为模型学得更不确定，但提高了准确率(accuracy)和 BLEU 得分”。因此：
- 标签平滑的数值：ϵ_ls = 0.1（证据见 P8 行5）。
- 对三项指标的方向：平滑后困惑度变差（即增加或变大），准确率提高，BLEU 提高（证据见 P8 行5–6）。
- 是否矛盾：在字面上并不存在逻辑矛盾。困惑度衡量模型对真实标注分布的平均似然（或负对数似然的指数化），标签平滑通过将目标分布“抹平”降低了模型对某一类的确定性，从而会增加负对数似然/困惑度；但准确率是基于 argmax 的离散判定（是否选对类别），BLEU 是基于输出序列与参考的 n-gram 重合度的序列级评价，两者可以在模型变得“更不自信”但更泛化的情况下提升。因此三者并非同义指标，它们度量不同性质（详见“推导”）。

推导（说明三指标不等同）：
- 困惑度/负对数似然反映模型为真实标签分配的概率大小的平均水平，概率分配越尖锐（对正确类的质量越高）通常困惑度越低；
- 准确率只关心最大概率所对应的类别是否正确（argmax），对概率分布的形状敏感度低；
- BLEU 为序列级的 n-gram 准确度（并含简短惩罚），取决于生成解码策略（如 beam search）和局部 n-gram 匹配，不直接等同于单步困惑度。

因此 P8 中对三项指标方向的陈述既一致又可理解为经验观察而非自相矛盾的数学矛盾（证据：P8 行5–6）。

模型 limitations字段：仅依据 P8 给出的两行陈述；关于为何三指标出现相反趋势的详细理论推导与实证分析不在 P8 的给定行内，本文的因果解释为教学推导而非作者在此处提供的证明。
```

**作者推导**：低perplexity通常更好，accuracy/BLEU高通常更好；目标概率、argmax分类和生成序列匹配不是同一对象。该句是特定训练经验，不是任意平滑均改善BLEU的定理。

**逐题核查**：原答方向正确；“会增加NLL”应收紧到本文观测/某些情况，非每个样本必然。


### D2：论文给定页是否定义平滑分布？教学采用 q=(1-epsilon)onehot+epsilon/K，K=4、epsilon=0.1，写出分布和归一化总和；与只分给错误类的约定区别？

**结论**：给定片段没有定义平滑分配；教学约定全类均匀混合。

**论文原文**：物理页8，P8:5–6。保留前后句如下：

```text
Label Smoothing During training, we employed label smoothing of value ϵls = 0 .1 [36]. This
hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.
```

**模型原答（原样）**：

```text
证据（P8 第5行）仅给出平滑强度 ϵ_ls = 0.1，并未在本段显式给出平滑目标分布的解析式或如何在 K 类上分配平滑质量（即 P8 在给定行没有明写 q 的具体公式或是否将 ε 分配到包括真实类在内的所有类）。因此平滑分布的具体实现细节在该摘录中是未定义的。

推导（教学采用的常见定义）：
- 常见且被广泛采用的一种定义为 q = (1-ϵ)·onehot + ϵ/K·1 ，即把 ε 均匀分配到所有 K 个类上；在这种约定下，真实类的目标概率为 1-ϵ+ϵ/K，其他每类为 ϵ/K。给定 K=4, ϵ=0.1：
  - 真实类：q_1 = 1 - 0.1 + 0.1/4 = 0.9 + 0.025 = 0.925
  - 其他三类：q_i = 0.025（i=2,3,4）
  - 归一化和：0.925 + 3×0.025 = 1.0（和为 1，归一化成立）。

与“只把平滑质量分给错误类”的约定区别（另一种常见约定）：
- 若把 ε 完全只分配给错误类（不把质量再分到真实类），则真实类概率为 1-ϵ，其他每类为 ε/(K-1)。同样 K=4, ϵ=0.1 时：
  - 真实类：q_1 = 0.9
  - 其他三类：q_i = 0.1/3 ≈ 0.033333…
  - 归一化和：0.9 + 3×0.033333… = 1.0（亦归一化成立）。

结论：P8 明确给出 ϵ_ls = 0.1（证据：P8 行5），但给定段落未指明采用哪种具体分配约定；上面给出的两种分配是常见的教学定义（推导），并分别列出了数值与归一化检验。

模型 limitations字段：P8 提到 ϵ_ls = 0.1，但在给定行未提供平滑分布的数学形式或引用的实现细节；上文的 q 公式与数值是教学推导/常见实现，并非 P8 在摘录中明确给出的内容。
```

**作者推导**：K=4、ε=0.1时q=[0.925,0.025,0.025,0.025]，总和1。仅给错类则[0.9,0.1/3,0.1/3,0.1/3]。不能仅凭ε相同假定两实现相同。

**逐题核查**：原答分清论文值和教学定义，两分布正确。


### D3：真实类为第1类，比较教学预测pA=[0.97,0.01,0.01,0.01]、pB=[0.925,0.025,0.025,0.025]的硬标签NLL、exp(NLL)、平滑交叉熵。两者argmax如何？

**结论**：A硬标签NLL/PPL较低，B平滑交叉熵较低，二者argmax相同。

**论文原文**：物理页8，P8:5–6。保留前后句如下：

```text
Label Smoothing During training, we employed label smoothing of value ϵls = 0 .1 [36]. This
hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.
```

**模型原答（原样）**：

```text
前提（来自 P8）：论文使用 ϵ_ls = 0.1（证据 P8 行5）。下文计算为教学推导，P8 未给出此数值算例或相应数值结果。

设真实类为第1类，候选预测分布为：
- pA = [0.97, 0.01, 0.01, 0.01]
- pB = [0.925, 0.025, 0.025, 0.025]
并采用上一问教学定义的平滑目标分布 q = (1-ϵ)onehot + ϵ/K，K=4, ϵ=0.1 ⇒ q = [0.925, 0.025, 0.025, 0.025]（推导）。

推导与数值计算：
1) 硬标签的 NLL（对真实类 j=1 的负对数似然）
- NLL_A = -ln pA_1 = -ln(0.97) ≈ 0.030459
- NLL_B = -ln pB_1 = -ln(0.925) ≈ 0.077961
2) exp(NLL)（单步 perplexity）
- exp(NLL_A) ≈ exp(0.030459) ≈ 1.03093
- exp(NLL_B) ≈ exp(0.077961) ≈ 1.08099
=> 在硬标签 NLL 与由其指数化的困惑度上，pA 表现更好（NLL 更低，perplexity 更低）。
3) 平滑交叉熵（使用 q 与 p 的交叉熵 H(q,p) = -∑_i q_i ln p_i）
- 对 pA：H_A = -[0.925·ln(0.97) + 3·0.025·ln(0.01)]
  = 0.0281806 + 0.3453878 ≈ 0.3735683
  （注：ln(0.97)≈-0.030459, ln(0.01)≈-4.605170）
- 对 pB：H_B = -[0.925·ln(0.925) + 3·0.025·ln(0.025)]
  = 0.0721119 + 0.2766659 ≈ 0.3487778
  （注：ln(0.925)≈-0.077961, ln(0.025)≈-3.688879）
=> 在平滑交叉熵指标下，pB 的值更小（更优），即标签平滑后 pB 优于 pA。

4) Argmax（预测类别）
- pA 的 argmax 为类1（概率 0.97）
- pB 的 argmax 也为类1（概率 0.925）
=> 两者的 argmax 相同，均预测真实类为第1类。

总结：在该算例下（采用常见的包含真实类的均匀平滑定义），pA 在硬标签 NLL 与困惑度上更好，但在使用平滑目标的交叉熵下 pB 更好；两者的 argmax 相同。因此这一算例演示了为何标签平滑可使困惑度变差但在平滑训练目标下可能降低训练损失或在序列级评价上产生提升（推导），但这些数值结果为教学推导，P8 未直接给出该数值算例或其实验验证（证据：P8 行5–6 仅给出定性陈述）。

模型 limitations字段：所有数值计算为教学推导并非 P8 中的实测数据；P8 未在给定段落中提供该算例或这些数值结果，因此无法从 P8 本身断言该具体算例的实验结论。
```

**作者推导**：计算用自然对数，硬NLL=-ln p_true，单token PPL=exp(NLL)，H(q,p)=-sum q_i ln p_i。A交叉熵0.3735625309，B交叉熵0.3487803849，B单tokenPPL=1.0810810811。教学未形成完整语料PPL、BLEU或任何训练实验。

**逐题核查**：原答A/B交叉熵写成约0.3735683/0.3487778，B单tokenPPL写成1.08099，有小数计算偏差。脚本纠正数值，不改变排序。


### D4：该算例能否证明BLEU必然提高？BLEU能否写成precision/recall或TP/FP/FN？论文regularization同时含哪些操作，因果外推有何边界？

**结论**：单token算例不证明BLEU提高；需区分训练损失、语料评价与因果。

**论文原文**：物理页8，P8:1–6。保留前后句如下：

```text
Residual Dropout We apply dropout [33] to the output of each sub-layer, before it is added to the
sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the
positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of
Pdrop = 0.1.
Label Smoothing During training, we employed label smoothing of value ϵls = 0 .1 [36]. This
hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.
```

**模型原答（原样）**：

```text
基于 P8 给出的句子（行5–6）：作者陈述“标签平滑……提高了准确率和 BLEU 得分”。这是一项经验性陈述，但 P8 中的该段落并未给出证据链、数学证明或将 BLEU 与单步概率或交叉熵直接等价化的公式。因此：
- 该算例不能证明 BLEU 必然提高：上题算例显示标签平滑能使基于平滑目标的训练损失减小（pB 平滑交叉熵更低），同时又可能使硬标签的困惑度变差且 argmax 不一定改变。BLEU 是一个序列级、基于 n-gram 精确度并含简短长度惩罚的度量，其数值取决于整体解码策略（例如 beam search、长度偏好）、模型在生成每个时间步上的概率分布对最终序列选择的影响，以及参考翻译的 n-gram 匹配。单个步的概率差异或交叉熵改善并不必然对应到 BLEU 的改进，因此不能从单步概率或此单例直接推出 BLEU 必然提高（此结论在 P8 中未被证明，只被报告为经验观察，见 P8 行5–6）。
- BLEU 不能简单写成传统二分类的 TP/FP/FN 或单一的 precision/recall：BLEU 主要是 n-gram 精确度的（在多个 n 下的几何均值）并含有 brevity penalty（长度惩罚），虽可以视为某种“precision”集合的组合，但它不是单一的 recall/precision 或单步 TP/FP/FN 的直接替代，因此不能用简单的二分类统计量来完全等价表示（此为教学推导说明，P8 未展开该定义）。
- 论文中同时包含的正则化操作：P8 在第1–4行说明了残差子层输出及 embedding+positional encoding 的 dropout（Pdrop = 0.1），并在第5行提到标签平滑（ϵ_ls = 0.1）。因此作者在训练中同时使用了 dropout 与标签平滑等正则化手段（证据见 P8 行1–4, 行5）。这意味着 BLEU 的提升可能来自多种因素的复合作用，而非单一由标签平滑直接导致（P8 在摘录中并未把改进的因果完全归于某一项）。

边界与限制（因果外推）：
- P8 的短段落仅给出经验性陈述（标签平滑与 BLEU/accuracy 的方向性关系），并未提供证明或将 BLEU 分解为单步概率项的等价变换；因此任何从本段对标签平滑得出“BLEU 必然提高”的强因果结论都超出了该证据的支持范围。
- 若要严谨证明标签平滑必然提高 BLEU，需要完整的序列级实验、解码设置及统计检验，以及对其他并行正则化（如 dropout、架构/优化差异）的控制，这些内容不在 P8 给定行内。

结论：P8 报告了标签平滑在其实验中提高 BLEU 的经验观察（证据：P8 行5–6），但从该段落和上述算例不能数学或逻辑上必然推导出 BLEU 在所有情形下必然提高；BLEU 也不能被简单地重写成传统 TP/FP/FN 形式，且论文同时使用了其它正则化手段，因果外推须谨慎。

模型 limitations字段：P8 的摘录仅为一句经验性陈述与对 dropout 的简短说明；关于 BLEU 改进的具体实验证据、解码设置、统计显著性与因果分离并未包含在该段落中。上述关于 BLEU 的不可等价性与因果边界为教学推导与一般性论述，而非 P8 在该段落中的证明。
```

**作者推导**：本片段含residual dropout、embedding+position dropout、label smoothing；base dropout=0.1。未在片段提供隔离所有因素的受控效应估计。BLEU定义属外部背景：候选n-gram截断precision几何平均乘brevity penalty，并非二分类TP/FP/FN或recall。

**逐题核查**：原答总体谨慎；“若要证明必然提高需实验统计”仍不够，有限实验最多支持限定分布上的经验效果，不能证明所有可能输入的必然性。外部定义非所给P8证据。

## 离线可复算算例

运行 `python offline_calc.py`，仅使用Python标准库，无网络、无GPU、不训练模型。以下为实际脚本输出；论文事实、题设教学输入与代数推导在各题中已区分。

```json
{
  "teaching_only": true,
  "target": [
    0.925,
    0.025,
    0.025,
    0.025
  ],
  "target_sum": 1.0,
  "wrong_only_target": [
    0.9,
    0.03333333333333333,
    0.03333333333333333,
    0.03333333333333333
  ],
  "A": {
    "hard_nll_nats": 0.030459207484708574,
    "single_token_ppl": 1.0309278350515465,
    "smooth_cross_entropy_nats": 0.3735625308724623,
    "argmax_1based": 1
  },
  "B": {
    "hard_nll_nats": 0.0779615414697118,
    "single_token_ppl": 1.081081081081081,
    "smooth_cross_entropy_nats": 0.34878038491802865,
    "argmax_1based": 1
  },
  "BLEU": "not computed; no translation sequence or reference"
}
```

源代码：[offline_calc.py](offline_calc.py)，保存输出：[offline-output.txt](offline-output.txt)。检查归一化、分母、比较方向的assert会在错误时中止；不是论文性能复现。

## 作者复核清单与局限

- 已按原页图核对公式、指数、转置、定义和上下文；页图见下方索引。文本抽取的断行公式已在调用前重建，保留原始冻结文件，事后不改模型输入或回答。
- 分母：逐题核对softmax归一化/平方根/宽度预算/交叉熵等对应分母；单位：参数个数、step、秒、nats、无量纲概率分开使用。比较方向已按各题算例核算。
- precision/recall、TP/FP/FN：本任务不是二分类检测，无混淆矩阵可报告；第75题的BLEU背景也不能替换成TP/FP/FN。未制造这些分数。
- 表头：本次关键证据为公式与正文，不从无表头行臆测数据；页图可能含其他表格，其数值未作为本题实测。
- 因果：单次调用、无对照、非盲、作者熟悉论文；不声称提示词带来提升，不用这批平台总分证明提示收益。
- 核查由Codex辅助执行逐项语义复核，不能冒称有独立人类审稿人或用户本人已人工签字。仍建议使用者在科研复用前复核关键推导。
- 所选片段可能不含完整定义；作者补充与题设教学假定明确标记。服务端seed/temperature和版本快照不可固定，重新调用可能给不同答案；离线脚本的确定性算术可直接复算。

## 复做步骤

1. 下载公开仓库；根README即完整正文，离线可读。按[SOURCES.md](SOURCES.md)获取同版论文，对照原页图和[numbered.txt](numbered.txt)。
2. `python validate.py sources.json output.json` 重放机械验收。第73题应保留失败字段；失败不是要求修写原答的理由。
3. `python offline_calc.py` 重放教学算术，对照保存输出。
4. 如需新真实调用，在已安装并登录Bohrium CLI的环境运行 `python reproduce_call.py --run-name my-new-run`；会产生新付费/额度调用，默认不会运行，已有目录禁止覆盖。结果是新的随机样本，不覆盖本次记录。
5. 参照[audit.md](audit.md)逐项复核原文支持性，区分论文值、教学推导和模型陈述。

## 材料与署名

原文证据：[sources.json](sources.json)、[numbered.txt](numbered.txt)。页图：[物理页8](page8.png)。完整输入输出、方法、审计与脚本均位于仓库根目录。

论文归Vaswani等；arXiv:1706.03762v7，NeurIPS 2017。本分享的原创解释与脚本MIT；论文片段版权归原作者，仅作为引用证据，不重新分发整篇PDF。外部指标定义如BLEU见[SOURCES.md](SOURCES.md)。
