运行前标准：逐项回答D1-D4，保留论文条件与假设；数值与公式可离线复算；区分论文观测、教学算术与作者判断；引文范围存在不等于语义支持。检查分母、单位、方向、因果、表头。precision/recall与TP/FP/FN不适用时明确标记，不造指标；输出错误必须保留。单次无对照，不宣称提示增益。
逐字核对label smoothing的值及对perplexity、accuracy、BLEU的方向；是否存在逻辑矛盾？这三个指标是否同义？
论文给定页是否定义平滑分布？教学采用 q=(1-epsilon)onehot+epsilon/K，K=4、epsilon=0.1，写出分布和归一化总和；与只分给错误类的约定区别？
真实类为第1类，比较教学预测pA=[0.97,0.01,0.01,0.01]、pB=[0.925,0.025,0.025,0.025]的硬标签NLL、exp(NLL)、平滑交叉熵。两者argmax如何？
该算例能否证明BLEU必然提高？BLEU能否写成precision/recall或TP/FP/FN？论文regularization同时含哪些操作，因果外推有何边界？