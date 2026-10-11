# 逐页覆盖与去重记录

以下按 PPT 实际幻灯片顺序编号。连续动画页合并为完整推导；“删除”仅指封面、导航、章节分隔或无关装饰，没有以删除理由代替技术内容。章节链接使用固定锚点。

## 第1份：快速理解神经网络（65页）

| 页码 | 处理与笔记位置 |
|---|---|
| 1–3 | 封面、导航与 Introduction 分隔删除；课件身份记于 README |
| 4–9 | [历史与基础模型](notes/01-neural-networks.md#history)，保留应用、时间线、模型谱系及技术结果；品牌、新闻和纯装饰图片删除 |
| 10 | Deep Neural Networks 分隔删除 |
| 11–15 | [神经元与网络符号](notes/01-neural-networks.md#neuron)，合并神经元图的重复符号 |
| 16 | [前向传播通式及激活](notes/01-neural-networks.md#forward) |
| 17–20 | [完整前向例题](notes/01-neural-networks.md#forward)，合并逐步输入/隐藏/输出页并记录权重符号差异 |
| 21–22 | [目标函数、梯度下降、反向传播](notes/01-neural-networks.md#training) |
| 23 | CNN 分隔删除 |
| 24–25 | [CNN 结构与层职责](notes/01-neural-networks.md#cnn) |
| 26–47 | [两卷积核完整例题](notes/01-neural-networks.md#cnn)，保留输入、核、全部18个滑动输出和结果矩阵 |
| 48 | [kernel、padding、stride、channel 等术语](notes/01-neural-networks.md#conv-options)；只有问号的装饰图片删除，备注与实际图片不一致已注明 |
| 49 | [补零和输出尺寸](notes/01-neural-networks.md#conv-options) |
| 50–54 | [步幅为2的完整四步结果](notes/01-neural-networks.md#conv-options)，合并重复窗口 |
| 55–56 | [最大/平均池化](notes/01-neural-networks.md#conv-options)，保留降采样意义并修正分母 |
| 57 | RNN and LSTM 分隔删除 |
| 58–60 | [RNN 与长期依赖](notes/01-neural-networks.md#rnn)，保留展开图、例句与标准 RNN 对照 |
| 61–63 | [LSTM 结构、细胞状态与门控思想](notes/01-neural-networks.md#lstm)，删除传送带、闸机照片，保留对应概念和技术图 |
| 64–65 | [全部 LSTM 门与状态更新公式](notes/01-neural-networks.md#lstm) |

## 第2份：Self Attention（64页）

| 页码 | 处理与笔记位置 |
|---|---|
| 1 | 封面删除，课件身份记于 README |
| 2–4 | [注意力直觉、故事QKV、技术时间线](notes/02-self-attention.md#origin)，删除人物、品牌、论文封面，保留模型与方法名称及日期勘误 |
| 5–9 | [向量化输入](notes/02-self-attention.md#inputs)，保留词表、embedding、语音采样与特征维度、社交图和分子节点特征 |
| 10–12 | [三类输出任务](notes/02-self-attention.md#inputs)，合并重复示意，保留全部应用 |
| 13–17 | [上下文建模、堆叠层次与四位置输入输出](notes/02-self-attention.md#context)，移除纹理和角色装饰 |
| 18–23 | [QKV 与分数→softmax→汇聚的完整过程](notes/02-self-attention.md#qkv)，保留自匹配、位置1与2的计算及并行性质 |
| 24–28 | [矩阵化和可学习参数](notes/02-self-attention.md#matrix)，保留16项匹配，注明行/列约定和完整维度 |
| 29–31 | [两头例题及拼接输出](notes/02-self-attention.md#multihead)，合并动画步骤 |
| 32–33 | [位置编码及 NLP 应用](notes/02-self-attention.md#position)，保留热图和备注，删除角色图片 |
| 34–36 | [CNN 关系、ViT 图与少样本评估](notes/02-self-attention.md#comparison) |
| 37–38 | [RNN/LSTM 比较及论文](notes/02-self-attention.md#comparison)，删除 WIN 装饰 |
| 39–41 | [退化、残差、ResNet-34](notes/02-self-attention.md#residual) |
| 42–43 | [Sigmoid 公式、导数、优缺点与使用](notes/02-self-attention.md#activations)，合并重复页 |
| 44–45 | [Tanh](notes/02-self-attention.md#activations)，合并重复页 |
| 46–47 | [ReLU](notes/02-self-attention.md#activations)，合并重复页 |
| 48–49 | [Leaky ReLU 与0.01参数](notes/02-self-attention.md#activations)，合并重复页 |
| 50–51 | [ELU](notes/02-self-attention.md#activations)，合并重复页并补充负区间饱和条件 |
| 52–53 | [SELU、常数及自归一化条件](notes/02-self-attention.md#activations)，保留全部备注动机与限制 |
| 54 | 下一讲预告与 ChatGPT 品牌图删除 |
| 55–56 | [LayerNorm 与 Pre/Post-LN](notes/02-self-attention.md#stability)，完整公式和统计轴链接至第3章 |
| 57 | 重复残差图合并到[残差一节](notes/02-self-attention.md#residual) |
| 58 | Decoder-only 合并至[第3章结构及实验](notes/03-transformer.md#decoder-only) |
| 59–60 | 优化器与调度合并至[第3章完整说明](notes/03-transformer.md#optimization)，两份原动画均保留 |
| 61–63 | [Hard Sigmoid、Hard Tanh、ArcTan](notes/02-self-attention.md#activations)，公式、图、优缺点和场景全部保留 |
| 64 | [均值方差稳定的五类动机](notes/02-self-attention.md#stability)，去重后保留机制与适用条件 |

## 第3份：Transformer（51页）

| 页码 | 处理与笔记位置 |
|---|---|
| 1 | 封面删除，课件身份记于 README |
| 2–3 | [Seq2seq 的任务与整体结构](notes/03-transformer.md#seq2seq) |
| 4–5 | [Encoder 和 block](notes/03-transformer.md#encoder) |
| 6–7 | [残差与完整 LayerNorm 公式](notes/03-transformer.md#normalization)，重复页合并 |
| 8–10 | [LayerNorm 的统计轴、变长序列和数据并行](notes/03-transformer.md#normalization) |
| 11 | [Pre/Post-LN 及性能权衡](notes/03-transformer.md#normalization) |
| 12 | 残差技术保留于[Encoder](notes/03-transformer.md#encoder)并引用[第2章完整说明](notes/02-self-attention.md#residual) |
| 13–14 | [BERT](notes/03-transformer.md#encoder)、[归一化拓展论文及原图](notes/03-transformer.md#normalization)，删除 BERT 角色图片 |
| 15–20 | [AT Decoder、词表分布及逐步生成](notes/03-transformer.md#autoregressive)，合并重复整体框架与历史回填页 |
| 21–22 | [因果掩码、四位置矩阵和故事例子](notes/03-transformer.md#autoregressive) |
| 23–26 | [“气”错误累积、未知长度、EOS 与停止](notes/03-transformer.md#autoregressive) |
| 27–28 | [NAT、长度预测、并行性与多峰目标](notes/03-transformer.md#nat) |
| 29–32 | [cross-attention 的 QKV 来源、三源向量例子及 T5](notes/03-transformer.md#cross-attention) |
| 33–35 | [训练分布、teacher forcing 与暴露偏差](notes/03-transformer.md#teacher-forcing) |
| 36 | [下一 token 对齐表及原图](notes/03-transformer.md#teacher-forcing) |
| 37–40 | [交叉熵全部公式、词表、概率和0.86302例题](notes/03-transformer.md#loss)，重复计算步骤合并 |
| 41 | [优化器比较及原动画](notes/03-transformer.md#optimization) |
| 42–43 | [预热、衰减、余弦曲线及全部配置参数](notes/03-transformer.md#optimization) |
| 44–45 | [变长 batch、左右padding与mask](notes/03-transformer.md#padding) |
| 46–47 | [贪心两路径概率、beam width=2与原图](notes/03-transformer.md#beam) |
| 48 | [温度公式、全部经验范围、原图及重复问题](notes/03-transformer.md#temperature) |
| 49 | [Decoder-only、模型例子及数据路径](notes/03-transformer.md#decoder-only) |
| 50 | 总结构图合并至[第1节](notes/03-transformer.md#seq2seq) |
| 51 | [Lab 1 的全部10项实现提示](notes/03-transformer.md#decoder-only)，补充 GELU 定义 |

## 第4份：GPT-1、BERT、GPT-2 与 GPT-3（86页）

| 页码 | 处理与笔记位置 |
|---|---|
| 1 | 封面删除，课件身份记于 README 与章节来源 |
| 2–6 | [三种结构、Encoder/Decoder 职责与技术时间线](notes/04-gpt-and-bert.md#models)；重复谱系链接第1章，引用量和品牌装饰删除 |
| 7–14 | [GPT-1 的预训练与迁移动机](notes/04-gpt-and-bert.md#gpt1-pretrain)及[统一范式](notes/04-gpt-and-bert.md#models)，论文标题改为来源链接，合并摘要/引言展开页 |
| 15–17 | [L1目标、log动机、窗口、因果模型与embedding/block/softmax](notes/04-gpt-and-bert.md#gpt1-pretrain)，完整重写图中公式 |
| 18–21 | [分类目标、L2/L3、辅助语言建模、现代微调讨论](notes/04-gpt-and-bert.md#gpt1-finetune)，合并重复并澄清SFT与遗忘说法 |
| 22–26 | [分类、蕴涵、对称相似、多选的全部输入格式及局限](notes/04-gpt-and-bert.md#gpt1-finetune)，保留技术原图及候选评分 |
| 27–28 | [BooksCorpus、完整训练设置、参数与BERT对照](notes/04-gpt-and-bert.md#gpt1-experiments)，修正数据单位及Base/Large语料误解 |
| 29–30 | [GPT-1全部实验任务与原图](notes/04-gpt-and-bert.md#gpt1-experiments)，BERT对比连接[第5节](notes/04-gpt-and-bert.md#bert) |
| 31–35 | [BERT双向表示与ELMo/GPT对比](notes/04-gpt-and-bert.md#bert)，保留相关工作的实质区别，删除论文封面及装饰类比 |
| 36–37 | [MLM、NSP及统一预训练/微调框架](notes/04-gpt-and-bert.md#bert)，保留预训练原图，限定NSP结论 |
| 38 | [Base/Large全部配置与深宽复杂度](notes/04-gpt-and-bert.md#bert) |
| 39 | [三部分embedding、句对特殊token与CLS](notes/04-gpt-and-bert.md#bert)，保留输入原图 |
| 40 | [15%目标位置、80/10/10规则及完整hairy例子](notes/04-gpt-and-bert.md#bert) |
| 41 | [NSP 50/50、连续与随机句对例子](notes/04-gpt-and-bert.md#bert) |
| 42 | [BERT预训练数据与文档连续性](notes/04-gpt-and-bert.md#bert) |
| 43 | [分类头、CLS、GLUE全表与微调结构原图](notes/04-gpt-and-bert.md#bert) |
| 44 | [抽取式QA、起止概率与合法区间、损失、SQuAD及训练设置](notes/04-gpt-and-bert.md#bert)，保留技术原图 |
| 45–48 | [GPT-2因果路线、规模及zero-shot目标](notes/04-gpt-and-bert.md#gpt2)，论文封面改来源链接，限定模型比较 |
| 49–53 | [单任务/多任务/微调成本及自然语言任务提示](notes/04-gpt-and-bert.md#gpt2)，保留迁移与特殊token讨论 |
| 54–56 | [WebText、Reddit筛选、45M链接、8M文档与40GB](notes/04-gpt-and-bert.md#gpt2)，实验室资源宣传删除 |
| 57–59 | [四种GPT-2配置、全部语言建模表与四任务曲线](notes/04-gpt-and-bert.md#gpt2) |
| 60–68 | [GPT-3的175B规模、泛化/污染动机、ICL、meta-learning、三设置及汇总曲线](notes/04-gpt-and-bert.md#gpt3-icl)，合并重复并修正微调/zero-shot/稀疏权重说法 |
| 69–73 | [zero/one/few-shot全部法语例子、监督微调对照与上下文限制](notes/04-gpt-and-bert.md#gpt3-icl)，保留完整原图 |
| 74 | [GPT-3架构变化、八种模型全部超参数与表内维度差异](notes/04-gpt-and-bert.md#gpt3-training) |
| 75–76 | [质量过滤、LSH/MinHash、混合采样全表与V100分布式训练](notes/04-gpt-and-bert.md#gpt3-training)，硬件名录保留为举例 |
| 77 | [K采样、提示、候选似然、归一化、beam4/α0.6与评测指标](notes/04-gpt-and-bert.md#gpt3-evaluation) |
| 78 | [验证损失、幂律公式与LAMBADA全图](notes/04-gpt-and-bert.md#gpt3-evaluation) |
| 79–82 | [三问答数据集、TriviaQA、六翻译方向、新闻生成完整例子](notes/04-gpt-and-bert.md#gpt3-evaluation)，合并展开图，限定指标与人类水平 |
| 83–85 | [长文、因果表示、样本效率、成本、可解释性、偏见与误用](notes/04-gpt-and-bert.md#gpt3-limitations)，保留性别/群体偏见原始图 |
| 86 | [后续模型名录及到第5章的连接](notes/04-gpt-and-bert.md#gpt3-limitations)，不补造该页未给出的配置 |

## 第5份：RMSNorm、Swish 与 RoPE（27页）

| 页码 | 处理与笔记位置 |
|---|---|
| 1 | 封面删除，课件身份记于 README 与章节来源 |
| 2 | [Llama家族、图例及备注的领域/聊天/多模态分支](notes/05-llama-advancements.md#llama-sizes)，复用已有原图 |
| 3 | [8B/70B/405B全表、10/6.4倍估算及参数规律条件](notes/05-llama-advancements.md#llama-sizes)，保留原表 |
| 4–5 | [四项改动与整体结构](notes/05-llama-advancements.md#architecture)及[GQA/KV Cache补充](notes/05-llama-advancements.md#gqa-cache)，删除时间性排名 |
| 6 | [LayerNorm/RMSNorm等价条件、均值与归一化差异](notes/05-llama-advancements.md#rmsnorm) |
| 7–8 | [RMS、gain/bias、Llama无bias版本及居中/缩放动机](notes/05-llama-advancements.md#rmsnorm)，[效率备注及实现条件](notes/05-llama-advancements.md#rmsnorm-performance) |
| 9 | [7%–64%结果、五方法全表、24.7/25.9%重算与pRMSNorm](notes/05-llama-advancements.md#rmsnorm-performance) |
| 10 | [调音类比、均值/方差/形状对照](notes/05-llama-advancements.md#rmsnorm)与[pRMSNorm近似统计条件](notes/05-llama-advancements.md#rmsnorm-performance) |
| 11 | [Pre-Norm、残差及warm-up条件](notes/05-llama-advancements.md#architecture)，去重链接第3章 |
| 12–15 | [Swish/SiLU/GLU/SwiGLU全部公式、代码、三投影、β极限、GLUE全表及解释条件](notes/05-llama-advancements.md#swiglu)，删除太极装饰与幽默引语 |
| 16–18 | [绝对/相对位置、RoFormer来源、符号与点积目标](notes/05-llama-advancements.md#rope-goal)，论文封面改来源链接 |
| 19–20 | [Q/K点积与位置差](notes/05-llama-advancements.md#rope-goal)，[二维角度原图与0.5例题](notes/05-llama-advancements.md#rope-2d)，合并动画并澄清30°/60° |
| 21–22 | [复数、共轭、实部、相对位置符号与二维旋转推导](notes/05-llama-advancements.md#rope-2d)，保留两份原图与自内积条件 |
| 23 | [多维频率、10000/500000与上下文版本勘误](notes/05-llama-advancements.md#rope-general) |
| 24 | [分块正交矩阵与保范数条件](notes/05-llama-advancements.md#rope-general)，[避免稠密矩阵](notes/05-llama-advancements.md#rope-implementation) |
| 25–26 | [逐元素高效式、配对/广播、实现与原图](notes/05-llama-advancements.md#rope-implementation) |
| 27 | [二维/多维旋转与矩阵笔误](notes/05-llama-advancements.md#rope-2d)、[整体结构原图](notes/05-llama-advancements.md#architecture)，合并重复公式 |

## 核对方法

提取全部 slide XML、Office Math、讲者备注与内嵌媒体；逐份阅读文字与媒体联系表；按图形位置恢复卷积矩阵和解码树；使用本表覆盖293页。公式的重复图像与对应 Office Math 合并为同一公式，技术图中的数值以原图和计算结果交叉核对。原图保留在章节正文与图片清单中，来源页码可追溯。新增两份的归一化、Swish极限、RoPE旋转与原始模型配置另进行数值或来源核对。
