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

## 核对方法

提取全部 slide XML、Office Math、讲者备注与内嵌媒体；逐份阅读文字与媒体联系表；按图形位置恢复卷积矩阵和解码树；使用本表覆盖180页。公式的重复图像与对应 Office Math 合并为同一公式，技术图中的数值以原图和计算结果交叉核对。原图保留在章节正文与图片清单中，来源页码可追溯。
