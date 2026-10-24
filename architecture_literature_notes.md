# 架构依据核验笔记

核验日期：2026-10-21。此处是针对架构选择的一手来源核验，不是系统综述。结构化元数据见 `prototypes/architecture_sources.json`；未查询引用数，未下载 PDF。下文“设计建议”均为本项目推论，须经训练和实测验证。

## 1. NAFBlock 适合作为首轮基本模块，但要保留实现细节

官方实现包含按空间位置对通道归一化的 LayerNorm2d、1×1 通道扩展、3×3 depthwise convolution、SimpleGate（通道二分相乘）、无 sigmoid 的简化通道注意力（全局平均池化后 1×1 convolution），以及两条零初始化 beta/gamma 的残差缩放。它不是纯线性模型，乘法仍引入非线性。首轮不必同时再加入 Transformer、频率分支或专家路由。[论文](https://arxiv.org/abs/2204.04676)；[官方模块代码](https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py)。

设计建议：独立实现一个紧凑 NAFBlock，复用现代 PyTorch 即可，避免为几层模块先安装整个旧 BasicSR 训练栈。若同时把“共享更新外部增益”和输出卷积都置零，会阻断部分初始梯度，应实测每条路径的梯度。

## 2. 全局统计的训练/测试尺寸差异必须单独检查

NAFNet 论文 §5.1 明确讨论 patch 训练与整图推理的差异，并采用 TLC。官方 `NAFNetLocal` 也是单独的变体，不应将其与普通全局池化版本混称为相同执行方式。[论文 §5.1](https://arxiv.org/html/2204.04676v4)；[官方 NAFNetLocal](https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py)。

设计建议：首轮固定全局池化版本并在 128/256/512 与原图上检查；若采用 TLC，应给所有预算使用相同转换规则，重新生成收益标签并计时。不能只在最终大预算出口转换。停止头的 pooled features 也要覆盖多种开发图像尺寸。

## 3. DURR 已经覆盖当前结果条件停止与长期回报

DURR 是 2018 预印本、ICLR 2019 正式论文。其共享残差 RNN 更新中，恢复单元读取当前恢复图像与原始退化观测，策略读取逐轮恢复输出。其 continue 奖励是相邻重建损失下降，Deep Q-learning 优化累计奖励；ICLR 版 §2.2 明确说明实验未使用正则项。[ICLR 正式版 §2.2](https://web.stanford.edu/~yplu/pub/durriclr.pdf)；[预印本与附录架构](https://arxiv.org/html/1805.07709v2)。

设计建议：不能把“监督预测长期收益”与 DURR 描述为一步收益和长期收益的根本区别。可以检验的变化是：缓存观测特征、保留跨轮潜在状态、用轻头避免策略重复编码整幅 RGB 图像，以及把实测新增延迟放进可调预算策略。它们能否形成方法贡献仍需强对照。

## 4. 跨轮特征状态传播也有明确先例

NLRN 使用共享权重递归，并在相邻递归状态传播深层特征相关性；其论文与代码已公开。因此，若加入隐藏状态，不能把“特征状态复用”本身写成首次贡献。[论文](https://arxiv.org/abs/1806.02919)；[官方代码链接，由论文给出](https://github.com/Ding-Liu/NLRN)。

设计建议：首轮只保留局部 NAF 更新，非局部相关性留作后续瓶颈驱动的选择。隐藏状态若保存多尺度特征，必须明确哪些特征一次缓存、哪些每轮更新、哪些需要重算，避免用“encoder once”掩盖全部重复计算。

## 5. AdaRevD 支持“强首出口＋便宜后续细化”的设计动机，但执行粒度不同

AdaRevD 复用预训练 encoder，构造多个可逆子 decoder，并按图像块的退化程度选择出口。它的 classifier 选择与本项目逐图、共享更新单元的执行方式不同。[论文 §3](https://arxiv.org/html/2406.09135v1)；[官方代码](https://github.com/DeepMed-Lab-ECNU/Single-Image-Deblur)。

设计建议：把一次性观测特征与可重复细化单元拆开是值得测的结构选择；若初始编码占总延迟过高，提前退出可省比例也被限制。对比时保留完整 decoder 重复执行的朴素共享版本，用延迟分解验证缓存是否真的节省。首轮整图停止与其 patch 结果不能直接进行跨任务数字比较。

## 6. 共享主干并不要求每轮行为完全相同，但阶段专属适配会限制预算外展开

LoRun（arXiv:2602.18697）共享预训练 denoiser，并为展开阶段使用轻量 LoRA 适配，其研究覆盖压缩感知、快照光谱重建及超分辨率。论文页面备注报告获 IEEE TIP 接收；此处未核验正式 DOI。[论文](https://arxiv.org/abs/2602.18697)；[方法 §III](https://arxiv.org/html/2602.18697v1)。

设计建议：它可作为后续“完全共享是否容量不足”的相关参照，但不能直接当盲 RGB 去噪的现成数值基线。首轮不要使用固定长度的每步 embedding 或每步独有 LoRA；它们在训练步数外没有自然定义。若需要阶段条件，优先读取当前状态或使用对任意步数有定义的标量函数，并仍需实测长展开可靠性。
