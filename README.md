# 图像恢复初步验证框架

当前包含一个固定预算 NAF 风格 U-Net 基线和一个共享迭代原型，统一进行盲 RGB 高斯去噪训练。每个模型独立一个文件；训练、数据和评估入口共用。更新日期：2026-10-24。

## 目录

```text
models/
  blocks.py                 公共 NAF 风格基本块
  nafnet.py                 固定预算两级 U-Net 基线
  reusable_refiner.py        缓存观测特征、共享迭代原型
  __init__.py                build_model 配置工厂
data/
  prepare.py                 下载图片、划分、保存 splits.json
  denoising.py               RGB 裁块、增强、噪声、训练/评估 Dataset
configs/denoise.yaml         默认配置
train.py                     多出口 L1、AMP、验证与完整断点
evaluate.py                  逐图 PSNR/SSIM、逐出口曲线、可选实测延迟
run.py                       统一命令：prepare / train / eval / compare / smoke
common.py                    随机状态、指标、checkpoint、结果保存
tests/                       完全离线的正确性测试
prototypes/                  2026-10-21 架构探针，保留作历史记录
```

## 环境

已在 `H:\anaconda3\envs\py312\python.exe`、PyTorch 2.9.1+cu126、RTX 4060 Laptop 8GB 上验证。现有环境已具备全部运行依赖，无需重新安装 PyTorch，也不要求 transformers、torchvision 或 Hugging Face datasets。

以下 PowerShell 命令在项目目录执行：

```powershell
$py = 'H:\anaconda3\envs\py312\python.exe'

# 仅在缺依赖时安装；保留已经配置的 CUDA PyTorch
& $py -m pip install -r requirements.txt

# 本地小图、四次更新，含第二步保存后恢复到第四步；不联网
& $py run.py smoke --device cuda --name smoke_check

# 使用配置中的完整模型、128px patch、batch=4 做同样的检查
& $py run.py smoke --device cuda --name smoke_full_check --full-model

# 无 GPU 也能测试完整接口
& $py run.py smoke --device cpu --name smoke_cpu_check
```

`smoke` 生成的结构化合成图只用于检查软件链路，不能作为研究数据或恢复效果证据。运行目录已存在时训练会拒绝覆盖；换 `--name` 或显式续训。

## 初步数据集：DIV2K HR，通过 Hugging Face 获取

默认只使用 [dwchoi/div2k800-ffhq1000](https://huggingface.co/datasets/dwchoi/div2k800-ffhq1000/blob/main/README.md) 中的 `DIV2K_train_HR/*.png`，不下载 FFHQ、文本提示或 LR 图。通过 `snapshot_download` 下载 HR 图片，然后用固定随机种子划分。默认原训练 HR 800 张固定划为 **640 训练 / 80 验证 / 80 开发测试**。完整 HR 约 3.3 GB；`max_images` 只限制划分后的使用数量，不减少下载量。

这是当前的工程验证协议，与已有文档里的 BSD400+WED / CBSD68+Urban100 正式研究协议分开。DIV2K 原始许可为学术研究使用，图片版权仍归原作者；具体来源和数据契约见 [data/README.md](data/README.md)。

```powershell
# 下载HR图片后，每个划分使用4张
# 两个模型顺序训练20次更新，再分别评估；这是链路检查，不是收敛实验
& $py run.py compare --name div2k_quick --max-images 4 --max-steps 20

# 单独准备数据，不训练
& $py run.py prepare --max-images 4

# 完整数据、默认20k次更新，先单独训练一个模型
& $py run.py train --model nafnet --name div2k_baseline
& $py run.py train --model reusable_refiner --name div2k_prototype
```

2026-10-23 本机验证时，直连 Hugging Face 超时，测试的镜像返回 403，尚未下载真实 DIV2K 图片。HF 准备逻辑已通过离线 API 模拟测试，CUDA 训练链路使用本地合成图验证。网络可用后运行上述命令即可；也可指定你能访问的端点：

```powershell
& $py run.py prepare --max-images 4 --endpoint 'https://YOUR_ACCESSIBLE_HF_ENDPOINT'
```

端点也可通过 `data.hf_endpoint` 或环境变量 `HF_ENDPOINT` 配置，代码不自动切换镜像。图片和划分列表保存在项目内 `data_cache/`。

本地已有图片时：

```powershell
& $py run.py compare --name local_quick --local-dir 'G:\path\to\clean_images' --max-steps 20
```

本地目录可使用已分好的 `train/val/test` 三个子目录，或直接提供待划分的图片目录。默认训练 patch=128、验证中心裁块=256，图片必须足够大；更小图像应在 YAML 中明确减小裁块。`max_images` 在划分之后限额，不改变同一图片的角色。所有方法共用相同测试噪声；准备阶段只保存路径列表，不扫描文件或像素 hash。

## 训练与续训

训练默认 AdamW、L1 重建、四步均匀深监督（原型）、FP16 AMP、梯度裁剪。收益头默认冻结，本轮不训练停止策略。`--max-steps` 表示优化器更新总数，梯度累积微批次数由 `train.grad_accumulation` 设置。默认 `num_workers=0` 适配 Windows；建议先测瓶颈再增加。

AMP 出现非有限梯度时，由 GradScaler 跳过该批次的优化器更新并降低 loss scale，日志记录 `amp_overflow`。跳过不增加成功更新数，不推进学习率；下一批次继续训练。连续 16 次溢出仍报错，避免掩盖持续发散。非有限 loss 或关闭 AMP 时的非有限梯度仍直接报错。梯度范数采用 FP64 规约，避免范数求和本身在 FP32 下溢出。正常 AMP 日志包含当前 scale 和累计跳过数。

```powershell
# 总计划2000步，只运行到第100步后退出，不改变2000步学习率计划
& $py run.py train --model reusable_refiner --name resumable --max-images 4 --max-steps 2000 --stop-after 100

# 自动复用checkpoint内的配置、manifest和输出目录，继续原计划
& $py run.py train --resume outputs/resumable/reusable_refiner/last.pt
```

`last.pt` 保存模型、优化器、调度器、GradScaler、已完成更新数、最佳验证分数、Python/NumPy/torch/CUDA 随机状态，以及下一训练批次的 epoch/位置。`best.pt` 只在固定验证间隔或真正计划终点参与选择；在首次验证之前中断时尚无 `best.pt`。恢复时要求模型、数据、种子、验证选择协议和训练计划相同；不能通过修改 `max_steps` 假装无缝延长原计划。

续训仅支持原目录的 `last.pt`，避免把其他实验或旧最佳分数混入新目录。若程序崩溃留下了晚于 checkpoint 的日志，恢复时会备份旧日志并截回已保存的更新数，再开始重放。

源图和裁块/噪声通过固定 seed 与 epoch/index 生成，DataLoader 使用独立随机数生成器。CPU 确定性测试已逐项确认连续训练与中断恢复状态完全一致；CUDA smoke 已验证可恢复执行，默认非确定性 CUDA 配置不保证跨硬件或版本的逐位一致。

## 评估与结果

```powershell
& $py run.py eval --checkpoint outputs/resumable/reusable_refiner/best.pt

# 实际执行不同固定步数，计入完整模型调度时间
& $py run.py eval --checkpoint outputs/resumable/reusable_refiner/best.pt --exits 1 2 3 4 6 8 --benchmark

# 需要修改噪声等级、整图模式等时，复制配置并通过 --config 指定
& $py run.py eval --config configs/denoise.yaml --checkpoint outputs/resumable/reusable_refiner/best.pt --exits 1 4
```

默认不计时，快速生成质量结果；开启 `--benchmark` 后，每个出口单独运行真实固定预算路径，GPU 同步加 CPU wall-clock。输入已在 GPU、输出留在 GPU，不含数据读取、传输、指标与文件保存。默认 5 次预热、3 次重复仅用于工程检查；正式曲线需要增加重复、控制方法顺序和热状态。

默认基线评估一步，原型评估 1/2/3/4/6/8 步。6/8 步标记为超过训练长度的裸轨迹诊断。质量按 RGB、[0,1]、输出 clip 计算逐图 PSNR 和 Gaussian 11×11/σ1.5 valid-window SSIM，再对图像等权平均。模型训练使用未 clip 输出；无 sigma 或真实任务标签作为网络输入。

结果目录：

```text
outputs/<name>/<model>/
  run.json                   配置、环境、参数量、manifest路径及hash
  train.jsonl                每步loss/lr/梯度/数据位置与验证记录
  last.pt / best.pt          完整断点
  training_summary.json      完成步数及计划是否结束
  test/
    per_image.csv            图像ID、噪声seed、σ、出口、PSNR/SSIM、可选延迟
    summary.json             按σ和出口汇总，协议与环境
outputs/<name>/comparison.json  compare命令的两模型汇总
```

固定基线和原型目前只匹配数据、训练更新数与设置，未按实测延迟匹配容量；初步比较不能代替论文里的强固定预算对照。学习停止、收益标签训练、固定混合包络和参考辅助分配留作下一阶段。

## 添加模型

新增 `models/your_model.py`，在 `models/__init__.py` 工厂注册，遵循：

```python
default_steps: int
supports_trajectory: bool
forward(y, steps=None) -> Tensor
forward_fixed(y, steps=None) -> Tensor
forward_trajectory(y, steps=None) -> list[Tensor]
```

固定模型只返回一个出口。默认 NAF 风格基线约 227,979 参数；共享原型总计 262,967 参数，其中收益头 6,595 参数冻结。它们是紧凑研究实现，不是官方预训练模型的数值复现。共享原型可严格读取此前 `prototypes/reusable_refiner.py` 的同配置 state_dict。

## 测试

```powershell
& $py -m unittest discover -s tests -v
```

覆盖共享更新梯度、真实提前退出、数据划分、跨方法噪声一致、HF下载入口，以及完整续训状态。测试不访问网络，不下载研究数据。

2026-10-23 验证记录：21 项 CPU 测试通过；两种默认 C24 模型在 CUDA 上完成 batch=4/patch=128 的四次更新、第二步恢复续训、验证、逐出口质量与实际固定出口计时；另单独验证了 CLI 的 train/resume/eval/compare。工程结果在 `outputs/smoke_default_cuda_20261023/` 和 `outputs/compare_cli_final_20261023/`。这些短程合成图结果仅证明软件链路可用。
