# 盲 RGB 去噪数据入口

本轮默认数据是 **DIV2K 原训练 HR 800 张的工程划分**：640 张主干训练、80 张验证、80 张开发测试。它不替代项目文档建议的 BSD400/WED 训练与 CBSD68/Urban100 独立测试，也不把 DIV2K 原训练集的一部分称为官方 test。验证与开发测试只用于这轮框架检查。

## 默认来源及许可

默认 Hugging Face 仓库为 [dwchoi/div2k800-ffhq1000](https://huggingface.co/datasets/dwchoi/div2k800-ffhq1000/blob/main/README.md)，只读取 `DIV2K_train_HR/*.png`。仓库说明这部分为 800 张原始 HR PNG，未缩放、重编码或裁块；FFHQ、prompt 和 metadata 文件均不下载。镜像只有原 DIV2K train split，本项目再次按源图分成三个工程角色。

原始 [DIV2K 官方页](https://data.vision.ee.ethz.ch/cvl/DIV2K/) 将其限制为学术研究使用，图片版权归原作者。镜像 README 也说明原始许可继续适用。数据准备记录 dataset ID、请求 revision、解析后的 commit SHA、具体 PNG 路径及文件/像素 hash。后续所有下载都固定在这个 SHA。

采用 `huggingface_hub` 直接获取 PNG，不调用 `datasets.load_dataset`，不执行远程数据脚本，也不安装旧版 dataset builder。准备阶段会读取 HF 元数据；`build_datasets` 本身只访问已经准备好的本地文件。

## API 与配置

```python
from pathlib import Path
from data import prepare_data, build_datasets

config = {"data": {
    "source": "huggingface",
    "root": "data_cache",
    "hf_repo": "dwchoi/div2k800-ffhq1000",
    "hf_prefix": "DIV2K_train_HR",
    "revision": "main",
    "split_ratios": [0.8, 0.1, 0.1],
    "seed": 20261023,
    "train_patch": 128,
    "eval_crop": 256,
    "eval_mode": "center_crop",
    "train_sigmas": [15, 25, 50],
    "eval_sigmas": [15, 25, 50],
    "noise_protocol": "clip_uint8",
    "max_images": None,
    "hf_timeout": 20,
}}
manifest_path = prepare_data(config, Path.cwd())
datasets_by_role = build_datasets(config, manifest_path)
datasets_by_role["train"].set_epoch(0)
sample = datasets_by_role["train"][0]
```

`prepare_data` 的 `root` 是项目根目录；配置的 `data.root` 相对此目录解析。返回的 `manifest_<hash>.json` 是内容寻址文件，新清单不会覆盖旧清单。训练应保存并复用返回路径。`max_images=2` 在完整角色分割之后每角色选两张，因此只下载六张 PNG；该模式会写 `subset=true`，不能作为完整数据训练结果。它也支持 `{"train": 8, "val": 2, "test": 2}`。

```powershell
& 'H:\anaconda3\envs\py312\python.exe' -m data.prepare --config configs/denoise.yaml --max-images 2
```

JSON 配置不需要额外解析包，仓库默认 YAML 配置需 PyYAML。`hf_endpoint` 或环境变量 `HF_ENDPOINT` 可指定可访问的 Hub 端点；端点记录在 manifest 中。此模块不会自动切换第三方镜像。

## 离线图片文件夹

把 `source` 改成 `folder`，设置 `local_dir`，即可完全离线准备。支持三种布局：

- `train/`、`val/`、`test/`：保留显式角色。
- `train/`、`validation/`：train 保持训练角色；validation 按固定源图 hash 顺序分为约 40% val、60% test，可用 `val_images` 指定 val 张数。100 张时默认 40/60。
- 平铺图片或不含上述角色名的普通图片树：按 `split_ratios` 和固定源图 hash 分角色；至少需要三张不同图片。

支持 PNG/JPEG/BMP/TIFF/WebP；均按 EXIF 方向转正后转 RGB。记录文件 SHA256 及规范化 RGB 像素 SHA256；选中图片中相同像素的重复源图直接拒绝，避免重复文件造成角色泄漏。开启 `max_images` 后只检查选中的源图，不下载或检查未选中图片；该范围写入 manifest。该检查无法证明不存在视觉近重复，正式数据还需近重复审查。

`build_datasets` 每次启动都会重新计算清单中每个文件的 SHA256，并检查 image ID、source ID、像素 hash 唯一性和裁块尺寸。图片被替换或清单被编辑出重复记录会报错，不能继续沿用原清单假设 exact resume 成立。启动校验耗时不计入模型推理延迟。

## 可复现契约

训练裁块、翻转、90 度旋转、sigma 采样和噪声均由 `(seed, epoch, index)` 确定，不读取 Python/NumPy/torch 全局随机状态。相同 epoch/index 重取完全相同，修改 epoch 后重新采样。epoch 保存在共享 CPU tensor 中，`set_epoch` 可以传到 persistent workers；每轮应在取样前设置，不能在尚有该轮请求执行时修改。

验证和测试包含每张干净源图与全部 `eval_sigmas` 的笛卡尔积。固定噪声 seed 由 `(eval_seed, image_id, sigma)` 的稳定 SHA256 派生，与方法、访问顺序和 worker 数无关。训练与评估图片都返回 `noisy`、`clean` 两个 float32 RGB CHW tensor，另带 `image_id`、`source_id`、`sigma`、`role`、`noise_seed`、`sample_seed` 和 `noise_protocol`。

默认 `clip_uint8` 协议在 RGB 0–255 上加高斯噪声，clip 到 [0,255]，再截断转换 uint8，最后归一化 [0,1]；它包含截断和量化。`float_awgn` 显式选择另一种不截断不量化协议，结果不得混列比较。模型输入中没有 sigma 通道；返回 sigma 只是分组统计标签。

训练使用随机 patch，评估使用固定中心裁块；`eval_mode=full` 或 `eval_crop=0` 可保留原尺寸。图片小于请求裁块会报错，不暗中缩放或 padding。full 模式图像可能大小不同，评估 DataLoader 使用 batch=1。裁块质量与整图质量应分开报告。
