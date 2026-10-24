# 数据准备

统一入口：

```powershell
& 'H:\anaconda3\envs\py312\python.exe' run.py prepare
```

`prepare.py` 只做三件事：

1. 用 Hugging Face `snapshot_download` 下载 `DIV2K_train_HR/*.png`。
2. 排序后按固定 seed 打乱，以 80% / 10% / 10% 划分。
3. 保存到 `data_cache/splits.json`：

```json
{
  "train": [".../0001.png", ".../0002.png"],
  "val": [".../0003.png"],
  "test": [".../0004.png"]
}
```

默认数据为 [dwchoi/div2k800-ffhq1000](https://huggingface.co/datasets/dwchoi/div2k800-ffhq1000/blob/main/README.md) 的 DIV2K HR，不下载 FFHQ。800 张原训练图分成 640/80/80，是本项目的工程划分。DIV2K 原始学术研究许可继续适用。

配置见 `configs/denoise.yaml` 的 `data`，主要是下载目录、数据仓库、随机种子和划分比例。可用 `hf_endpoint` 或 `HF_ENDPOINT` 指定能访问的端点。

```powershell
# 先下载，再每个划分只使用4张；不减少下载量
& 'H:\anaconda3\envs\py312\python.exe' run.py prepare --max-images 4

# 已有图片时跳过下载
& 'H:\anaconda3\envs\py312\python.exe' run.py prepare --local-dir 'G:\path\to\images'
```

本地目录如果已有 `train/val/test`，直接保留其划分；否则收集 PNG/JPG/JPEG 后统一划分。相同文件列表、seed 和比例会得到相同划分。再次准备会更新 `splits.json`；不同实验需要不同划分时，可使用不同的 `data.root` 或保存一份列表，通过 `--manifest` 指定。

`build_datasets` 读取三组路径并构造 Dataset，同时兼容以前保存的 records 格式。不计算图片文件 hash、像素 hash 或生成审计元数据。

Dataset 继续负责 RGB 读取、训练随机裁块与增强、在线加噪；验证和测试用固定噪声。训练噪声由 `(seed, epoch, index)` 确定，评估噪声由图像路径和 sigma 确定。默认 sigma 为15/25/50，采用 clip 到 [0,255] 后转 uint8 的加噪协议。

训练 patch 默认128，评估中心裁块默认256；图片需要足够大。`eval_mode: full` 可评估原尺寸。

