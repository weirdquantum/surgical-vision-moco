# 子项目 1：手术器械分类

对腹腔镜图像中的 7 类器械做单图分类：Bipolar、Clipper、Grasper、Hook、Irrigator、Scissors、SpecimenBag。数据共 301 张，来自 10 段视频（v01–v10）。官方划分为 train 178 / val 33 / test 90 张。

## 相比原作业的修改

### Bug 与评估问题

| 问题 | 原实现 | 现在 |
|---|---|---|
| C0 未设随机种子 | 只有 C1–C5 调用了 `seed_task1`，C0 与其余配置的对比没有控制随机性 | 所有运行统一 `set_seed`，DataLoader 使用独立 `Generator` 和 `worker_init_fn` |
| 单次运行 | 每个配置只跑一次，33 张验证图上一张图就是 3 个百分点 | 默认 5 个种子（legacy 为 3 个），报告均值 ± 标准差 |
| checkpoint 按验证准确率选择 | 33 张图上准确率是离散、噪声很大的值，平局时取最早的 epoch | 改用验证集交叉熵（连续值）选择；EMA 权重进一步平滑 |
| 视频级泄漏 | 三个划分都包含 v01–v10 的帧，相邻帧（如 `v08_038625` / `v08_038725`）分别进入训练集和测试集 | 新增 `split: video_cv`：按视频做 5 折交叉验证（训练/验证/测试视频互不重叠），作为无泄漏的性能估计 |
| 增强破坏图像 | `RandomRotation(90)` 加垂直翻转，在 175×300 的长方形图上产生大面积黑角 | `RandomResizedCrop` 加 ±10° 旋转、水平翻转、颜色扰动和 RandomErasing |
| 宽高比 | 源图约 16:9，被缩放到 175×300 | 输入改为 224×384（≈1.71:1），保留更多细节 |
| 每个 batch 都返回 numpy 图像 | `__getitem__` 额外返回原始图像数组，被 collate 成大 tensor | 只返回 (tensor, label, path)；图像解码一次后缓存在内存 |

### 模型与训练

- **基线 CNN**：原网络 88% 的参数（2360 万）都在 `11520→2048` 的全连接层。改成全局平均池化加 Dropout 加线性层后，参数从 2668 万降到 98 万，且不再依赖输入尺寸（`convnet_scratch.yaml`）。原结构保留为 `convnet_legacy`，仅用于复现 C0。
- **预训练主干**：新增 ConvNeXt-Tiny 和 EfficientNetV2-S（torchvision ImageNet 权重）。
- **优化**：AdamW；主干学习率是分类头的 0.1 倍；norm 层和 bias 不做权重衰减；3 个 epoch warmup 后 cosine 退火；label smoothing 0.1；梯度裁剪；EMA；CUDA 上用混合精度；drop_last 避免出现只有 2 张图的 BN batch。
- **推理**：水平翻转 TTA；可以对多个模型的概率取平均做集成（`ensemble.py`，成员需事先确定，不能按测试结果挑选）。

## 配置

| 配置 | 说明 |
|---|---|
| `convnext_tiny.yaml` | **主模型**：ImageNet ConvNeXt-Tiny，改进协议 |
| `efficientnet_v2_s.yaml` / `resnet18_imagenet.yaml` | 其他预训练主干，协议相同 |
| `convnet_scratch.yaml` / `convnet_scratch_300ep.yaml` | 作业 CNN（全局平均池化版），从零训练 150 / 300 epoch |
| `convnext_tiny_video_cv.yaml` | 主模型，按视频做 5 折交叉验证（无泄漏估计） |
| `legacy_c0.yaml` … `legacy_c5.yaml` | 原作业 C0–C5 协议的忠实复现（固定种子、多种子），用作改进前后对比 |

## 运行

```bash
python -m instrument_classification.train --config instrument_classification/configs/convnext_tiny.yaml
python -m instrument_classification.train --config instrument_classification/configs/convnext_tiny.yaml \
    --set train.epochs=30 model.dropout=0.3 --seeds 0 1   # 覆盖任意配置项
python -m instrument_classification.ensemble runs/instrument_classification/convnext_tiny \
    runs/instrument_classification/efficientnet_v2_s --output runs/instrument_classification/ensemble
```

每个种子的输出位于 `runs/instrument_classification/<name>/seed<k>/`：`history.json`、`best.pt`、`metrics.json`（val/test 指标、每类 P/R/F1、混淆矩阵）、`predictions.npz`、学习曲线和混淆矩阵图。汇总在 `summary.json` 和 `summary.md`。

## 结果

在 Colab A100 上训练。官方划分的测试集有 90 张图，表中为多个种子的均值 ± 标准差。全部数字和同种子配对检验见 [`results/ALL_RESULTS.md`](../results/ALL_RESULTS.md)。

| 配置 | 种子 | Val Acc | Test Acc | Test macro-F1 |
|---|---|---|---|---|
| 原作业 C5（单次运行，旧结果） | 1 | 93.9% | 75.6% | 0.760 |
| `legacy_c0`：原基线 CNN 复现 | 3 | 66.7% | 54.8 ± 11.2% | 0.523 |
| `legacy_c5`：原最佳配置复现 | 3 | 91.9% | 84.8 ± 5.5% | 0.843 |
| `convnet_scratch`（150 epoch） | 5 | 84.2% | 81.8 ± 4.2% | 0.806 |
| `convnet_scratch_300ep` | 5 | 86.1% | 87.8 ± 2.2% | 0.868 |
| `resnet18_imagenet` | 5 | 100% | 88.9 ± 0.8% | 0.884 |
| **`convnext_tiny`** | 5 | 98.2% | **96.9 ± 1.4%** | **0.968** |
| `efficientnet_v2_s` | 5 | 98.8% | 97.3 ± 2.2% | 0.972 |
| **集成**（ConvNeXt + EfficientNet） | 5 | 98.8% | **97.6 ± 1.8%** | **0.974** |
| `convnext_tiny_video_cv`（按视频 5 折） | 5 折 | 84.1% | 83.5 ± 10.3%（合并后 **82.7%**） | 0.828（合并后） |

**各项改动的贡献**（官方测试集均值）：

| 步骤 | Test Acc | 配对差值（同种子） | 显著性 |
|---|---|---|---|
| 原协议复现（ResNet-18） | 84.8 ± 5.5% | — | — |
| + 新训练方案（同一个 ResNet-18） | 88.9 ± 0.8% | +3.7（3 个种子中 2 个更好） | p = 0.41，不显著；主要收益是稳定性（标准差 5.5 → 0.8） |
| + ConvNeXt-Tiny 主干 | 96.9 ± 1.4% | +8.0（5/5） | p < 0.001 |
| + 集成 | 97.6 ± 1.8% | +0.7（3/5） | p = 0.07，不显著 |

作业里的 CNN 改用全局平均池化并配合新训练方案后，从 54.8% 提升到 81.8%（+27），训练延长到 300 epoch 后达到 87.8%（+33），参数量从 2668 万降到 98 万。

### 原作业四组对比的多种子复现（C0–C5，各 3 个种子）

原作业用单次运行、按验证集准确率得出了四个结论。这里在原协议下用 3 个种子重跑，并做同种子配对比较（测试集）：

| 配置 | 内容 | Test Acc |
|---|---|---|
| C0 | 基线 CNN，SGD，batch 8，随机初始化 | 54.8 ± 11.2% |
| C1 | 手写 ResNet-18，SGD，batch 8 | 58.1 ± 6.1% |
| C2 | 手写 ResNet-18，Adam，batch 8 | 67.4 ± 3.2% |
| C3 | 手写 ResNet-18，Adam，batch 16 | 64.4 ± 5.9% |
| C4 | torchvision ResNet-18，随机初始化，Adam，batch 16 | 64.8 ± 5.3% |
| C5 | torchvision ResNet-18，ImageNet 初始化，Adam，batch 16 | 84.8 ± 5.5% |

| 因素 | 原作业结论（单次运行，验证集） | 多种子复现（测试集，配对） |
|---|---|---|
| 架构（C0 → C1） | +3.0 | +3.3，2/3 个种子更好，p = 0.64，**不显著** |
| 优化器（C1 → C2） | +15.2 | +9.3，3/3，p = 0.085，方向一致但证据有限 |
| 批大小（C2 → C3） | −3.0 | −3.0，0/3，p = 0.21，方向一致但不显著 |
| 初始化（C4 → C5） | +9.1 | **+20.0，3/3，p = 0.019**，唯一显著的因素 |

只有 ImageNet 初始化的作用在统计上站得住，而且比原作业估计的更大。另外，手写 ResNet-18（C3，64.4%）和 torchvision 版本（C4，64.8%）在相同设置下几乎一样，说明从零实现的网络是正确的。

**说明：**

- **视频级泄漏的影响：** 同一个 ConvNeXt 模型，测试视频从未在训练中出现时，准确率是 82.7%，比官方划分低约 14 个百分点。官方划分中相邻帧同时出现在训练集和测试集，分数偏高。各折之间差异也很大（68%–96%），说明不同手术视频之间的差异很大。
- **验证集已经饱和：** 33 张验证图上，多个模型达到 98%–100%，已经无法区分模型好坏，所以 checkpoint 改按验证损失选择。
- **作业 CNN 需要更长的训练：** 150 epoch 时，最佳 epoch 几乎都落在最后一个，说明还没收敛。延长到 300 epoch 后，测试准确率从 81.8% 提升到 87.8%，5 个种子全部提升（+3.3 到 +8.9 个百分点，p = 0.004），标准差从 4.2 降到 2.2。最佳 epoch 落在 222–300 之间，基本收敛。
