# 子项目 3：MoCo 对比学习与迁移

用 420 张无标签器械图像（`dataset-full/`）做 MoCo 自监督预训练，再迁移到子项目 1 的 7 类器械分类。

## 相比原作业的修改

### Bug

**Key 编码器的 BatchNorm 统计量从未更新（主要 bug）。** 原实现把 `encoder_k` 设为 eval 模式，但动量更新只更新了*参数*，没有更新 BN 的 running mean/var 缓冲区。因此 key 编码器在整个训练过程中都用初始统计量（均值 0、方差 1）做归一化，相当于在未归一化的激活上计算 key，与 query 的分布严重不一致。这很可能是原来 InfoNCE 损失在 3.3–4.5 之间波动、始终不下降的主要原因。

修复方式：key 编码器在 train 模式下用批统计量，并按 MoCo 原论文的 **Shuffle BN** 打乱 key 的 batch 顺序。单卡环境下用 `SplitBatchNorm` 把 batch 拆成 4 个子批分别计算统计量，模拟多卡效果，使 query 和它的正样本 key 不共享 BN 统计量，防止模型利用批内信息“作弊”。原行为保留为 `key_bn: frozen_eval`，只用于复现。单元测试会验证 key BN 的统计量确实在更新。

### 其他问题与改进（MoCo v2 配方）

| 项目 | 原实现 | 现在 | 原因 |
|---|---|---|---|
| 投影头 | 单层 256→128 | 2 层 MLP | MoCo v2 中带来的提升最大 |
| 温度 τ | 0.07 | 0.2 | MoCo v2 推荐值 |
| 增强 | 翻转、±15° 旋转、弱颜色扰动 | RandomResizedCrop(0.2–1)、强颜色扰动（p=0.8）、随机灰度、高斯模糊 | 对比学习的效果高度依赖裁剪和颜色增强 |
| batch / 队列 | 8 / 160 | 64 / 256 | 负样本更多；队列小于数据集（420），避免同一图像的旧 key 作为负样本 |
| 损失 | 单向 | 对称（两个视图互为 query/key） | 小数据下每张图的利用率翻倍 |
| 优化 | Adam 1e-3，恒定学习率，200 epoch | SGD 0.06，cosine，800 epoch | 标准 MoCo 设置；数据少，所以需要更多 epoch |
| 主干 | ConvNet | ConvNet / **ImageNet ResNet-18 + MoCo 继续预训练** | 领域自适应预训练通常优于从零开始 |
| 训练监控 | 只看损失 | 每 20 个 epoch 计算一次 kNN 准确率（labelled train→val），保存最佳 encoder | 损失下降不代表表征有用 |

### 评估问题

- **缺少对照实验**：原报告无法判断对比预训练是否有帮助，因为没有在相同微调设置下训练随机初始化的模型。现在每组微调配置只有初始化不同（scratch vs MoCo，ImageNet vs ImageNet+MoCo），其他超参数完全一致，并跑 5 个种子。
- **冻结表征评估**：新增 `probe.py`，对冻结的主干特征做加权 kNN（k=20）和线性探测（逻辑回归，正则系数 C 在验证集上选），并与随机初始化和 ImageNet 特征比较。

## 配置

| 预训练 | 微调（成对比较） |
|---|---|
| `legacy_moco_convnet.yaml`（原设置，含 BN bug） | `finetune_convnet_legacy_moco.yaml` |
| `moco_convnet.yaml` | `finetune_convnet_moco.yaml` vs `finetune_convnet_scratch.yaml`（150 epoch）；`finetune_convnet_moco_300ep.yaml` vs 子项目 1 的 `convnet_scratch_300ep.yaml` |
| `moco_resnet18_imagenet.yaml` | `finetune_resnet18_imagenet_moco.yaml` vs `finetune_resnet18_imagenet.yaml` |

## 运行

```bash
python -m contrastive_learning.pretrain --config contrastive_learning/configs/moco_resnet18_imagenet.yaml
python -m instrument_classification.train --config contrastive_learning/configs/finetune_resnet18_imagenet_moco.yaml
python -m contrastive_learning.probe --encoders imagenet:resnet18 \
    runs/contrastive_learning/moco_resnet18_imagenet/seed0/encoder_best.pt
```

微调使用子项目 1 的训练代码；预训练权重通过 `model.init_checkpoint` 加载到主干。

## 结果

在 Colab A100 上训练。测试集为子项目 1 官方划分的 90 张图。全部数字和同种子配对检验见 [`results/ALL_RESULTS.md`](../results/ALL_RESULTS.md)。

### 预训练过程

| 预训练 | InfoNCE 损失 | 正样本命中率 | 最佳 kNN 验证准确率 |
|---|---|---|---|
| 旧版（含 BN bug） | 3.97 → 2.91，在 2.85–4.98 间剧烈波动 | 2%–47% 间波动 | 51.5%（第 20 epoch），之后下降 |
| 修复后 ConvNet | 4.70 → 2.07，稳定下降 | 7% → 69% | 69.7% |
| ImageNet + MoCo ResNet-18 | 4.60 → 1.62 | 32% → 86% | 97.0% |

正样本命中率指在 257 个候选中选对正样本的比例。

### 冻结特征评估（`probe.py`，不做微调）

| 编码器 | kNN 测试 | 线性探测测试 |
|---|---|---|
| 随机初始化 ConvNet | 43.3% | 68.9% |
| 旧版 MoCo ConvNet（含 BN bug） | 43.3% | 65.6% |
| **修复后 MoCo ConvNet** | **62.2%** | **80.0%** |
| ImageNet ResNet-18 | 58.9% | 83.3% |
| **ImageNet + MoCo ResNet-18** | **80.0%** | **86.7%** |

“随机初始化 ConvNet”一行来自单次随机初始化，换一个初始化会有几个百分点的波动（`probe.py` 现已固定种子）；这不影响“旧版 MoCo 与随机初始化无异、修复后明显更好”的结论。

### 微调成对对照（5 个种子，均值 ± 标准差）

| 初始化 | Test Acc | Test macro-F1 |
|---|---|---|
| 原作业：ConvNet + MoCo（单次运行，旧结果） | 74.4% | – |
| ConvNet 随机初始化，150 epoch（对照） | 81.8 ± 4.2% | 0.806 |
| ConvNet + 旧版 MoCo，150 epoch | 83.8 ± 0.6% | 0.829 |
| ConvNet + 修复后 MoCo，150 epoch | 85.8 ± 0.9% | 0.846 |
| ConvNet 随机初始化，300 epoch（对照） | **87.8 ± 2.2%** | **0.868** |
| ConvNet + 修复后 MoCo，300 epoch | 86.0 ± 0.6% | 0.848 |
| ResNet-18 ImageNet（对照） | 89.1 ± 0.9% | 0.886 |
| ResNet-18 ImageNet + MoCo | 89.6 ± 1.3% | 0.885 |

**结论：**

- **BN bug 让原预训练基本无效：** 旧版 MoCo 的冻结特征和随机初始化一样差（kNN 都是 43.3%）。修复后，kNN 提升 19 个百分点，线性探测提升 11 个百分点。
- **用全部标签微调时，MoCo 没有提高最终准确率：**
  - 150 epoch 时，MoCo 初始化领先 +4.0 个百分点，但只在 3/5 个种子上更好（p = 0.125），不显著。
  - 两组都训练到收敛（300 epoch）后，随机初始化反而略高：−1.8，MoCo 只在 1/5 个种子上更好（p = 0.16），差异同样不显著。
- **MoCo 的实际收益是收敛更快、结果更稳：**
  - MoCo 初始化在 150 epoch 时已经达到 85.8%，延长到 300 epoch 几乎不变（86.0%），而随机初始化需要 300 epoch。
  - 种子间标准差为 0.6–0.9，随机初始化为 2.2–4.2。
- **领域自适应预训练明显改善特征空间：** ImageNet 特征的 kNN 准确率从 58.9% 提升到 80.0%。但完整微调后，与纯 ImageNet 初始化持平（+0.4，p = 0.18）。
- **整体判断：** 只有 420 张无标签图像时，MoCo 能学到明显更好的冻结特征（kNN、线性探测），但在有 178 张标注图可以完整微调的情况下，最终准确率没有显著提升。标签更少时它是否有帮助，本项目没有测试。
- **局限：** 测试集小（1 张图 ≈ 1.1 个百分点）。`encoder_best.pt` 按验证集 kNN 挑选，所以验证集指标偏乐观，表中只报告测试集。无标签预训练集包含 178 张训练图（未使用标签），不含验证集和测试集。
