# 手术器械分类、手术动作识别与 MoCo 对比学习 (Surgical Vision with PyTorch)

NUS BN5211 Medical Robotic Intelligence 课程项目（2026 秋）。基于 PyTorch，在腹腔镜/机器人辅助手术图像上完成三项任务：

1. **手术器械分类**：自建 CNN 与从零实现的 ResNet18，对架构、优化器、批大小、初始化四个因素做受控对比实验
2. **手术动作识别**：ImageNet 预训练 ResNet18 对 RARP（机器人辅助前列腺切除术）缝合视频逐帧做 8 类动作分类
3. **对比学习**：从零实现 MoCo（动量编码器 + 负样本队列 + InfoNCE），无标签预训练后迁移到器械分类

全部代码、训练日志和图表都在已执行的 notebook [`notebooks/BN5211_Assignment2.ipynb`](notebooks/BN5211_Assignment2.ipynb) 中。

## 项目结构

```
.
├── notebooks/
│   └── BN5211_Assignment2.ipynb   # 完整实现 + 运行输出（Colab T4 GPU）
├── requirements.txt
└── README.md
```

Notebook 内部按任务划分：

| 部分 | 主要内容 |
|---|---|
| Preparation | 挂载 Google Drive、`ImageDataset`（读取文件名中的类别缩写，resize 到 175×300，翻转/旋转增强，ImageNet 归一化） |
| Task 1 | `ConvNet`（5 个 Conv-BN-ReLU-MaxPool 块 + 3 层 FC + Dropout）、`ResNet18Custom` / `ResidualBlock`、`train_net` / `eval_on_dataset`、C0–C5 六组实验、基于验证集的模型选择与一次性测试评估、混淆矩阵与分类报告 |
| Task 2 | `RARPReader`（按 `start,end,class` 区间标注为每帧打标签）、`RARPDataset`（拼接多段视频）、`RARPResnet`、`rarp_training` / `rarp_evaluate`（Accuracy、micro/macro/weighted F1）、StepLR 学习率调度 |
| Task 3 | `PairDataset`（同一图像两次独立增强构成正样本对）、`ConvNetNew`（编码器 + 256→128 投影头）、`ConLNet`（MoCo：encoder_q/encoder_k、动量更新、队列入队/出队、InfoNCE）、冻结特征诊断（余弦相似度分布、PCA、1-NN 检索）、加载预训练骨干微调 |

工程上的一些细节：

- 所有实验固定随机种子，DataLoader 使用独立的 `torch.Generator`，保证 C1–C5 只改变一个因素
- 模型选择只用验证集；选定后记录 checkpoint 的 SHA-256 并缓存测试结果，防止“看了测试集再换模型”
- 对预训练集与各划分做了图像哈希去重审计

## 数据集

数据由课程提供，**不包含在本仓库中**。在 Google Drive 中按如下结构放置（`drive_root = /content/drive/MyDrive/BN5211`）：

```
BN5211/
├── Dataset/            # Task 1：train 178 / val 33 / test 90 张
│   ├── train/  val/  test/   # 文件名形如 v01_007125_Gr.jpg，末尾为类别缩写
├── dataset-full/       # Task 3：420 张无标签图像，用于对比学习预训练
├── RARP_1FPS/          # Task 2：1 FPS 采样的 RARP 视频帧
│   └── {train,val,test}/
│       ├── images/video_xx/00000.jpg ...
│       └── actions/video_xx.txt     # 每行 start_frame,end_frame,class_id
└── weights/            # 训练时自动生成：checkpoint、历史记录、分析图表
```

- 器械类别（7 类）：Bipolar、Clipper、Grasper、Hook、Irrigator、Scissors、SpecimenBag
- 动作类别（8 类）：Other、拾针、定位针尖、穿针、拔针、打结、剪线、放回/丢针

## 实验结果

### Task 1：器械分类（验证集 33 张）

学习率 1e-3，200 epoch，每组保留验证准确率最高的 checkpoint。

| 配置 | 架构 | 优化器 | Batch | 初始化 | Val Acc |
|---|---|---|---|---|---|
| C0 | 自建 CNN | SGD | 8 | 随机 | 63.6% |
| C1 | ResNet18（自实现） | SGD | 8 | 随机 | 66.7% |
| C2 | ResNet18（自实现） | Adam | 8 | 随机 | 81.8% |
| C3 | ResNet18（自实现） | Adam | 16 | 随机 | 78.8% |
| C4 | ResNet18（torchvision） | Adam | 16 | 随机 | 84.8% |
| **C5** | ResNet18（torchvision） | Adam | 16 | **ImageNet** | **93.9%** |

最终选择 C5，**测试集准确率 75.6%**（macro-F1 0.760）。Irrigator、Bipolar 识别最好，Clipper 与 Hook 的 F1 最低（约 0.56–0.59）。验证集与测试集之间约 18 个百分点的落差说明 33 张验证图像给出的估计方差很大。

### Task 2：手术动作识别（测试集 454 帧）

ImageNet 预训练 ResNet18，Adam lr 5e-4，StepLR(8, 0.5)，20 epoch，按验证集 macro-F1 选 checkpoint。

| 指标 | 测试集 |
|---|---|
| Accuracy | 69.8% |
| Weighted F1 | 0.687 |
| Macro-F1（8 类） | 0.437 |
| Macro-F1（测试集中出现的 7 类） | 0.500 |

“Other”、“穿针”、“定位针尖”表现较好（F1 0.70–0.82）；“拾针”、“剪线”这类样本少、持续时间短的动作几乎识别不出。训练准确率接近 100% 而最后几个 epoch 的验证准确率在 52–61% 间波动，过拟合明显。“打结”在验证集和测试集中都没有样本。

### Task 3：MoCo 对比学习（τ=0.07, m=0.99, 队列 K=160）

| 阶段 | 结果 |
|---|---|
| 预训练（420 张无标签图像，200 epoch） | InfoNCE 损失在 3.3–4.5 之间波动，没有持续下降 |
| 冻结特征 1-NN 标签一致率 | 训练集 59.6%，验证集 66.7% |
| 迁移到 `ConvNet` 后全量微调 | 验证集 81.8%，**测试集 74.4%** |

对比参照：同架构随机初始化 + SGD 的 C0 验证准确率为 63.6%。但两者优化器不同，这一差距不能完全归因于对比预训练（见下文改进方向）。

## 运行

Notebook 在 Google Colab（T4 GPU）上运行。将数据放到 Drive 后按顺序执行即可；本地运行时把 `drive_root` 改为本地路径，并删除 `google.colab` 挂载那一段。

```bash
pip install -r requirements.txt
```

## 已知局限与改进方向

**数据与评估**

- **划分存在视频级泄漏**：Task 1 的 train/val/test 都包含 v01–v10 全部 10 段视频的帧，相邻帧（如 `v08_038625` 与 `v08_038725`）可能分别落入训练集和测试集，分数会偏乐观。应按视频划分，或做 leave-one-video-out 交叉验证。
- **验证集太小**：33 张图像，每张对应 3 个百分点，单次运行的排名不稳定。应做多种子重复（如 5 个种子）并报告均值 ± 标准差或置信区间。
- Task 2 中“打结”在验证集和测试集都缺失，“剪线”只有 5 帧。可以重新按视频分层划分，或在类别缺失时只报告有样本的类别。

**Task 1**

- 学习率固定为 1e-3，没有调度器和 weight decay，验证曲线震荡明显。可加入 cosine/OneCycle 调度、weight decay、早停，并为 SGD/Adam 分别搜索学习率，比较才更公平。
- `RandomRotation(90)` 加垂直翻转对 175×300 的非正方形图像会引入大块黑边，可换成 `RandomResizedCrop`、小角度旋转和颜色扰动。
- 自建 CNN 中 88% 的参数集中在第一层全连接层（11520→2048），可用全局平均池化替代，参数量能减少一个数量级。
- 小样本下可冻结部分骨干层或分层设置学习率，并尝试 label smoothing、MixUp/CutMix。

**Task 2**

- 目前是单帧分类，没有利用时间信息，而手术动作本身是时序过程。可以在帧特征上加 TCN / LSTM / Transformer（如 MS-TCN），或直接用视频模型（如 SlowFast、Video Swin）。最简单的改进是对预测结果做时间平滑（滑动窗口投票或 HMM）。
- 类别严重不平衡（最多 521 帧，最少 43 帧）：可以用类别加权的交叉熵、Focal Loss，或 `WeightedRandomSampler`。
- 256×512 的 resize 改变了原始宽高比，可改为保持比例的 resize 加 padding。

**Task 3**

- **缺少对照实验**：没有在相同的 Adam 微调设置下训练随机初始化的 `ConvNet`，因此无法量化对比预训练的实际增益。这是最该补的实验。另外可加入线性探测（冻结骨干只训练线性分类器），更直接地评估表征质量。
- 损失没有下降，说明预训练基本没有学到东西。可能原因：队列只有 160、batch 只有 8，负样本太少；增强太弱（缺少 MoCo v2 / SimCLR 中关键的 `RandomResizedCrop`、高斯模糊和随机灰度）；420 张图像相对 200 epoch 来说数据量太小。可按 MoCo v2 的配置改为 2 层 MLP 投影头，加大队列，并使用 cosine 学习率调度。
- 单 GPU 下未使用 Shuffle BN，BatchNorm 可能让模型“作弊”，可改用 GroupNorm 或 SyncBN 模拟。
- 预训练集包含全部 178 张训练图像（已做哈希核对，与 val/test 无完全重复），但同一视频的相邻帧仍可能接近测试图像。

**工程**

- 所有代码都在一个依赖 Colab 和 Google Drive 的 notebook 里。可以把数据集、模型、训练循环拆成 `src/` 下的模块，用 YAML/argparse 管理超参数，notebook 只负责分析和可视化。
- 引入 TensorBoard / Weights & Biases 记录实验，替代打印日志和 JSON 文件。
- 用 `num_workers`、`pin_memory` 和 AMP 混合精度加速训练（Task 1 的 DataLoader 目前是单进程）。
- 补充单元测试（如检查 `ResidualBlock` 输出形状、MoCo 队列指针是否循环正确）。
