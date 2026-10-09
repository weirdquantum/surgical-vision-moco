# Surgical Vision：手术视频理解的三个子项目

基于 PyTorch 的手术场景视觉项目，由 NUS BN5211 课程作业重构扩展而来。三个子项目共享同一套数据、网络和训练代码：

| 子项目 | 任务 | 主要方法 |
|---|---|---|
| [1. 器械分类](instrument_classification/) | 腹腔镜图像 7 类器械分类 | ImageNet ConvNeXt / EfficientNet 微调、多种子评估、按视频交叉验证、模型集成 |
| [2. 动作识别](action_recognition/) | RARP 缝合视频 8 类动作逐帧识别 | 帧编码器 + 时间平滑 + 交叉拟合的 MS-TCN 时序模型，分段指标 |
| [3. 对比学习](contrastive_learning/) | 无标签预训练后迁移到器械分类 | 修复后的 MoCo v2（Shuffle-BN、MLP 头、强增强），kNN/线性探测，成对对照实验 |

## 仓库结构

```
.
├── common/                       # 三个项目共用
│   ├── utils.py                  # YAML 配置（_base_ 继承 + --set 覆盖）、随机种子、设备、日志
│   ├── data.py                   # 增强流水线、DataLoader、按组 k 折划分
│   ├── networks.py               # ConvNet / 手写 ResNet-18 / torchvision 主干 + 分类头
│   ├── engine.py                 # 训练循环：AdamW、分层学习率、warmup+cosine、EMA、AMP
│   ├── metrics.py                # Accuracy、macro-F1、混淆矩阵、多次运行汇总
│   └── plotting.py
├── instrument_classification/    # 子项目 1：data.py / train.py / ensemble.py / configs/
├── action_recognition/           # 子项目 2：data.py / prepare.py / temporal.py / metrics.py / train.py / configs/
├── contrastive_learning/         # 子项目 3：moco.py / pretrain.py / probe.py / configs/
├── tests/                        # pytest 单元测试（不需要数据集）
├── scripts/reproduce.sh          # 按依赖顺序复现 README 中的全部实验
├── scripts/collect_results.py    # 汇总所有结果并做同种子配对 t 检验
├── results/ALL_RESULTS.md        # 汇总结果（由上一个脚本生成）
├── colab/run_on_colab.ipynb      # 在 Colab GPU 上运行
└── legacy/                       # 原课程作业 notebook（已执行，含原始结果）
```

## 快速开始

```bash
pip install -r requirements.txt
ln -s /path/to/course/data data      # 包含 Dataset/、dataset-full/、RARP_1FPS/
pytest -q tests                      # 17 个单元测试
bash scripts/reproduce.sh            # 全部实验（A100 上约 7 小时）；也可以只跑 p1 / p2 / p3
```

所有入口都支持：`--config`、`--set key=value`（覆盖任意配置项）、`--seeds`、`--device`，以及 `--smoke`（2 个 epoch、每个 epoch 2 个 batch，用于检查流程能否跑通）。已经完成的运行会自动跳过，中断后可以续跑。设备会自动选择 CUDA、MPS 或 CPU。

## 数据

课程提供的数据**不包含在本仓库中**：

```
data/
├── Dataset/{train,val,test}/v01_007125_Gr.jpg   # 子项目 1：178 / 33 / 90 张
├── dataset-full/                                # 子项目 3：420 张无标签图像
└── RARP_1FPS/{train,val,test}/
    ├── images/video_xx/00060.jpg               # 子项目 2：1 FPS 帧
    └── actions/video_xx.txt                     # start_frame,end_frame,class_id
```

## 相比原作业的主要修复

1. **MoCo 的 key 编码器 BN 统计量从未更新**：key 一直用初始的均值 0、方差 1 做归一化，这很可能是原 InfoNCE 损失不下降的主要原因。改为 Shuffle-BN（单卡下用 SplitBatchNorm 模拟）。
2. **数据泄漏**：器械数据的相邻帧同时出现在 train 和 test 中；动作数据的 val 视频与一段 train 视频来自同一台手术。新增按视频/按手术的交叉验证。
3. **评估不可靠**：原先每个配置只跑一次、C0 没设随机种子、checkpoint 按 33 张图的准确率选择。现在固定种子、多种子报告均值 ± 标准差，按验证集损失或 macro-F1 选择。
4. **指标定义**：动作识别的 macro-F1 原先把测试集中不存在的“打结”类计为 0，现在同时报告两种口径。
5. **缺少对照实验**：对比学习的每组微调都配有只改变初始化的对照组。
6. **数据处理**：器械分类的增强会产生大面积黑角，动作识别的输入宽高比失真，RARP 每个 epoch 都要解码全高清 JPEG。

各子项目 README 中有完整的修改清单和原因。

## 主要结果

全部实验在 Colab（NVIDIA A100）上运行，多种子报告均值 ± 标准差。完整结果和同种子配对 t 检验见 [`results/ALL_RESULTS.md`](results/ALL_RESULTS.md)。

| 子项目 | 原作业（单次运行） | 原协议多种子复现 | 改进后 |
|---|---|---|---|
| 1. 器械分类（测试准确率） | 75.6% | 84.8 ± 5.5% | **96.9 ± 1.4%**（ConvNeXt-Tiny）；集成后 **97.6%**；按视频交叉验证 **82.7%** |
| 2. 动作识别（测试准确率 / macro-F1 / Edit） | 69.8% / 0.437* / – | 64.1% / 0.388* / 39.8 | **80.2% / 0.607* / 81.0**（ConvNeXt + MS-TCN）；按手术交叉验证 **66.1% / 0.507* / 77.3** |
| 3. MoCo → 器械分类（测试准确率） | 74.4% | – | **85.8 ± 0.9%**（修复后 MoCo，150 epoch）；训练到收敛后与随机初始化无显著差异（86.0% vs 87.8%） |

\* 8 类口径的 macro-F1（缺席的“打结”类计为 0）。只平均有样本的类时，改进后为 0.693。

几个关键发现：

- **官方划分高估了泛化能力：** 器械分类按视频交叉验证比官方划分低约 14 个百分点（82.7% vs 96.9%）；动作识别按手术交叉验证也低约 14 个百分点（66.1% vs 80.2%）。
- **时序建模是动作识别最大的提升来源：** MS-TCN 在官方划分上准确率 +8.6、Edit 从 45 升到 81（p = 0.003）；在没见过的手术上也稳定提升（+12.0，4/4 折，p = 0.004）。交叉拟合是关键设计：去掉后 macro-F1 下降 0.14（p = 0.025）。
- **原作业的结论大多经不起多种子检验：** 四个因素中，只有 ImageNet 初始化（+20.0，p = 0.019）显著；架构和批大小的差异都在噪声范围内。
- **MoCo：修 bug 让特征明显变好，但对完整微调帮助有限：** 修复后 kNN 准确率提升 19 个百分点，领域自适应预训练让 ImageNet 特征的 kNN 从 58.9% 升到 80.0%。但在 178 张标注图上训练到收敛后，与对照组没有显著差异；实际收益是收敛更快、方差更小。
