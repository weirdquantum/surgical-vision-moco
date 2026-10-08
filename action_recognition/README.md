# 子项目 2：手术动作识别（RARP 缝合）

对机器人辅助前列腺切除术（RARP）缝合视频中的 8 类动作做逐帧识别：Other、拾针、定位针尖、穿针、拔针、打结、剪线、放回/丢针。帧以 1 FPS 采样，共 12 段视频：train 8 / val 2 / test 2。

## 相比原作业的修改

### Bug 与评估问题

| 问题 | 说明与处理 |
|---|---|
| 验证集与训练集来自同一台手术 | val 的 `video_15_2` 和 train 的 `video_15_1` 是同一台手术的两段，验证分数偏乐观，checkpoint 选择也会因此有偏差。官方划分保持不变以便和原结果对比；新增 `split: surgery_cv`，按手术（而不是视频片段）做交叉验证 |
| macro-F1 计入不存在的类 | “打结”在 val 和 test 中都没有样本，原 macro-F1 把它计为 0，把上限压到 7/8。现在 `macro_f1` 只平均有样本的类别，`macro_f1_all` 保留原定义 |
| 宽高比失真 | 1920×1080 的帧被缩放到 256×512（2:1）。现在用 256×448，保持 16:9 |
| 每个 epoch 都解码全高清 JPEG | `prepare.py` 一次性把帧缩放到 288×512 并缓存，训练时 I/O 开销大幅下降 |
| 只训练单个种子 | 默认 3 个种子 |

### 方法改进

1. **更强的帧编码器**：ConvNeXt-Tiny（原为 ResNet-18），配 AdamW、分层学习率、cosine、EMA 和 label smoothing。
2. **类别不平衡**：最多的类有 521 帧，最少的只有 43 帧。改用按 √(1/频率) 加权的交叉熵，按验证集 macro-F1 选择 checkpoint。
3. **关闭水平翻转**：缝合动作中左右手分工明确，翻转会制造不真实的样本。
4. **时间信息**（原方法完全没有使用）：
   - **滑动平均平滑**：在验证集上选择窗口大小，再应用到测试集。几乎零成本。
   - **MS-TCN**（Farha & Gall, CVPR 2019）：3 个阶段 × 8 层膨胀卷积，感受野约 511 帧，覆盖整段视频。输入是逐帧模型的 log-概率序列，损失为交叉熵加 T-MSE 平滑项。
   - **交叉拟合（cross-fitting）**：逐帧模型在训练视频上接近 100% 准确，直接用它的输出训练 MS-TCN，会让时序模型学到“照抄输入”。因此按手术把训练集分成 4 折，每折用其余手术训练一个逐帧模型，对留出的视频做预测，得到真实的样本外输入。
5. **分段指标**：除逐帧 Accuracy/F1 外，还报告时序分割常用的 Edit score 和 F1@{10,25,50}，用来衡量预测是否碎片化。

## 配置

| 配置 | 说明 |
|---|---|
| `convnext_tiny_mstcn.yaml` | **主模型**：ConvNeXt-Tiny + 平滑 + 交叉拟合的 MS-TCN（一次运行同时报告 frame / smoothed / temporal 三种结果） |
| `convnext_tiny_frame_only.yaml` | 只训练逐帧模型加平滑，成本约为主模型的 1/3 |
| `efficientnet_v2_s_mstcn.yaml` | 换用 EfficientNetV2-S 主干 |
| `convnext_tiny_surgery_cv.yaml` | 按手术做 4 折交叉验证（无泄漏估计） |
| `legacy_resnet18.yaml` | 原作业 Task 2 协议的复现（多种子） |

## 运行

```bash
python -m action_recognition.prepare --data-root data      # 一次性，约 10 秒
python -m action_recognition.train --config action_recognition/configs/convnext_tiny_mstcn.yaml
```

已完成的阶段（逐帧模型、导出的输出、交叉拟合各折）会被跳过，中断后重新运行即可续跑。

## 结果

| 配置 | Test Acc | Test macro-F1（8 类） | Test macro-F1（有样本的类） |
|---|---|---|---|
| 原作业 ResNet-18（单次运行） | 69.8% | 0.437 | 0.500 |
| 改进后 | 待重新训练 | | |
