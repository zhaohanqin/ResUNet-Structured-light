# ResUNet for Dual-View 3D Reconstruction

本项目是论文《Deep Learning-Driven One-Shot Dual-View 3-D Reconstruction for Dual-Projector System》中提出的 **ResUNet** 模型的 PyTorch 实现。该模型旨在从单张由双投影仪系统捕获的叠加光栅图像中，一次性恢复出两个视角的绝对相位图，从而实现快速、高精度的三维重建。

## 文件结构

```bash
.
├── ResUNet.py                                  # 核心脚本：包含模型定义、训练和评估逻辑
├── ResUNet_Architecture_Summary_Revised.md     # 网络结构详细文档
├── ResUNet_Operation_Guide.md                  # 项目运行、调试指南
└── README.md                                   # 本文件：项目概览和高级说明
```

## 网络架构详解

ResUNet 结合了 U-Net 的编码器-解码器结构、ResNet 的残差连接思想以及 Inception 的多尺度特征提取能力，使其能够高效地处理复杂的叠加光栅模式。

### 1. 核心构建块：`ResidualModule`

`ResidualModule` 是网络的基本单元，其设计旨在从输入特征中提取丰富且多样的信息。

- **结构**：包含5个并行的卷积分支。
  - **Branch 0**: 一个 `1x1` 卷积，作为主要的残差连接路径 (Shortcut)。
  - **Branch 1-4**: 分别由1到4个串联的 `3x3` 卷积组成，用于捕获不同尺度和复杂度的特征。
- **融合机制**：

  1. 分支1至4的输出在通道维度上被拼接（Concatenate）。
  2. 一个 `1x1` 卷积层用于融合这些拼接后的特征，并将通道数调整为期望的输出维度。
  3. 最后，融合后的结果与分支0的输出通过逐元素相加（Element-wise Addition）进行合并，形成残差连接。

这种设计使得网络既能捕捉多样的特征，又能通过残差连接确保梯度顺畅流动，避免深度网络中的梯度消失问题。

### 2. 整体架构：`ResUNet`

`ResUNet` 采用经典的 U 形对称结构，包含一个逐步压缩特征的编码器路径和一个逐步恢复空间分辨率的解码器路径。

#### 编码器（下采样路径）

编码器由5个 `ResidualModule` 块和4个 `MaxPool2d` 层组成，作用是提取图像从低级到高级的语义特征。

- **流程**：`ResidualModule` → `MaxPool2d` → `ResidualModule` → ...
- **通道变化**：随着网络加深，特征图的通道数逐层增加 (`1 -> 64 -> 128 -> 256 -> 512 -> 1024`)，而空间尺寸通过最大池化（`2x2`）减半。
- **瓶颈层**：编码器最深层（`enc5`）输出 `1024` 通道的特征图，作为网络的瓶颈，它包含了最高级的语义信息。

#### 解码器（上采样路径）

解码器负责将编码器提取的高级特征图逐步上采样，恢复图像的细节和空间分辨率，最终生成相位图。

- **流程**：`ConvTranspose2d` → `Concatenate` with Skip Connection → `ResidualModule`
- **通道变化**：通道数逐层减少 (`1024 -> 512 -> 256 -> 128 -> 64`)。
- **核心操作**：
  1. **上采样 (`ConvTranspose2d`)**: 将特征图的空间尺寸放大一倍，通道数减半。
  2. **跳跃连接 (`Skip Connection`)**: 将上采样后的特征图与编码器对应层的特征图在通道维度上进行拼接。**这是 U-Net 的精髓**，它将编码器中的低级空间细节（如边缘、纹理）直接传递给解码器，极大地帮助了细节的恢复。
  3. **特征处理 (`ResidualModule`)**: 使用残差模块处理拼接后的特征。

#### 网络参数表

以下表格详细描述了当输入为 `(1, H, W)` 时，网络各层的输出尺寸和通道变化：

| 路径     | 层级       | 操作                               | 输出通道 | 输出尺寸 (示例)          |
| :------- | :--------- | :--------------------------------- | :------- | :----------------------- |
| **输入** | -          | -                                  | 1        | `(H, W)`                 |
| **编码器** | `enc1`     | `ResidualModule`                   | 64       | `(H, W)`                 |
|          | `pool1`    | `MaxPool2d`                        | 64       | `(H/2, W/2)`             |
|          | `enc2`     | `ResidualModule`                   | 128      | `(H/2, W/2)`             |
|          | `pool2`    | `MaxPool2d`                        | 128      | `(H/4, W/4)`             |
|          | `enc3`     | `ResidualModule`                   | 256      | `(H/4, W/4)`             |
|          | `pool3`    | `MaxPool2d`                        | 256      | `(H/8, W/8)`             |
|          | `enc4`     | `ResidualModule`                   | 512      | `(H/8, W/8)`             |
|          | `pool4`    | `MaxPool2d`                        | 512      | `(H/16, W/16)`           |
| **瓶颈层** | `enc5`     | `ResidualModule`                   | 1024     | `(H/16, W/16)`           |
| **解码器** | `up4`      | `ConvTranspose2d`                  | 512      | `(H/8, W/8)`             |
|          | `dec4`     | `Concat(up4, enc4)` + `ResidualModule` | 512      | `(H/8, W/8)`             |
|          | `up3`      | `ConvTranspose2d`                  | 256      | `(H/4, W/4)`             |
|          | `dec3`     | `Concat(up3, enc3)` + `ResidualModule` | 256      | `(H/4, W/4)`             |
|          | `up2`      | `ConvTranspose2d`                  | 128      | `(H/2, W/2)`             |
|          | `dec2`     | `Concat(up2, enc2)` + `ResidualModule` | 128      | `(H/2, W/2)`             |
|          | `up1`      | `ConvTranspose2d`                  | 64       | `(H, W)`                 |
|          | `dec1`     | `Concat(up1, enc1)` + `ResidualModule` | 64       | `(H, W)`                 |
| **输出** | `out`      | `1x1 Conv`                         | 2        | `(H, W)`                 |

## 参数与超参数

理解模型参数和超参数的区别对于调整和优化模型至关重要。

### 可训练参数 (Trainable Parameters)

这些是模型在训练过程中通过反向传播自动学习和更新的权重。它们主要存在于：

- **`nn.Conv2d`** 和 **`nn.ConvTranspose2d`** 层中的卷积核权重和偏置。
- **`nn.BatchNorm2d`** 层中的仿射变换参数（`gamma` 和 `beta`）。

### 超参数 (Hyperparameters)

这些是需要在训练开始前由用户手动设置的参数，它们控制着训练过程的行为。在 `ResUNet.py` 的 `if __name__ == "__main__":` 部分可以找到主要超参数：

- **`dataset_path`**: **(需要修改)** 数据集存放的根目录路径。
- **`dataset_type`**: `'real'` 或 `'simulation'`，用于选择使用真实数据集还是模拟数据集。
- **`num_epochs`**: 训练的总轮数。决定了模型遍历整个训练集的次数。
- **`batch_size`**: 批量大小。即每次迭代输入模型的样本数量。**如果遇到 `CUDA out of memory` 错误，应首先减小此值**。
- **`lr` (Learning Rate)**: 学习率。控制模型权重更新的步长。

其他在 `train_and_evaluate` 函数中定义的重要超参数：

- **`weight_decay`**: 权重衰减系数（L2正则化），用于防止模型过拟合。
- **`T_max`**: 学习率调度器 `CosineAnnealingLR` 的参数，定义了学习率下降周期的长度。

## 运行指南

### 1. 环境配置

确保您已安装 Python 和以下库。推荐使用 `conda` 或 `venv` 创建虚拟环境。

```bash
pip install torch torchvision numpy scipy
```

- **Python**: `3.9` 或更高版本。
- **PyTorch**: `2.0.1` 或更高版本（建议使用 CUDA-enabled 版本以利用 GPU 加速）。

### 2. 数据集准备

1. 从 [DPSL3D-measurement GitHub](https://github.com/LiYiMingM/DPSL3D-measurement) 仓库下载数据集。
2. 解压数据集，并确保其目录结构如下：

    ```bash
    /path/to/your/data/
    └── DPSL3D-measurement/
        ├── double_projectors_real_dataset/
        │   ├── train/
        │   │   ├── inpout/
        │   │   └── gt/
        │   └── test/
        │       └── ...
        └── double_projectors_simulation_dataset/
            └── ...
    ```

3. `.mat` 文件中的变量名应为 `input`（输入光栅图）和 `phase`（真值相位图）。

### 3. 运行训练

1. **修改脚本**: 打开 `ResUNet.py` 文件。
2. **定位到 `if __name__ == "__main__":` 代码块**。
3. **修改 `dataset_path`**: 将路径 `/path/to/DPSL3D-measurement` 替换为您在第2步中存放数据集的 **上一级目录**。例如，如果您的数据集在 `E:/data/DPSL3D-measurement`，则应将 `dataset_path` 设置为 `E:/data`。
4. **运行脚本**:

    ```bash
    python ResUNet.py
    ```

5. 训练过程中的损失（Loss）、平均绝对误差（MAE）和均方根误差（RMSE）将会被打印在控制台。
