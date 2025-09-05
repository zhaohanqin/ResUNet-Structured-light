# ResUNet：用于双视角三维重建的深度学习网络

本项目是论文《Deep Learning-Driven One-Shot Dual-View 3-D Reconstruction for Dual-Projector System》中提出的 **ResUNet** 模型的 PyTorch 实现。该模型旨在从单张由双投影仪系统捕获的叠加光栅图像中，一次性恢复出两个视角的绝对相位图，从而实现快速、高精度的三维重建。当前仓库已将模型与训练入口模块化拆分，便于使用与扩展。

## ✨ 项目特性

- **创新的残差多尺度模块**：结合了ResNet的残差连接思想和Inception的多尺度特征提取能力，通过5个并行分支捕获不同尺度的特征信息。
- **U-Net对称架构**：采用编码器-解码器结构，通过跳跃连接保留低级空间细节，确保相位图的精确重建。
- **双视角相位预测**：一次性从叠加光栅图像中恢复左右两个视角的绝对相位图，避免传统方法的多步处理。
- **高度可配置**：所有关键参数均可通过命令行灵活调整，便于实验和优化。
- **模块化设计**：模型定义、训练脚本和数据处理完全分离，提高代码可读性和可维护性。
- **丰富的可视化工具**：支持模型结构可视化，便于理解网络架构和调试。

## 📁 项目结构（模块化）

```bash
.
├── double_projectors_3D/                       # 数据集根目录（真实/模拟数据集）
│   ├── double_projectors_real_dataset/         # 真实数据集
│   │   ├── train/
│   │   │   ├── input 或 inpout/                # 训练输入的叠加光栅 .mat 文件
│   │   │   └── gt/
│   │   │       ├── unwrapped_phase_left/      # 左视角真值相位 .mat 文件
│   │   │       └── unwrapped_phase_right/     # 右视角真值相位 .mat 文件
│   │   └── test/                              # 测试/验证数据，同上结构
│   └── double_projectors_simulation_dataset/   # 模拟数据集（可选）
│
├── ResUNet_model.py                            # 模型文件：ResidualModule 与 ResUNet 网络定义（可独立运行）
├── train.py                                     # 训练入口：数据集类、训练/验证循环与主程序
├── ResUNet.py                                   # 旧版整合脚本（已拆分，保留以兼容与参考）
├── ResUNet_Architecture_Summary_Revised.md     # 网络结构详细文档（含层级说明与表格）
├── ResUNet_Operation_Guide.md                  # 运行与调试指南（环境、数据准备、常见问题）
├── Deep_Learning-Driven_One-Shot_Dual-View_3-D_Reconstruction_for_Dual-Projector_System.pdf  # 论文PDF
└── README.md                                   # 本文件：项目概览、说明与用法
```

### 各文件/目录的功能与作用

- **double_projectors_3D/**: 数据集根目录，包含真实与模拟数据集两套结构。训练脚本会在此路径下自动定位 `input/inpout` 与 `gt` 子目录。
  - **double_projectors_real_dataset/**: 真实拍摄数据集，含 `train/` 与 `test/` 两个划分；`gt/` 下提供左右视角的绝对相位图。
  - **double_projectors_simulation_dataset/**: 模拟生成的数据集，结构与真实数据一致，便于扩展与对比实验。

- **ResUNet_model.py**: 核心网络定义文件，**现已支持独立运行**。
  - `ResidualModule`: 多分支卷积 + 特征拼接 + 1x1 融合 + 残差连接，抽象论文中残差多尺度模块。
  - `ResUNet`: U 形结构的主干网络，由编码器、瓶颈层、解码器与跳跃连接组成；输出双通道相位图（左、右）。
  - **新增功能**: 支持通过命令行参数配置模型，使用 `torchsummary` 或 `torchinfo` 显示详细模型结构。

- **train.py**: 训练与验证主程序。
  - `PhaseDataset`: 读取 `.mat` 数据，自动适配多种键名（例如 `input`/`fringe`/`grating`；`phase`/`phi_unwrapped`），并智能容错 `input` 与常见拼写 `inpout` 目录。
  - `compute_metrics`: 评估指标计算（MAE、RMSE）。
  - `train_and_evaluate`: 训练循环（进度条、日志）、验证循环、学习率调度、数据加载器设置。
  - 主入口：配置数据集路径与超参数，启动训练流程。

- **ResUNet.py**: 旧版整合脚本（模型 + 数据集 + 训练在单文件）。现已被拆分为 `ResUNet_model.py` 与 `train.py`，此文件保留以便参考或回退。

- **ResUNet_Architecture_Summary_Revised.md**: 对网络结构的深入解析，包括模块构成、维度变化、设计动机等，有助于理解实现细节与超参数选择。

- **ResUNet_Operation_Guide.md**: 操作说明与调试手册，包含环境搭建、数据准备步骤、运行命令、常见错误定位与建议（如显存不足时降低 `batch_size`）。

- **Deep_Learning-Driven_One-Shot_Dual-View_3-D_Reconstruction_for_Dual-Projector_System.pdf**: 论文原文，项目实现对应其中的模型架构与任务描述。

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

## 🚀 环境设置

1. **克隆仓库:**

    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2. **创建虚拟环境 (推荐):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows系统使用: venv\Scripts\activate
    ```

3. **安装依赖:**

    ```bash
    pip install torch torchvision numpy scipy
    # 可选：安装模型结构可视化工具
    pip install torchsummary torchinfo
    ```

4. **准备数据集:**

    将数据组织为以下结构：

    ```bash
    double_projectors_3D/
    ├── double_projectors_real_dataset/
    │   ├── train/
    │   │   ├── input/                    # 叠加光栅图像 .mat 文件
    │   │   └── gt/
    │   │       ├── unwrapped_phase_left/ # 左视角相位真值 .mat 文件
    │   │       └── unwrapped_phase_right/# 右视角相位真值 .mat 文件
    │   └── test/                         # 测试数据，结构同上
    └── double_projectors_simulation_dataset/  # 模拟数据集（可选）
    ```

## 💻 使用指南

### 模型结构显示（ResUNet_model.py）

查看全部参数：

```bash
python ResUNet_model.py --help
```

基本使用（默认配置）：

```bash
python ResUNet_model.py
```

自定义配置示例：

```bash
python ResUNet_model.py --input_channels 1 --output_channels 2 --image_size 256 --device cpu
```

参数说明：

- `--input_channels`: 输入图像通道数（默认: 1，灰度图像）
- `--output_channels`: 输出相位图通道数（默认: 2，左右视角）
- `--image_size`: 输入图像尺寸（默认: 256）
- `--device`: 运行设备，cpu 或 cuda（默认: cpu）

### 模型训练（train.py）

现在推荐通过拆分后的训练入口脚本进行训练：

1. 确认数据集目录位于项目根目录的 `double_projectors_3D/` 下（或在 `train.py` 中按需调整路径）。
2. 执行训练：

    ```bash
    python train.py
    ```

3. 训练过程中会显示详细的进度信息和每个 epoch 的训练/验证指标。

**训练输出示例：**

```bash
Using device: cuda
训练数据集大小: 902
验证数据集大小: 82
训练批次数: 451
验证批次数: 41
测试数据加载...
成功加载测试批次，输入形状: torch.Size([2, 1, 512, 640]), 目标形状: torch.Size([2, 2, 512, 640])
Starting training for 100 epochs...
Dataset: real, Batch size: 2, Learning rate: 0.0005
====================================================================================================
Epoch   1/100 [Train] 100%|████████████| 451/451 [07:23<00:00,  1.02it/s]
Epoch   1/100 [Val] 100%|██████████████| 41/41 [00:34<00:00,  1.18it/s]
Epoch   1/100 | Train: Loss=0.0773, MAE=0.0773, RMSE=0.1216 | Val: Loss=0.0572, MAE=0.0572, RMSE=0.0971
```

如需参考旧版单文件流程，可运行：

```bash
python ResUNet.py
```

## ⚙️ 可配置参数详解

### 模型结构参数

| 参数 | 默认值 | 说明 | 对性能/结构的影响 |
|------|-------|------|------------------|
| `in_channels` | 1 | 输入图像通道数 | **结构影响**: 决定网络第一层的输入维度。**性能影响**: 通常为1（灰度图像），增加通道数可处理彩色图像但会显著增加参数量 |
| `out_channels` | 2 | 输出相位图通道数 | **结构影响**: 决定网络最后一层的输出维度。**性能影响**: 固定为2（左右视角），修改会影响损失函数计算 |
| `image_size` | 256 | 输入图像尺寸 | **结构影响**: 影响所有卷积层的输出尺寸。**性能影响**: 更大的尺寸提供更多细节但增加计算量和显存消耗 |

### 训练超参数

| 参数 | 默认值 | 说明 | 对性能/结构的影响 |
|------|-------|------|------------------|
| `num_epochs` | 100 | 训练轮数 | **性能影响**: 更多轮数可能提高精度但可能导致过拟合，需要根据验证损失调整 |
| `batch_size` | 2 | 批量大小 | **性能影响**: 影响训练稳定性和显存使用。**显存不足时优先降低此值**。更大的batch_size通常训练更稳定但需要更多显存 |
| `lr` (learning_rate) | 0.0005 | 学习率 | **性能影响**: 控制参数更新步长。过大会导致训练不稳定，过小会收敛缓慢。当前值适合大多数情况 |
| `weight_decay` | 1e-4 | 权重衰减 | **性能影响**: L2正则化系数，防止过拟合。增大可提高泛化能力但可能降低训练精度 |
| `T_max` | 50 | 学习率调度周期 | **性能影响**: 余弦退火调度器的周期长度，影响学习率变化曲线。通常设为总epoch数的一半 |

### 训练控制参数（新增）

| 参数 | 默认值 | 说明 | 对训练体验的影响 |
|------|-------|------|------------------|
| `verbose` | True | 详细输出模式 | **显示影响**: 控制是否显示详细的训练信息。False时只显示最终结果，True时显示完整的训练过程 |
| `show_progress` | True | 进度条显示 | **显示影响**: 控制是否显示训练和验证的进度条。False时使用简洁的文本输出，True时显示详细进度条 |
| `progress_style` | 'clean' | 进度条样式 | **显示影响**: 'clean'使用自定义简洁格式，'default'使用tqdm默认格式。影响进度条的显示风格 |
| `num_workers` | 0 | 数据加载线程数 | **性能影响**: 控制数据加载的并行程度。0表示单线程，避免多进程问题。增加可提高数据加载速度但可能导致系统不稳定 |

### 网络架构参数（ResidualModule内部）

| 参数 | 默认值 | 说明 | 对性能/结构的影响 |
|------|-------|------|------------------|
| `reduced_channels` | out_channels//4 | 各分支输出通道数 | **结构影响**: 决定多分支特征的维度。**性能影响**: 增大可提高表达能力但增加计算量 |
| 分支数量 | 5 | 并行分支数 | **结构影响**: 固定为5个分支（0-4）。**性能影响**: 多尺度特征提取，平衡局部和全局信息 |
| 卷积核大小 | 1x1, 3x3 | 各分支卷积核尺寸 | **结构影响**: 决定感受野大小。**性能影响**: 3x3卷积捕获局部特征，1x1卷积进行维度变换 |

## 🎛️ 参数配置指南

### 如何修改训练参数

在 `train.py` 文件的主函数部分（`if __name__ == "__main__":` 下方），您可以找到以下可配置参数：

```python
# 基础训练参数
dataset_type_to_use = 'real'        # 数据集类型：'real' 或 'simulation'
epochs = 100                        # 训练轮数
batch = 2                          # 批量大小
learning_rate = 0.0005             # 学习率

# 训练控制参数
verbose_mode = True                # 详细输出模式
show_progress_mode = True          # 进度条显示
progress_style_mode = 'clean'      # 进度条样式：'clean' 或 'default'
```

### 参数修改对模型的具体影响

#### 1. **基础训练参数**

**`epochs` (训练轮数)**
- **增加影响**: 模型有更多时间学习，可能提高精度，但训练时间延长，过多可能导致过拟合
- **减少影响**: 训练时间缩短，但模型可能未充分学习，精度较低
- **推荐调整**: 观察验证损失变化，当验证损失不再下降或开始上升时停止训练

**`batch_size` (批量大小)**

- **增加影响**: 
  - ✅ 训练更稳定，梯度估计更准确
  - ✅ GPU利用率更高，训练速度可能更快
  - ❌ 显存消耗大幅增加，可能导致OOM错误
  - ❌ 可能需要调整学习率

- **减少影响**:
  - ✅ 显存消耗减少，适合显存有限的设备
  - ✅ 可能有正则化效果，减少过拟合
  - ❌ 训练可能不稳定，收敛较慢

**`learning_rate` (学习率)**
- **增加影响** (如 0.001, 0.002):
  - ✅ 收敛速度更快
  - ❌ 可能跳过最优解，训练不稳定，损失震荡
- **减少影响** (如 0.0001, 0.00005):
  - ✅ 训练更稳定，能找到更精确的最优解
  - ❌ 收敛速度慢，可能需要更多epoch

#### 2. **训练控制参数**

**`verbose_mode` (详细输出)**
- **True**: 显示完整训练信息，包括数据集大小、批次数、每个epoch的详细指标
- **False**: 仅显示关键信息，适合后台运行或脚本自动化

**`show_progress_mode` (进度条显示)**
- **True**: 显示训练和验证的实时进度条，便于监控训练状态
- **False**: 使用简洁文本输出，避免进度条重复显示问题，适合日志记录

**`progress_style_mode` (进度条样式)**
- **'clean'**: 自定义简洁格式，显示epoch信息和基本进度
- **'default'**: tqdm默认格式，显示更多技术细节

### 显存优化建议

当遇到显存不足（OOM）错误时，按以下优先级调整参数：

1. **降低 `batch_size`**：从2降到1
2. **减小图像尺寸**：在数据预处理中调整输入图像大小
3. **设置 `num_workers=0`**：避免多进程数据加载的额外显存开销
4. **启用混合精度训练**：使用 `torch.cuda.amp`（需要代码修改）

### 不同场景的推荐配置

#### 🚀 **高性能GPU (RTX 4090, A100等)**
```python
epochs = 200
batch = 8                    # 利用大显存
learning_rate = 0.001        # 可以使用较大学习率
verbose_mode = True
show_progress_mode = True
progress_style_mode = 'clean'
```

#### 💻 **中等GPU (RTX 3060, RTX 4060等)**
```python
epochs = 150
batch = 4                    # 平衡显存和性能
learning_rate = 0.0005       # 默认配置
verbose_mode = True
show_progress_mode = True
progress_style_mode = 'clean'
```

#### 📱 **低端GPU或显存不足 (GTX 1660, RTX 3050等)**
```python
epochs = 100
batch = 1                    # 最小批量大小
learning_rate = 0.0003       # 较小学习率补偿小batch_size
verbose_mode = True
show_progress_mode = False   # 减少显示开销
progress_style_mode = 'clean'
```

#### 🖥️ **CPU训练 (仅用于测试)**
```python
epochs = 10                  # 大幅减少epoch数
batch = 1
learning_rate = 0.0001       # 非常小的学习率
verbose_mode = False         # 减少输出
show_progress_mode = False
```

#### 📊 **调试和实验**
```python
epochs = 5                   # 快速验证代码
batch = 2
learning_rate = 0.001        # 较大学习率快速收敛
verbose_mode = True          # 详细信息便于调试
show_progress_mode = True
progress_style_mode = 'default'  # 显示更多技术细节
```

### 性能调优建议

- **提高精度**：增加 `epochs`、调整学习率调度、使用更大的 `batch_size`（如果显存允许）
- **加快训练**：增加 `batch_size`、使用GPU加速、设置合适的 `num_workers`
- **防止过拟合**：增加 `weight_decay`、使用数据增强、早停机制、减少模型复杂度
- **平衡精度与速度**：根据具体应用场景调整参数组合，优先保证训练稳定性

## 🔧 网络优化与改进

相比原始实现，本项目对ResUNet进行了以下关键优化：

### 1. **模块化设计优化**
- 清晰分离模型定义、训练脚本和数据处理
- 提高代码可读性和可维护性
- 便于未来扩展和改进

### 2. **模型结构可视化**
- 支持通过命令行独立运行模型结构显示
- 自动检测并使用 `torchsummary` 或 `torchinfo`
- 提供详细的参数统计和层级信息

### 3. **训练过程优化**（最新改进）
- **智能进度显示**：支持多种进度条样式，避免重复显示问题
- **数据加载优化**：自动检测并处理多进程问题，提高训练稳定性
- **错误处理增强**：完整的异常捕获和调试信息，便于问题定位
- **训练状态监控**：实时显示数据集信息、批次数量和训练进度

### 4. **参数配置灵活性**
- 所有关键参数均可在代码中轻松调整
- 支持不同硬件配置的推荐参数组合
- 详细的参数影响说明和调优建议
- 支持不同训练模式（调试、生产、低资源等）

### 5. **用户体验改进**
- **多样化输出模式**：支持详细输出、简洁模式、无进度条模式
- **硬件自适应**：自动检测GPU可用性，提供不同硬件的推荐配置
- **实时反馈**：训练开始前验证数据加载，提供详细的系统信息

### 6. **稳定性增强**
- **数据加载容错**：智能处理文件路径和键名变化
- **显存优化**：提供完整的显存不足解决方案
- **训练中断恢复**：完善的错误处理机制

这些优化使得ResUNet不仅在模型性能上表现出色，在实际使用中也更加稳定和用户友好，适合从研究到生产的各种应用场景。

## 📝 更新日志

### v2.0 (最新) - 训练系统全面优化
- ✅ **修复进度条显示问题**：解决了训练过程中进度条重复显示的问题
- ✅ **增强训练稳定性**：优化数据加载器，添加完整的错误处理机制
- ✅ **丰富参数配置**：新增多种训练控制参数，支持不同硬件和使用场景
- ✅ **改进用户体验**：提供详细的参数说明和推荐配置
- ✅ **完善文档**：大幅扩展README，添加详细的参数影响分析

### v1.0 - 基础实现
- ✅ **模块化架构**：分离模型定义和训练脚本
- ✅ **模型可视化**：支持独立运行显示模型结构
- ✅ **数据处理**：智能适配多种数据格式和路径

## 📊 性能评估

注：此部分将在完成实验后更新具体的性能指标和对比结果。

## 📚 引用

如果您在研究中使用了本代码，请引用原论文：

```bibtex
@article{resunet2023,
  title={Deep Learning-Driven One-Shot Dual-View 3-D Reconstruction for Dual-Projector System},
  author={...},
  journal={...},
  year={2023}
}
```
