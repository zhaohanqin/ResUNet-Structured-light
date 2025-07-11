# ResUNet 操作文档

## 概述

`ResUNet.py` 是论文《Deep Learning-Driven One-Shot Dual-View 3-D Reconstruction for Dual-Projector System》中提出的 ResUNet 模型的 PyTorch 实现，用于双投影仪单相机结构光 3D 测量系统（DL-DPSL）。该模型通过单次拍摄的叠加相位移光栅图像，生成两个绝对相位图（对应左右投影视角），实现快速、精准的 3D 重建。本文档详细说明如何配置环境、准备数据集、运行训练和推理，以及解决常见问题。

## 功能

- **输入**：单通道叠加相位移光栅图像（分辨率通常为 640x512 或 256x256）。
- **输出**：两个单通道绝对相位图（\(\Phi_1\) 和 \(\Phi_2\)），用于 3D 点云重建。
- **性能**：
  - 训练后模型可在约 0.018 秒内完成相位恢复，适合实时工业应用。
  - 模拟数据上平均绝对误差（MAE）降低高达 20%，真实数据在连续区域表现更优。
  - 稳定性优于现有最快深度学习系统（DL_SPSL）。

## 环境配置

### 1. 硬件要求

- **推荐**：NVIDIA GPU（如 RTX 3090），支持 CUDA 加速。
- **最低**：CPU（训练速度较慢）。
- **存储**：数据集约为真实数据（1000 组，约 10GB）或模拟数据（3000 组，约 5GB），建议至少 20GB 可用磁盘空间。

### 2. 软件依赖

- **Python**：3.9 或更高版本（代码基于 Python 3.9.13 测试）。
- **PyTorch**：2.0.1（支持 CUDA，例如 `cu117`）。
- **其他库**：
  - `torchvision`：用于数据转换。
  - `numpy`：处理数据数组。
  - `scipy`：加载 `.mat` 文件。
  - `os` 和 `pathlib`：文件路径操作。
- **安装命令**：

  ```bash
  pip install torch==2.0.1 torchvision numpy scipy
  ```

- **验证安装**：

  ```python
  import torch
  import torchvision
  import numpy
  import scipy
  print(torch.__version__)  # 应输出 2.0.1 或兼容版本
  print(torch.cuda.is_available())  # 应输出 True（若有 GPU）
  ```

## 数据集准备

### 1. 数据集来源

- **来源**：数据集托管在 GitHub 仓库 [https://github.com/LiYiMingM/DPSL3D-measurement](https://github.com/LiYiMingM/DPSL3D-measurement)。
- **内容**：
  - **真实数据集**：约 1000 组，640x512 分辨率，包含叠加光栅图像和左右相位图。
  - **模拟数据集**：约 3000 组，256x256 分辨率，适合快速测试。
- **下载**：
  - 访问上述 GitHub 链接，下载数据集压缩包。
  - 解压到本地目录，例如 `/home/user/data/DPSL3D-measurement`。

### 2. 数据集结构

数据集需按以下结构组织：

```bash
/path/to/DPSL3D-measurement/
├── double_projectors_real_dataset/
│   ├── train/
│   │   ├── inpout/         # 叠加光栅图像 (.mat)
│   │   ├── gt/left/        # 左视角相位图 (.mat)
│   │   ├── gt/right/       # 右视角相位图 (.mat)
│   ├── test/
│   │   ├── inpout/
│   │   ├── gt/left/
│   │   ├── gt/right/
├── double_projectors_simulation_dataset/
│   ├── train/
│   │   ├── inpout/
│   │   ├── gt/left/
│   │   ├── gt/right/
│   ├── test/
│   │   ├── inpout/
│   │   ├── gt/left/
│   │   ├── gt/right/
```

- **文件格式**：`.mat` 文件，包含：

  - `inpout/*.mat`：变量名为 `input`，存储单通道叠加光栅图像（浮点型，`np.float32`）。
  - `gt/left/*.mat` 和 `gt/right/*.mat`：变量名为 `phase`，存储绝对相位图（浮点型，`np.float32`）。
- **验证数据集**：
  - 检查 `.mat` 文件内容：

    ```python
    import scipy.io as sio
    mat = sio.loadmat('/path/to/DPSL3D-measurement/double_projectors_real_dataset/train/inpout/sample.mat')
    print(mat.keys())  # 应包含 'input'
    mat = sio.loadmat('/path/to/DPSL3D-measurement/double_projectors_real_dataset/train/gt/left/sample.mat')
    print(mat.keys())  # 应包含 'phase'
    ```

### 3. 数据集路径配置

- 在 `ResUNet.py` 中，修改 `dataset_path` 为实际数据集根目录：

  ```python
  dataset_path = "/home/user/data/DPSL3D-measurement"  # 替换为实际路径
  ```

- **注意**：
  - 路径必须指向包含 `double_projectors_real_dataset` 和 `double_projectors_simulation_dataset` 的目录。
  - 确保文件夹名称准确（例如，`inpout` 而非 `input`）。
  - 如果变量名不同（例如 `fringe` 而非 `input`），修改 `PhaseDataset.__getitem__`：

    ```python
    input_data = sio.loadmat(input_path)['fringe'].astype(np.float32)  # 替换为实际变量名
    gt_left = sio.loadmat(gt_left_path)['phase_map'].astype(np.float32)
    gt_right = sio.loadmat(gt_right_path)['phase_map'].astype(np.float32)
    ```

### 4. 数据预处理

- **当前预处理**：
  - 代码使用 `transforms.ToTensor()` 将 NumPy 数组转换为 PyTorch 张量，自动归一化到 `[0, 1]`。
  - 假设 `.mat` 文件中的数据已是 `np.float32` 格式，值范围合理（例如 `[0, 1]` 或 `[0, 255]`）。
- **建议预处理**：
  - 如果数据范围不一致（例如负值或超出 `[0, 1]`），添加归一化：

    ```python
    # 在 PhaseDataset.__getitem__ 中添加
    input_data = (input_data - input_data.min()) / (input_data.max() - input_data.min())
    gt_left = (gt_left - gt_left.min()) / (gt_left.max() - gt_left.min())
    gt_right = (gt_right - gt_right.min()) / (gt_right.max() - gt_right.min())
    ```

  - 如果需要数据增强（例如随机噪声），可扩展 `transforms.Compose`：

    ```python
    self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2)
    ])
    ```

## 训练流程

### 1. 运行训练

- **代码入口**：

  ```python
  if __name__ == "__main__":
      dataset_path = "/home/user/data/DPSL3D-measurement"
      train_and_evaluate(dataset_path, dataset_type='real')  # 或 'simulation'
  ```

- **命令**：

  ```bash
  python ResUNet.py
  ```

- **参数说明**：
  - `dataset_type`：`'real'`（真实数据集，640x512）或 `'simulation'`（模拟数据集，256x256）。
  - 批量大小：8（`batch_size=8`）。
  - 学习率：0.0005（`lr=0.0005`）。
  - 权重衰减：1e-4（`weight_decay=1e-4`）。
  - 训练轮数：100（`num_epochs=100`）。
  - 优化器：Adam。
  - 损失函数：L1 损失。
  - 学习率调度：余弦退火（`CosineAnnealingLR`，`T_max=50`）。
- **输出**：
  - 每轮打印训练和验证的损失、MAE 和 RMSE：

    ```bash
    Epoch 1/100, Train Loss: 0.1234, Train MAE: 0.1000, Train RMSE: 0.1500, Val Loss: 0.1300, Val MAE: 0.1100, Val RMSE: 0.1600
    ```

### 2. 保存模型

- 当前代码未包含模型保存逻辑。建议在 `train_and_evaluate` 函数中添加：

  ```python
  best_val_loss = float('inf')
  for epoch in range(num_epochs):
      # ... 训练和验证代码 ...
      if val_loss / len(val_loader) < best_val_loss:
          best_val_loss = val_loss / len(val_loader)
          torch.save(model.state_dict(), 'best_resunet.pth')
          print(f"Saved best model at epoch {epoch+1} with Val Loss: {best_val_loss:.4f}")
  ```

- 保存路径：`best_resunet.pth`，可在训练后用于推理。

### 3. 推理

- 使用训练好的模型进行推理：

  ```python
  model = ResUNet().to(device)
  model.load_state_dict(torch.load('best_resunet.pth'))
  model.eval()
  with torch.no_grad():
      inputs = inputs.to(device)  # 形状 (batch, 1, H, W)
      outputs = model(inputs)     # 形状 (batch, 2, H, W)
  ```

- **输出格式**：`outputs` 包含两个通道，分别为 \(\Phi_1\) 和 \(\Phi_2\)，可用于后续 3D 点云重建。

## 常见问题及解决方案

### 1. 数据集路径错误

- **问题**：`FileNotFoundError` 或空文件列表。
- **解决方案**：
  - 确保 `dataset_path` 正确指向数据集根目录。
  - 检查文件夹名称（例如 `inpout` 而非 `input`）。
  - 验证 `.mat` 文件是否存在：

    ```bash
    ls /path/to/DPSL3D-measurement/double_projectors_real_dataset/train/inpout/*.mat
    ```

### 2. 变量名不匹配

- **问题**：`KeyError: 'input'` 或 `KeyError: 'phase'`。
- **解决方案**：
  - 检查 `.mat` 文件的变量名：

    ```python
    import scipy.io as sio
    mat = sio.loadmat('/path/to/inpout/sample.mat')
    print(mat.keys())
    ```

  - 修改 `PhaseDataset.__getitem__` 中的键名，例如：

    ```python
    input_data = sio.loadmat(input_path)['fringe'].astype(np.float32)
    ```

### 3. GPU 内存不足

- **问题**：真实数据集（640x512）批量大小为 8 可能导致 `CUDA out of memory`。
- **解决方案**：
  - 减小批量大小：

    ```python
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    ```

  - 使用模拟数据集（256x256，分辨率较低）。
  - 释放 GPU 内存：

    ```bash
    nvidia-smi
    kill -9 <PID>  # 终止占用 GPU 的进程
    ```

### 4. 数据范围问题

- **问题**：输入或相位图值范围不一致，导致训练不收敛。
- **解决方案**：
  - 添加归一化预处理（见“数据预处理”部分）。
  - 检查数据值范围：

    ```python
    print(input_data.min(), input_data.max())  # 应为 [0, 1] 或 [0, 255]
    print(gt_left.min(), gt_left.max())
    ```

### 5. 训练速度慢

- **问题**：在 CPU 上训练过慢或 GPU 利用率低。
- **解决方案**：
  - 确保使用 GPU（检查 `torch.cuda.is_available()`）。
  - 优化数据加载：

    ```python
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4, pin_memory=True)
    ```

  - 减少训练轮数（例如 `num_epochs=50`）进行测试。

### 6. 模型性能不佳

- **问题**：MAE 或 RMSE 高，未达到论文中的 20% 误差降低。
- **解决方案**：
  - 检查数据集是否正确（真实 vs. 模拟）。
  - 调整超参数：
    - 学习率：尝试 0.0001 或 0.001。
    - Dropout 比例：调整 `ResidualModule` 中的 `p=0.3` 为 0.2 或 0.4。
  - 增加训练轮数（例如 `num_epochs=200`）。
  - 使用预训练模型（若可用）：

    ```python
    model.load_state_dict(torch.load('pretrained_resunet.pth'))
    ```

## 结果评估

- **指标**：
  - 训练和验证阶段输出 L1 损失、MAE 和 RMSE。
  - 目标：MAE 降低 20%，与论文 Table II 一致。
- **可视化**：
  - 可视化预测相位图与真实相位图：

    ```python
    import matplotlib.pyplot as plt
    outputs = outputs.cpu().numpy()  # (batch, 2, H, W)
    targets = targets.cpu().numpy()  # (batch, 2, H, W)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(outputs[0, 0], cmap='gray')
    plt.title('Predicted Phase (Left)')
    plt.subplot(1, 2, 2)
    plt.imshow(targets[0, 0], cmap='gray')
    plt.title('Ground Truth Phase (Left)')
    plt.show()
    ```

## 扩展功能

- **日志记录**：
  - 添加 TensorBoard 支持：

    ```python
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('runs/resunet')
    for epoch in range(num_epochs):
        # ... 训练和验证代码 ...
        writer.add_scalar('Train/Loss', train_loss / len(train_loader), epoch)
        writer.add_scalar('Val/Loss', val_loss / len(val_loader), epoch)
        writer.add_scalar('Train/MAE', train_mae / len(train_loader), epoch)
        writer.add_scalar('Val/MAE', val_mae / len(val_loader), epoch)
    writer.close()
    ```

  - 查看日志：
  
    ```bash
    tensorboard --logdir runs/resunet
    ```

- **多分辨率支持**：
  - 代码支持 640x512 和 256x256 分辨率。若使用其他分辨率，需确保输入尺寸为 2 的倍数（因多次池化）。
- **模型优化**：
  - 减少通道数（例如 `enc5` 从 512 改为 256）以降低计算量：

    ```python
    self.enc5 = ResidualModule(512, 256)
    self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.dec4 = ResidualModule(256+256, 128)
    ```

## 结论

`ResUNet.py` 提供了一个完整的 ResUNet 模型实现，支持训练和推理。通过正确配置数据集路径、安装依赖和处理潜在问题，用户可以直接运行代码进行训练。建议下载 [DPSL3D-measurement 数据集](https://github.com/LiYiMingM/DPSL3D-measurement)，确保文件结构和变量名一致，并根据硬件条件调整批量大小和训练轮数。训练后的模型可用于快速、精准的 3D 重建，适用于工业场景。
