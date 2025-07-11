import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import scipy.io as sio
import os
from pathlib import Path

# ----------------------------------
# 1. 数据集加载 (Dataset Loading)
# ----------------------------------
# 定义一个自定义的 PyTorch 数据集类，用于加载相位数据。
# PyTorch 中的 Dataset 类是所有数据集的抽象基类，我们需要实现 __len__ 和 __getitem__ 方法。
class PhaseDataset(Dataset):
    """
    用于加载叠加光栅图像和对应真值相位图的数据集。
    - 从 .mat 文件中读取数据。
    - 输入是单通道的叠加光栅图。
    - 输出是双通道的真值相位图（左、右视角）。
    """
    def __init__(self, dataset_path, dataset_type='real', split='train'):
        """
        初始化数据集。
        Args:
            dataset_path (str): 数据集根目录的路径。
            dataset_type (str): 'real' (真实) 或 'simulation' (模拟)。
            split (str): 'train' (训练集) 或 'test' (测试集)。
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_type = dataset_type
        self.split = split
        # 定义数据预处理流程：将 Numpy 数组转换为 PyTorch 张量。
        self.transform = transforms.Compose([transforms.ToTensor()])
        
        # 根据数据集类型，构建输入图像和真值(GT)标签的目录路径。
        base_dir_name = 'double_projectors_real_dataset' if dataset_type == 'real' else 'double_projectors_simulation_dataset'
        base_dir_path = self.dataset_path / base_dir_name

        # --- 新增：智能路径验证与修正 ---
        if not base_dir_path.is_dir():
            print(f"提示：在 '{self.dataset_path}' 中未直接找到 '{base_dir_name}'。")
            # 寻找可能的子目录
            sub_dirs = [d for d in self.dataset_path.iterdir() if d.is_dir()]
            found = False
            for sub_dir in sub_dirs:
                if (sub_dir / base_dir_name).is_dir():
                    base_dir_path = sub_dir / base_dir_name
                    print(f"成功：已在子目录 '{sub_dir.name}' 中找到数据集路径: {base_dir_path}")
                    found = True
                    break
            if not found:
                print(f"错误：无法在 '{self.dataset_path}' 或其直接子目录中找到 '{base_dir_name}'。")
                print(f"请确保您的目录结构类似于: '{self.dataset_path / 'DPSL3D-measurement' / base_dir_name}'")
                raise FileNotFoundError(f"Could not find dataset directory '{base_dir_name}'")

        split_path = base_dir_path / split
        
        # --- 自动检测 'input' 或 'inpout' 文件夹 ---
        possible_input_dirs = ['input', 'inpout']
        found_input_dir_path = None
        for dir_name in possible_input_dirs:
            if (split_path / dir_name).is_dir():
                found_input_dir_path = split_path / dir_name
                break
        
        if found_input_dir_path:
            self.input_dir = found_input_dir_path
            print(f"提示：已成功找到输入文件夹: {self.input_dir}")
        else:
            print(f"错误：在 '{split_path}' 路径下既找不到 'input' 也找不到 'inpout' 文件夹。")
            raise FileNotFoundError(f"Could not find input directory ('input' or 'inpout') in {split_path}")
        
        gt_path = split_path / 'gt'
        
        # --- 自动检测左相位图(GT)文件夹 ---
        possible_left_gt_dirs = ['unwrapped_phase_left', 'left']
        found_left_gt_dir = None
        for dir_name in possible_left_gt_dirs:
            if (gt_path / dir_name).is_dir():
                found_left_gt_dir = gt_path / dir_name
                break
        
        if found_left_gt_dir:
            self.gt_left_dir = found_left_gt_dir
            print(f"提示：已成功找到左相位图文件夹: {self.gt_left_dir}")
        else:
            print(f"错误：在 '{gt_path}' 路径下找不到左相位图文件夹 (尝试了: {possible_left_gt_dirs})。")
            raise FileNotFoundError(f"Could not find left ground truth directory in {gt_path}")

        # --- 自动检测右相位图(GT)文件夹 ---
        possible_right_gt_dirs = ['unwrapped_phase_right', 'right']
        found_right_gt_dir = None
        for dir_name in possible_right_gt_dirs:
            if (gt_path / dir_name).is_dir():
                found_right_gt_dir = gt_path / dir_name
                break
        
        if found_right_gt_dir:
            self.gt_right_dir = found_right_gt_dir
            print(f"提示：已成功找到右相位图文件夹: {self.gt_right_dir}")
        else:
            print(f"错误：在 '{gt_path}' 路径下找不到右相位图文件夹 (尝试了: {possible_right_gt_dirs})。")
            raise FileNotFoundError(f"Could not find right ground truth directory in {gt_path}")
        
        # 获取并排序所有输入文件的文件名，确保输入和标签一一对应。
        self.file_names = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.mat')])
        if not self.file_names:
            print(f"警告：在 '{self.input_dir}' 中没有找到任何 .mat 文件。请检查您的数据集内容。")
        
    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.file_names)
    
    def __getitem__(self, idx):
        """
        根据给定的索引 `idx`，加载并返回一个数据样本。
        - PyTorch 的 DataLoader 会调用这个方法来获取一个批次的数据。
        """
        # 获取文件名并构建完整的文件路径。
        file_name = self.file_names[idx]
        input_path = self.input_dir / file_name
        gt_left_path = self.gt_left_dir / file_name
        gt_right_path = self.gt_right_dir / file_name
        
        # 使用 scipy.io.loadmat 加载 .mat 文件，并确保数据类型为 float32。
        # --- 自动检测 .mat 文件中的变量名 'input' 或 'fringe' ---
        try:
            mat_data = sio.loadmat(input_path)
            if 'input' in mat_data:
                input_data = mat_data['input'].astype(np.float32)
            elif 'fringe' in mat_data:
                input_data = mat_data['fringe'].astype(np.float32)
            elif 'grating' in mat_data: # --- 新增：最终修正，添加 'grating' 作为有效键 ---
                input_data = mat_data['grating'].astype(np.float32)
            else:
                # --- 诊断性报错，打印所有可用键 ---
                available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                error_msg = f"在 .mat 文件中找不到 'input'、'fringe' 或 'grating'。可用的数据键为: {available_keys}"
                raise KeyError(error_msg)
        except Exception as e:
            print(f"错误: 加载或解析 .mat 文件失败: {input_path}")
            print(f"原始错误: {e}")
            raise

        try:
            # Load left GT
            gt_left_mat = sio.loadmat(gt_left_path)
            if 'phase' in gt_left_mat:
                gt_left = gt_left_mat['phase'].astype(np.float32)
            elif 'phi_unwrapped' in gt_left_mat: # --- 新增：最终修正，添加 'phi_unwrapped' 作为有效键 ---
                gt_left = gt_left_mat['phi_unwrapped'].astype(np.float32)
            else:
                available_keys = [k for k in gt_left_mat.keys() if not k.startswith('__')]
                error_msg = f"在 GT (左) 文件中找不到 'phase' 或 'phi_unwrapped'。可用的数据键为: {available_keys}"
                raise KeyError(error_msg)
            
            # Load right GT
            gt_right_mat = sio.loadmat(gt_right_path)
            if 'phase' in gt_right_mat:
                gt_right = gt_right_mat['phase'].astype(np.float32)
            elif 'phi_unwrapped' in gt_right_mat: # --- 新增：最终修正，添加 'phi_unwrapped' 作为有效键 ---
                gt_right = gt_right_mat['phi_unwrapped'].astype(np.float32)
            else:
                available_keys = [k for k in gt_right_mat.keys() if not k.startswith('__')]
                error_msg = f"在 GT (右) 文件中找不到 'phase' 或 'phi_unwrapped'。可用的数据键为: {available_keys}"
                raise KeyError(error_msg)

        except Exception as e:
            print(f"错误: 加载 GT .mat 文件失败: {gt_left_path} 或 {gt_right_path}")
            print(f"原始错误: {e}")
            raise
        
        # 将左、右两个视角的相位图堆叠成一个双通道的 Numpy 数组。
        # 形状从 (H, W) 和 (H, W) 变为 (2, H, W)。
        targets = np.stack([gt_left, gt_right], axis=0)
        
        # 将 Numpy 数组转换为 PyTorch 张量。
        # 输入张量形状: [1, H, W] (单通道)
        # 目标张量形状: [2, H, W] (双通道)
        input_tensor = self.transform(input_data)
        # 注意: ToTensor() 会自动将 (C, H, W) 的Numpy数组转换为 (C, H, W) 的张量。
        targets_tensor = torch.from_numpy(targets)
        
        return input_tensor, targets_tensor

# ----------------------------------
# 2. 网络核心模块 (Core Modules)
# ----------------------------------
# 该残差模块是 ResUNet 的核心构建块，严格遵循论文图2的设计。
# 它结合了 Inception 的思想（多分支、多尺度处理）和 ResNet 的思想（残差连接）。
class ResidualModule(nn.Module):
    """
    论文中描述的残差模块 (Residual Module)。
    - 包含5个并行分支，用于在不同尺度上提取特征。
    - 分支1-4的输出被拼接并通过1x1卷积融合。
    - 融合结果与分支0的输出（通过残差连接）相加。
    """
    def __init__(self, in_channels, out_channels):
        """
        初始化残差模块的各个层。
        Args:
            in_channels (int): 输入特征图的通道数。
            out_channels (int): 输出特征图的通道数。
        """
        super(ResidualModule, self).__init__()
        # 每个特征提取分支的输出通道数，为总输出通道数的1/4。
        reduced_channels = out_channels // 4
        
        # Branch 0 (残差连接分支): 一个 1x1 卷积，直接将输入映射到输出维度。
        # 1x1 卷积用于调整通道数并增加非线性。
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels), # 批量归一化，加速收敛并稳定训练
            nn.ReLU(inplace=True)         # ReLU 激活函数
        )
        
        # Branch 1: 一个 3x3 卷积，捕获局部特征。
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=3, padding=1), # padding=1 保持尺寸不变
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 两个串联的 3x3 卷积，感受野更大，能学习更复杂的特征。
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 三个串联的 3x3 卷积。
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: 四个串联的 3x3 卷积。
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        # 融合层: 使用一个 1x1 卷积来融合四个分支的输出。
        # 输入通道数是 reduced_channels * 4，输出通道数是 out_channels。
        self.conv_concat = nn.Conv2d(reduced_channels * 4, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """定义前向传播过程。"""
        # 计算每个分支的输出。
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # 在通道维度 (dim=1) 上拼接分支1-4的输出。
        concat = torch.cat([b1, b2, b3, b4], dim=1)
        
        # 融合拼接后的特征。
        concat = self.conv_concat(concat)
        concat = self.bn(concat)
        
        # 残差连接：将融合后的特征与分支0的输出相加。
        # 这使得网络可以轻松地学习恒等映射，有助于缓解梯度消失问题。
        out = self.relu(concat + b0)
        return out

# ----------------------------------
# 3. ResUNet 网络结构
# ----------------------------------
# 这是论文中提出的 ResUNet 模型，严格遵循图3(b)的 U 型架构。
# 它由一个编码器（下采样路径）、一个解码器（上采样路径）和跳跃连接组成。
class ResUNet(nn.Module):
    """
    ResUNet 网络结构。
    - 编码器部分通过残差模块和最大池化层逐步提取特征并减小空间尺寸。
    - 解码器部分通过转置卷积进行上采样，并与编码器对应层的特征图进行拼接。
    - 跳跃连接 (Skip Connection) 使得解码器可以利用编码器的低级特征，有助于恢复细节。
    """
    def __init__(self, in_channels=1, out_channels=2):
        super(ResUNet, self).__init__()
        
        # ---- 编码器 (Encoder Path) ----
        # 每一层编码器包含一个残差模块和一个最大池化层。
        # 通道数增加: 1 -> 64 -> 128 -> 256 -> 512 -> 1024
        self.enc1 = ResidualModule(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2) # 2x2 最大池化，空间尺寸减半
        self.enc2 = ResidualModule(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualModule(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualModule(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        # 瓶颈层 (Bottleneck)，只用残差模块，没有池化。
        self.enc5 = ResidualModule(512, 1024)

        # ---- 解码器 (Decoder Path) ----
        # 每一层解码器包含一个转置卷积（上采样）和一个残差模块。
        # 通道数减少: 1024 -> 512 -> 256 -> 128 -> 64
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # 空间尺寸翻倍
        # 解码器的输入通道数 = 上采样输出通道数 + 跳跃连接的编码器特征通道数
        self.dec4 = ResidualModule(1024, 512) # 512 (from up4) + 512 (from enc4) = 1024
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ResidualModule(512, 256)  # 256 (from up3) + 256 (from enc3) = 512
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResidualModule(256, 128)  # 128 (from up2) + 128 (from enc2) = 256
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResidualModule(128, 64)   # 64 (from up1) + 64 (from enc1) = 128
        
        # ---- 输出层 (Output Layer) ----
        # 使用一个 1x1 卷积将特征图的通道数调整为最终的输出通道数 (默认为2)。
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        """定义完整的前向传播路径。"""
        # ---- 编码路径 ----
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        e5 = self.enc5(self.pool4(e4)) # 瓶颈层特征

        # ---- 解码路径 + 跳跃连接 ----
        # 第4层解码
        d4 = self.up4(e5)                # 上采样
        d4 = torch.cat([e4, d4], dim=1)  # 跳跃连接：与编码器第4层特征拼接
        d4 = self.dec4(d4)               # 通过残差模块
        
        # 第3层解码
        d3 = self.up3(d4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        
        # 第2层解码
        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        # 第1层解码
        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        # 输出最终结果
        return self.out(d1)

# ----------------------------------
# 4. 训练与评估 (Training & Evaluation)
# ----------------------------------
# 定义一个辅助函数，用于计算评估指标。
def compute_metrics(outputs, targets):
    """
    计算平均绝对误差 (MAE) 和均方根误差 (RMSE)。
    Args:
        outputs (Tensor): 模型的预测输出。
        targets (Tensor): 真实的标签。
    Returns:
        tuple: (mae, rmse)
    """
    # MAE: 预测值和真实值之差的绝对值的平均值，对异常值不敏感。
    mae = torch.mean(torch.abs(outputs - targets)).item()
    # RMSE: 预测值和真实值之差的平方的均值的平方根，对大误差更敏感。
    rmse = torch.sqrt(torch.mean((outputs - targets) ** 2)).item()
    return mae, rmse

# 定义主函数，用于执行训练和评估流程。
def train_and_evaluate(dataset_path, dataset_type='real', num_epochs=100, batch_size=8, lr=0.0005):
    """
    完整的训练和验证流程。
    Args:
        dataset_path (str): 数据集根目录。
        dataset_type (str): 'real' 或 'simulation'。
        num_epochs (int): 训练的总轮数。
        batch_size (int): 批量大小。
        lr (float): 学习率。
    """
    # 自动选择设备：如果 CUDA 可用，则使用 GPU，否则使用 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 实例化模型并将其移动到指定设备。
    model = ResUNet().to(device)
    
    # 定义损失函数：L1Loss (即 MAE Loss)，适合图像回归任务。
    criterion = nn.L1Loss()
    
    # 定义优化器：Adam，一种自适应学习率的优化算法。
    # weight_decay 用于 L2 正则化，防止过拟合。
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 定义学习率调度器：余弦退火 (CosineAnnealingLR)。
    # 学习率会根据余弦函数周期性地在初始值和最小值之间变化。
    # T_max 是半个周期的长度（以 epoch 为单位）。
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    
    # ---- 数据加载 ----
    # 创建训练和验证集
    train_dataset = PhaseDataset(dataset_path, dataset_type, 'train')
    val_dataset = PhaseDataset(dataset_path, dataset_type, 'test')
    # 创建数据加载器 (DataLoader)，用于批量加载数据。
    # shuffle=True 表示在每个 epoch 开始时打乱训练数据。
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    # ---- 训练循环 ----
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # -- 训练阶段 --
        model.train() # 将模型设置为训练模式
        train_loss, train_mae, train_rmse = 0.0, 0.0, 0.0
        
        # 使用 tqdm 创建训练进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False)
        for inputs, targets in train_pbar:
            # 将数据移动到 GPU/CPU
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 1. 梯度清零
            optimizer.zero_grad()
            # 2. 前向传播
            outputs = model(inputs)
            # 3. 计算损失
            loss = criterion(outputs, targets)
            # 4. 反向传播
            loss.backward()
            # 5. 更新权重
            optimizer.step()
            
            # 累计损失和指标
            mae, rmse = compute_metrics(outputs, targets)
            train_loss += loss.item()
            train_mae += mae
            train_rmse += rmse
            
            # 更新进度条后缀，显示实时指标
            train_pbar.set_postfix(loss=f'{loss.item():.4f}', mae=f'{mae:.4f}')
        
        # 更新学习率
        scheduler.step()
        
        # -- 验证阶段 --
        model.eval() # 将模型设置为评估模式 (例如，禁用 Dropout)
        val_loss, val_mae, val_rmse = 0.0, 0.0, 0.0
        
        # 使用 tqdm 创建验证进度条
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=False)
        with torch.no_grad(): # 在此代码块中，禁用梯度计算，节省内存和计算资源
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                mae, rmse = compute_metrics(outputs, targets)
                val_mae += mae
                val_rmse += rmse

                # 更新进度条后缀
                val_pbar.set_postfix(loss=f'{loss.item():.4f}', mae=f'{mae:.4f}')
        
        # ---- 打印日志 ----
        # 计算并打印每个 epoch 的平均损失和指标
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train MAE: {train_mae/len(train_loader):.4f}, "
              f"Train RMSE: {train_rmse/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val MAE: {val_mae/len(val_loader):.4f}, "
              f"Val RMSE: {val_rmse/len(val_loader):.4f}")

# ----------------------------------
# 5. 程序入口
# ----------------------------------
if __name__ == "__main__":
    # ---- 超参数配置 ----
    # !!重要!!: 自动获取项目根目录并拼接数据集路径，避免手动设置错误。
    # 假设 'double_projectors_3D' 文件夹与 ResUNet.py 在同一项目根目录下。
    project_root = os.path.abspath('.') 
    dataset_path = os.path.join(project_root, "double_projectors_3D")
    
    # 检查路径是否存在，如果不存在则给出提示
    if not os.path.isdir(dataset_path):
        print(f"错误：数据集路径不存在 -> {dataset_path}")
        print("请确保名为 'double_projectors_3D' 的数据集文件夹位于项目根目录中。")
        exit() # 退出程序

    # 选择数据集类型: 'real' 或 'simulation'
    dataset_type_to_use = 'real' 
    
    # 定义训练超参数
    epochs = 100
    # !!重要!!: batch_size 决定了一次性送入 GPU 的图片数量。
    # 如果遇到 'CUDA out of memory' 错误，请首先减小此值。
    # 对于 12GB 显存和 640x512 的图像，2 或 4 是一个比较安全的值。
    batch = 2 
    learning_rate = 0.0005

    # 开始训练
    train_and_evaluate(
        dataset_path=dataset_path, 
        dataset_type=dataset_type_to_use, 
        num_epochs=epochs, 
        batch_size=batch, 
        lr=learning_rate
    )