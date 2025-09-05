import os
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ResUNet_model import ResUNet


class PhaseDataset(Dataset):
    """
    用于加载叠加光栅图像和对应真值相位图的数据集。
    - 从 .mat 文件中读取数据。
    - 输入是单通道的叠加光栅图。
    - 输出是双通道的真值相位图（左、右视角）。
    """
    def __init__(self, dataset_path, dataset_type='real', split='train'):
        self.dataset_path = Path(dataset_path)
        self.dataset_type = dataset_type
        self.split = split
        self.transform = transforms.Compose([transforms.ToTensor()])

        base_dir_name = 'double_projectors_real_dataset' if dataset_type == 'real' else 'double_projectors_simulation_dataset'
        base_dir_path = self.dataset_path / base_dir_name

        if not base_dir_path.is_dir():
            print(f"提示：在 '{self.dataset_path}' 中未直接找到 '{base_dir_name}'。")
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

        # 'input' or common misspelling 'inpout'
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

        # left gt
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

        # right gt
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

        self.file_names = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.mat')])
        if not self.file_names:
            print(f"警告：在 '{self.input_dir}' 中没有找到任何 .mat 文件。请检查您的数据集内容。")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        input_path = self.input_dir / file_name
        gt_left_path = self.gt_left_dir / file_name
        gt_right_path = self.gt_right_dir / file_name

        try:
            mat_data = sio.loadmat(input_path)
            if 'input' in mat_data:
                input_data = mat_data['input'].astype(np.float32)
            elif 'fringe' in mat_data:
                input_data = mat_data['fringe'].astype(np.float32)
            elif 'grating' in mat_data:
                input_data = mat_data['grating'].astype(np.float32)
            else:
                available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                raise KeyError(f"在 .mat 文件中找不到 'input'、'fringe' 或 'grating'。可用的数据键为: {available_keys}")
        except Exception as e:
            print(f"错误: 加载或解析 .mat 文件失败: {input_path}")
            print(f"原始错误: {e}")
            raise

        try:
            gt_left_mat = sio.loadmat(gt_left_path)
            if 'phase' in gt_left_mat:
                gt_left = gt_left_mat['phase'].astype(np.float32)
            elif 'phi_unwrapped' in gt_left_mat:
                gt_left = gt_left_mat['phi_unwrapped'].astype(np.float32)
            else:
                available_keys = [k for k in gt_left_mat.keys() if not k.startswith('__')]
                raise KeyError(f"在 GT (左) 文件中找不到 'phase' 或 'phi_unwrapped'。可用的数据键为: {available_keys}")

            gt_right_mat = sio.loadmat(gt_right_path)
            if 'phase' in gt_right_mat:
                gt_right = gt_right_mat['phase'].astype(np.float32)
            elif 'phi_unwrapped' in gt_right_mat:
                gt_right = gt_right_mat['phi_unwrapped'].astype(np.float32)
            else:
                available_keys = [k for k in gt_right_mat.keys() if not k.startswith('__')]
                raise KeyError(f"在 GT (右) 文件中找不到 'phase' 或 'phi_unwrapped'。可用的数据键为: {available_keys}")
        except Exception as e:
            print(f"错误: 加载 GT .mat 文件失败: {gt_left_path} 或 {gt_right_path}")
            print(f"原始错误: {e}")
            raise

        targets = np.stack([gt_left, gt_right], axis=0)
        input_tensor = self.transform(input_data)
        targets_tensor = torch.from_numpy(targets)
        return input_tensor, targets_tensor


def compute_metrics(outputs, targets):
    mae = torch.mean(torch.abs(outputs - targets)).item()
    rmse = torch.sqrt(torch.mean((outputs - targets) ** 2)).item()
    return mae, rmse


def train_and_evaluate(dataset_path, dataset_type='real', num_epochs=100, batch_size=8, lr=0.0005, verbose=True, show_progress=True, progress_style='clean'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ResUNet().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    train_dataset = PhaseDataset(dataset_path, dataset_type, 'train')
    val_dataset = PhaseDataset(dataset_path, dataset_type, 'test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)  # 设置num_workers=0避免多进程问题
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    
    print(f"训练数据集大小: {len(train_dataset)}")
    print(f"验证数据集大小: {len(val_dataset)}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")

    # 测试数据加载
    try:
        print("测试数据加载...")
        test_batch = next(iter(train_loader))
        print(f"成功加载测试批次，输入形状: {test_batch[0].shape}, 目标形状: {test_batch[1].shape}")
    except Exception as e:
        print(f"数据加载测试失败: {e}")
        return
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Dataset: {dataset_type}, Batch size: {batch_size}, Learning rate: {lr}")
    print("=" * 100)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss, train_mae, train_rmse = 0.0, 0.0, 0.0
        
        # 简化的训练循环
        if verbose and not show_progress:
            print(f"Epoch {epoch+1:3d}/{num_epochs} [Training]...", end=" ", flush=True)
        
        # 使用简单的进度显示，避免复杂的tqdm设置
        if show_progress and verbose:
            if progress_style == 'clean':
                train_pbar = tqdm(train_loader, 
                                desc=f"Epoch {epoch+1:3d}/{num_epochs} [Train]", 
                                leave=False, 
                                disable=False, 
                                bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            else:
                train_pbar = tqdm(train_loader, 
                                desc=f"Epoch {epoch+1:3d}/{num_epochs} [Train]", 
                                leave=False)
        else:
            train_pbar = train_loader
        
        batch_count = 0
        try:
            for batch_idx, (inputs, targets) in enumerate(train_pbar):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                mae, rmse = compute_metrics(outputs, targets)
                train_loss += loss.item()
                train_mae += mae
                train_rmse += rmse
                batch_count += 1
                
                # 每50个batch显示一次进度（当不使用进度条时）
                if not show_progress and verbose and (batch_idx + 1) % 50 == 0:
                    print(f"{batch_idx + 1}/{len(train_loader)}", end=" ", flush=True)
                    
                # 第一个batch后显示成功信息（只在非进度条模式下）
                if batch_idx == 0 and verbose and not show_progress:
                    print(f"First batch processed successfully. Loss: {loss.item():.4f}", end=" ", flush=True)
        except Exception as e:
            print(f"\n训练过程中发生错误: {e}")
            raise

        scheduler.step()

        # 验证阶段
        model.eval()
        val_loss, val_mae, val_rmse = 0.0, 0.0, 0.0
        
        if verbose and not show_progress:
            print("Validating...", end=" ", flush=True)
        
        # 验证阶段也使用简化的进度显示
        if show_progress and verbose:
            if progress_style == 'clean':
                val_pbar = tqdm(val_loader, 
                              desc=f"Epoch {epoch+1:3d}/{num_epochs} [Val]", 
                              leave=False, 
                              disable=False,
                              bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            else:
                val_pbar = tqdm(val_loader, 
                              desc=f"Epoch {epoch+1:3d}/{num_epochs} [Val]", 
                              leave=False)
        else:
            val_pbar = val_loader
        
        val_batch_count = 0
        try:
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_pbar):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    mae, rmse = compute_metrics(outputs, targets)
                    val_mae += mae
                    val_rmse += rmse
                    val_batch_count += 1
                    
                    # 每25个batch显示一次进度（验证通常batch较少）
                    if not show_progress and verbose and (batch_idx + 1) % 25 == 0:
                        print(f"{batch_idx + 1}/{len(val_loader)}", end=" ", flush=True)
        except Exception as e:
            print(f"\n验证过程中发生错误: {e}")
            raise

        # 计算平均指标
        avg_train_loss = train_loss / len(train_loader)
        avg_train_mae = train_mae / len(train_loader)
        avg_train_rmse = train_rmse / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        avg_val_rmse = val_rmse / len(val_loader)
        
        # 格式化输出每个epoch的结果
        print(f"Done! | "
              f"Train: Loss={avg_train_loss:.4f}, MAE={avg_train_mae:.4f}, RMSE={avg_train_rmse:.4f} | "
              f"Val: Loss={avg_val_loss:.4f}, MAE={avg_val_mae:.4f}, RMSE={avg_val_rmse:.4f}")
        
        # 每10个epoch打印分隔线
        if (epoch + 1) % 10 == 0:
            print("-" * 100)
    
    # 训练完成后的总结
    print("\n" + "=" * 100)
    print("训练完成！")
    print(f"总训练轮数: {num_epochs}")
    print(f"最终训练损失: {avg_train_loss:.4f}")
    print(f"最终验证损失: {avg_val_loss:.4f}")
    print(f"最终训练MAE: {avg_train_mae:.4f}")
    print(f"最终验证MAE: {avg_val_mae:.4f}")
    print("=" * 100)


if __name__ == "__main__":
    project_root = os.path.abspath('.')
    dataset_path = os.path.join(project_root, "double_projectors_3D")

    if not os.path.isdir(dataset_path):
        print(f"错误：数据集路径不存在 -> {dataset_path}")
        print("请确保名为 'double_projectors_3D' 的数据集文件夹位于项目根目录中。")
        exit()

    dataset_type_to_use = 'real'
    epochs = 100
    batch = 2
    learning_rate = 0.0005
    verbose_mode = True  # 设置为False可以禁用详细输出
    show_progress_mode = True  # 启用进度条
    progress_style_mode = 'clean'  # 'clean' 或 'default'

    train_and_evaluate(
        dataset_path=dataset_path,
        dataset_type=dataset_type_to_use,
        num_epochs=epochs,
        batch_size=batch,
        lr=learning_rate,
        verbose=verbose_mode,
        show_progress=show_progress_mode,
        progress_style=progress_style_mode
    )


