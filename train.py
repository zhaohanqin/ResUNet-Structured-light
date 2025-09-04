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


def train_and_evaluate(dataset_path, dataset_type='real', num_epochs=100, batch_size=8, lr=0.0005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ResUNet().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    train_dataset = PhaseDataset(dataset_path, dataset_type, 'train')
    val_dataset = PhaseDataset(dataset_path, dataset_type, 'test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_mae, train_rmse = 0.0, 0.0, 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False)
        for inputs, targets in train_pbar:
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
            train_pbar.set_postfix(loss=f'{loss.item():.4f}', mae=f'{mae:.4f}')

        scheduler.step()

        model.eval()
        val_loss, val_mae, val_rmse = 0.0, 0.0, 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=False)
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                mae, rmse = compute_metrics(outputs, targets)
                val_mae += mae
                val_rmse += rmse
                val_pbar.set_postfix(loss=f'{loss.item():.4f}', mae=f'{mae:.4f}')

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train MAE: {train_mae/len(train_loader):.4f}, "
              f"Train RMSE: {train_rmse/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val MAE: {val_mae/len(val_loader):.4f}, "
              f"Val RMSE: {val_rmse/len(val_loader):.4f}")


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

    train_and_evaluate(
        dataset_path=dataset_path,
        dataset_type=dataset_type_to_use,
        num_epochs=epochs,
        batch_size=batch,
        lr=learning_rate
    )


