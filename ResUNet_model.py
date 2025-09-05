import torch
import torch.nn as nn
import argparse
import sys

# 尝试导入 torchsummary 和 torchinfo
try:
    from torchsummary import summary
    TORCHSUMMARY_AVAILABLE = True
except ImportError:
    TORCHSUMMARY_AVAILABLE = False

try:
    from torchinfo import summary as torchinfo_summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False


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
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 1: 一个 3x3 卷积，捕获局部特征。
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=3, padding=1),
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
        out = self.relu(concat + b0)
        return out


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
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualModule(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualModule(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualModule(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        # 瓶颈层 (Bottleneck)
        self.enc5 = ResidualModule(512, 1024)

        # ---- 解码器 (Decoder Path) ----
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ResidualModule(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ResidualModule(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResidualModule(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResidualModule(128, 64)
        
        # ---- 输出层 (Output Layer) ----
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        """定义完整的前向传播路径。"""
        # ---- 编码路径 ----
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        e5 = self.enc5(self.pool4(e4))

        # ---- 解码路径 + 跳跃连接 ----
        d4 = self.up4(e5)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)


def print_model_summary(model, input_size, device='cpu'):
    """
    打印模型结构摘要，优先使用 torchsummary，如果不可用则使用 torchinfo
    """
    print(f"模型设备: {device}")
    print(f"输入尺寸: {input_size}")
    print("=" * 80)
    
    if TORCHSUMMARY_AVAILABLE:
        print("使用 torchsummary 显示模型结构:")
        print("-" * 40)
        try:
            summary(model, input_size, device=device)
        except Exception as e:
            print(f"torchsummary 出错: {e}")
            if TORCHINFO_AVAILABLE:
                print("\n回退到 torchinfo:")
                print("-" * 40)
                torchinfo_summary(model, input_size=input_size, device=device)
    elif TORCHINFO_AVAILABLE:
        print("使用 torchinfo 显示模型结构:")
        print("-" * 40)
        torchinfo_summary(model, input_size=input_size, device=device)
    else:
        print("警告: 未安装 torchsummary 或 torchinfo")
        print("请安装其中一个库来查看模型结构:")
        print("pip install torchsummary")
        print("或")
        print("pip install torchinfo")
        
        # 显示基本的模型信息
        print("\n基本模型信息:")
        print("-" * 40)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (假设float32)")


def main():
    """主函数，用于独立运行模型结构显示"""
    parser = argparse.ArgumentParser(description='ResUNet 模型结构显示工具')
    parser.add_argument('--input_channels', type=int, default=1, 
                       help='输入图像通道数 (默认: 1)')
    parser.add_argument('--output_channels', type=int, default=2, 
                       help='输出相位图通道数 (默认: 2)')
    parser.add_argument('--image_size', type=int, default=256, 
                       help='输入图像尺寸 (默认: 256)')
    parser.add_argument('--device', type=str, default='cpu', 
                       choices=['cpu', 'cuda'], help='运行设备 (默认: cpu)')
    
    args = parser.parse_args()
    
    # 检查设备可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，切换到 CPU")
        args.device = 'cpu'
    
    # 创建模型
    print("创建 ResUNet 模型...")
    model = ResUNet(in_channels=args.input_channels, out_channels=args.output_channels)
    model = model.to(args.device)
    
    # 打印模型结构
    input_size = (args.input_channels, args.image_size, args.image_size)
    print_model_summary(model, input_size, args.device)
    
    # 显示模型配置信息
    print("\n" + "=" * 80)
    print("模型配置信息:")
    print("-" * 40)
    print(f"输入通道数: {args.input_channels}")
    print(f"输出通道数: {args.output_channels}")
    print(f"输入图像尺寸: {args.image_size} x {args.image_size}")
    print(f"运行设备: {args.device}")
    
    # 显示网络层级信息
    print("\n网络层级信息:")
    print("-" * 40)
    print("编码器路径:")
    print("  enc1: ResidualModule(1→64) + MaxPool2d")
    print("  enc2: ResidualModule(64→128) + MaxPool2d")
    print("  enc3: ResidualModule(128→256) + MaxPool2d")
    print("  enc4: ResidualModule(256→512) + MaxPool2d")
    print("  enc5: ResidualModule(512→1024) [瓶颈层]")
    print("\n解码器路径:")
    print("  up4: ConvTranspose2d(1024→512) + Concat + ResidualModule(1024→512)")
    print("  up3: ConvTranspose2d(512→256) + Concat + ResidualModule(512→256)")
    print("  up2: ConvTranspose2d(256→128) + Concat + ResidualModule(256→128)")
    print("  up1: ConvTranspose2d(128→64) + Concat + ResidualModule(128→64)")
    print("  out: Conv2d(64→2) [输出层]")


if __name__ == "__main__":
    main()


