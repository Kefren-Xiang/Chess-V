"""
CNN_Model_for_Value.py
CNN_Model_for_Value采用二维卷积核组，将board转成board_pipelines的形式，并使模型学习pipelines和Q值关系。
"""


import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import chess
from torch.utils.data import Dataset, DataLoader
from util_function import get_board_pipelines
import csv
import matplotlib.pyplot as plt

# 创建保存目录
SAVE_DIR = "CNN_for_Value"
os.makedirs(SAVE_DIR, exist_ok=True)

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自定义数据集类
class ChessDataset(Dataset):
    def __init__(self, csv_path, max_samples=50000):
        self.data = []
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                if i >= max_samples:
                    break
                self.data.append((row['FEN'], float(row['Q'])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, q_value = self.data[idx]
        board = chess.Board(fen)
        
        # 生成12通道特征矩阵
        pipelines = get_board_pipelines(board)
        
        # 转换为numpy数组并调整维度顺序 (12, 8, 8)
        feature = np.zeros((12, 8, 8), dtype=np.float32)
        channel_order = [
            "black_king", "black_queen", "black_rook",
            "black_knight", "black_bishop", "black_pawn",
            "white_pawn", "white_bishop", "white_knight",
            "white_rook", "white_queen", "white_king"
        ]
        
        for i, key in enumerate(channel_order):
            feature[i] = np.array(pipelines[key], dtype=np.float32)
        
        return torch.tensor(feature), torch.tensor(q_value).float()

# CNN模型定义
class ChessValueCNN(nn.Module):
    def __init__(self):
        super(ChessValueCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # 输入: 12通道 8x8
            nn.Conv2d(12, 64, kernel_size=3, padding=1),  # 64通道 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 128通道 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128通道 4x4
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 256通道 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), # 512通道 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 512通道 2x2
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(512*2*2, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        return self.fc_layers(x)

# 训练函数
def train_model(csv_path, num_epochs=20, batch_size=256):
    # 初始化
    dataset = ChessDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model = ChessValueCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    
    # 记录训练过程
    train_losses = []
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * features.size(0)
        
        # 计算平均损失
        epoch_loss /= len(dataset)
        scheduler.step(epoch_loss)
        train_losses.append(epoch_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
        # 每5个epoch保存一次检查点
        if (epoch+1) % 5 == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
    
    # 保存最终模型
    final_model_path = os.path.join(SAVE_DIR, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    
    # 绘制损失曲线
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'loss_curve.png'))
    plt.close()
    
    print("训练完成！模型和训练记录已保存至 CNN_for_Value 目录")

if __name__ == "__main__":
    # 开始训练（使用生成的CSV文件）
    train_model("Chess_Information_I.csv", num_epochs=20)