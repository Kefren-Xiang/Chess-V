import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        
        # 定义卷积层
        self.conv1 = nn.Conv2d(12, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # 全连接层
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        
        # 策略头：输出棋盘上每个合法位置的概率（8x8=64个位置）
        self.policy_head = nn.Linear(512, 64)
        
        # 价值头：输出棋局的价值（-1到1之间的值）
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        # 网络前向传播
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 拉平为一维
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # 输出策略和价值
        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value
