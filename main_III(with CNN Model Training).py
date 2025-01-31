import chess
import csv
import math
import os
import torch
import numpy as np
from tqdm import tqdm
from util_function import normalize_fen, get_board_pipelines, get_white_result
from torch.utils.data import Dataset, DataLoader
from ai import SmartAI

# 配置参数
NUM_GAMES = 10000
SAVE_INTERVAL = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- CNN模型定义 -------------------------
class ChessValueCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(12, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(512*2*2, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# ------------------------- 数据集定义 -------------------------
class ChessDataset(Dataset):
    def __init__(self, csv_path):
        self.data = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append((row['FEN'], float(row['Q'])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, q = self.data[idx]
        board = chess.Board(fen)
        pipelines = get_board_pipelines(board)
        feature = np.zeros((12, 8, 8), dtype=np.float32)
        channel_order = [
            "black_king", "black_queen", "black_rook",
            "black_knight", "black_bishop", "black_pawn",
            "white_pawn", "white_bishop", "white_knight",
            "white_rook", "white_queen", "white_king"
        ]
        for i, key in enumerate(channel_order):
            feature[i] = np.array(pipelines[key], dtype=np.float32)
        return torch.tensor(feature), torch.tensor(q).float()

# ------------------------- MCTS树与AI类 -------------------------
class MCTSTree:
    def __init__(self, model_path=None):
        self.nodes = {}
        self.model = ChessValueCNN().to(DEVICE)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.train_step = 0

    def predict_v(self, board):
        with torch.no_grad():
            pipelines = get_board_pipelines(board)
            tensor = torch.zeros((12, 8, 8), dtype=torch.float32)
            channel_order = [
                "black_king", "black_queen", "black_rook",
                "black_knight", "black_bishop", "black_pawn",
                "white_pawn", "white_bishop", "white_knight",
                "white_rook", "white_queen", "white_king"
            ]
            for i, key in enumerate(channel_order):
                tensor[i] = torch.tensor(pipelines[key], dtype=torch.float32)
            return self.model(tensor.unsqueeze(0).to(DEVICE)).item()

class MCTSAI(SmartAI):
    def __init__(self, color, tree):
        super().__init__(color)
        self.tree = tree

    def choose_move(self, board):
        current_fen = normalize_fen(board)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # 计算所有走法的V+U值
        scores = []
        parent_node = self.tree.nodes.get(current_fen, {'children': {}, 'N': 0, 'W': 0})
        total_N = sum(child['N'] for child in parent_node['children'].values())
        
        for move in legal_moves:
            board.push(move)
            child_fen = normalize_fen(board)
            board.pop()
            
            # 初始化子节点
            if child_fen not in self.tree.nodes:
                self.tree.nodes[child_fen] = {
                    'W': 0.0,
                    'N': 1,
                    'parent': current_fen,
                    'children': {}
                }
            
            # 计算U值
            child_N = self.tree.nodes[child_fen]['N']
            P = (parent_node['children'].get(move.uci(), {'N': 0})['N'] + 1) / (total_N + len(legal_moves))
            k = 3 / ((self.tree.train_step % 10000)**1.5 + 0.5) + 1
            U = k * P * math.sqrt(total_N + 1) / (child_N + 1e-5)
            
            # 计算V值
            V = self.tree.predict_v(chess.Board(child_fen))
            
            scores.append( (move, V + U) )
        
        return max(scores, key=lambda x: x[1])[0]

# ------------------------- 核心逻辑 -------------------------
def backpropagate(path, result, tree):
    """区分中间节点和叶子节点反向传播"""
    for fen in reversed(path):
        node = tree.nodes.setdefault(fen, {'W': 0.0, 'N': 0, 'children': {}, 'parent': None})
        node['W'] += result
        node['N'] += 1

def save_append_data(tree, filename):
    """追加模式保存数据"""
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['FEN', 'W', 'N', 'Q', 'P', 'U'])
        
        for fen in tree.nodes:
            node = tree.nodes[fen]
            Q = node['W'] / node['N'] if node['N'] > 0 else 0.0
            
            # 计算P值
            parent = tree.nodes.get(node.get('parent'), {'children': {}})
            total_N = sum(c['N'] for c in parent['children'].values()) if node.get('parent') else 0
            P = node['N'] / total_N if total_N > 0 else 0.0
            
            # 计算U值
            k = 3 / ((tree.train_step % 10000)**1.5 + 0.5) + 1
            U = k * P * math.sqrt(total_N) / node['N'] if node['N'] > 0 else 0.0
            
            writer.writerow([fen, node['W'], node['N'], Q, P, U])

def train_cnn_model(csv_path, model_path):
    """增量训练CNN模型"""
    dataset = ChessDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = ChessValueCNN().to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    model.train()
    for epoch in range(3):  # 快速微调
        for features, targets in dataloader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), model_path)
    return model

if __name__ == "__main__":
    # 初始化
    os.makedirs("CNN_for_Value", exist_ok=True)
    model_path = "CNN_for_Value/latest_model.pth"
    tree = MCTSTree(model_path if os.path.exists(model_path) else None)
    
    # 分批次运行
    for batch in range(NUM_GAMES // SAVE_INTERVAL):
        # 运行一万次对局
        for _ in tqdm(range(SAVE_INTERVAL), desc=f"Batch {batch+1}"):
            board = chess.Board()
            path = [normalize_fen(board)]
            ai_white = MCTSAI(chess.WHITE, tree)
            ai_black = MCTSAI(chess.BLACK, tree)
            
            while not board.is_game_over():
                current_ai = ai_white if board.turn == chess.WHITE else ai_black
                move = current_ai.choose_move(board)
                if not move:
                    break
                board.push(move)
                path.append(normalize_fen(board))
            
            # 获取结果
            result = get_white_result(board) if board.is_game_over() else 0.5
            backpropagate(path, result, tree)
            tree.train_step += 1
        
        # 保存数据并训练
        save_append_data(tree, "Chess_Information_II.csv")
        tree.model = train_cnn_model("Chess_Information_II.csv", model_path)
    
    print("训练完成！最终模型已保存至 CNN_for_Value/")