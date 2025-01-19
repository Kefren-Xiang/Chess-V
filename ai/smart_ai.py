import chess
import torch
import random
from .model import ChessModel

class SmartAI:
    def __init__(self, color, model_file, device, mode='train'):
        self.color = color
        self.model = ChessModel().to(device)
        self.device = device
        self.model.load_state_dict(torch.load(model_file))
        self.mode = mode

    def choose_move(self, board):
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 0:
            return None

        # 如果是训练阶段，使用MCTS选择动作
        if self.mode == 'train':
            move = self.mcts(board)
        else:
            move = self.policy_move(board, legal_moves)
        return move

    def mcts(self, board):
        # 使用MCTS算法进行选择，简化版，实际应有更复杂的搜索逻辑
        legal_moves = list(board.legal_moves)
        move = random.choice(legal_moves)  # 随机选择
        return move

    def policy_move(self, board, legal_moves):
        # 获取棋盘状态的输入数据
        input_tensor = self.board_to_input_tensor(board)
        policy, value = self.model(input_tensor)

        # 计算策略概率
        policy_probs = torch.softmax(policy, dim=0).cpu().detach().numpy()
        move_probs = [policy_probs[move] for move in legal_moves]
        selected_move = random.choices(legal_moves, move_probs, k=1)[0]

        return selected_move

    def board_to_input_tensor(self, board):
        # 将棋盘转换为8x8x12的输入数据形式
        input_data = torch.zeros(12, 8, 8)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                color_idx = 0 if piece.color == chess.WHITE else 1
                piece_idx = piece.piece_type - 1
                input_data[2 * color_idx + piece_idx, square // 8, square % 8] = 1
        return input_data.unsqueeze(0).to(self.device)
