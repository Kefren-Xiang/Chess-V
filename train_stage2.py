import torch
import pandas as pd
from ai.smart_ai import SmartAI
from ai.model import ChessModel
from utils import load_game_data, update_model_weights

def train_stage2():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化模型
    model = ChessModel().to(device)
    white_ai = SmartAI(chess.WHITE, model_file='model/model.pth', device=device, mode='train')
    black_ai = SmartAI(chess.BLACK, model_file='model/model.pth', device=device, mode='train')

    # 加载第一阶段训练数据
    game_data = load_game_data('data/train_stage1.csv')

    for episode in range(100000):  # 十万次对局
        game = random.choice(game_data)
        board = chess.Board(game['board'])

        while not board.is_game_over():
            current_color = board.turn
            ai_player = white_ai if current_color == chess.WHITE else black_ai
            move = ai_player.choose_move(board)
            board.push(move)

        # 更新模型权重
        update_model_weights(model, board, game)

    # 保存训练后的模型
    torch.save(model.state_dict(), 'model/model.pth')

if __name__ == '__main__':
    train_stage2()
