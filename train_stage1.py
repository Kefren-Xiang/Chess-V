import chess
import torch
import random
import pandas as pd
from ai.smart_ai import SmartAI
from ai.model import ChessModel
from utils import save_game_data

def train_stage1():
    board = chess.Board()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ChessModel().to(device)
    white_ai = SmartAI(chess.WHITE, model_file='model/model.pth', device=device, mode='train')
    black_ai = SmartAI(chess.BLACK, model_file='model/model.pth', device=device, mode='train')

    game_data = []

    for episode in range(100000):  # 十万次对局
        board = chess.Board()
        while not board.is_game_over():
            current_color = board.turn
            ai_player = white_ai if current_color == chess.WHITE else black_ai
            move = ai_player.choose_move(board)
            board.push(move)

        # 收集数据
        game_data.append(collect_game_data(board, white_ai, black_ai))

    # 保存数据
    save_game_data(game_data, 'data/train_stage1.csv')

def collect_game_data(board, white_ai, black_ai):
    # 假设返回棋局数据和游戏结果
    return {'board': board.fen(), 'result': board.result()}

if __name__ == '__main__':
    train_stage1()
