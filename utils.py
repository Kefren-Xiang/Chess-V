import pandas as pd
import torch
import chess
from ai.model import ChessModel
import pickle

def save_game_data(game_data, file_path):
    # 保存游戏数据到CSV文件
    df = pd.DataFrame(game_data)
    df.to_csv(file_path, index=False)

def load_game_data(file_path):
    # 加载游戏数据
    df = pd.read_csv(file_path)
    game_data = df.to_dict(orient='records')
    return game_data

def update_model_weights(model, board, game):
    # 更新模型权重的伪代码，需要根据实际逻辑调整
    policy, value = model(board)
    # 根据当前局面和游戏结果来调整权重
    # ... 具体的更新逻辑
    return model

def save_scaler(scaler, scaler_file):
    # 保存标准化器（假设在数据预处理阶段使用）
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)

def load_scaler(scaler_file):
    # 加载标准化器
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    return scaler
