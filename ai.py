"""
ai.py
ai棋类辅助手
"""

import chess
import random

class SmartAI:
    def __init__(self, color, depth=3, use_controller=False,
                 model_file=None, scaler_file=None, device='cpu', mode='random'):
        """
        国际象棋AI基类
        :param color: 执棋方 chess.WHITE/chess.BLACK
        :param depth: 搜索深度（保留参数）
        :param use_controller: 使用控制器（保留参数）
        :param model_file: 模型文件路径（保留参数）
        :param scaler_file: 数据标准化文件（保留参数）
        :param device: 计算设备（保留参数）
        :param mode: 模式选择（保留参数）
        """
        self.color = color
        self.mode = mode

    def choose_move(self, board):
        """
        选择走法的主接口
        :param board: 当前棋盘状态
        :return: chess.Move对象或None
        """
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves) if legal_moves else None