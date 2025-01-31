"""
util_function.py
util_function为其他文件提供各个功能的函数支持。
"""

import chess

def normalize_fen(board):
    """深度标准化FEN：移除所有非必要信息"""
    fen = board.fen()
    return ' '.join(fen.split(' ')[:4])  # 保留：局面、轮次、易位权、过路兵

def get_white_result(board):
    """获取白方结果（1=赢，0.5=平，-1=输）"""
    if board.is_checkmate():
        return 1.0 if board.turn == chess.BLACK else -1.0
    elif board.is_game_over():
        return 0.5
    return 0.0  # 不会执行到这里

def print_pipelines(pipelines):
    """可视化打印管道信息"""
    piece_order = [
        "black_king", "black_queen", "black_rook",
        "black_knight", "black_bishop", "black_pawn",
        "white_pawn", "white_bishop", "white_knight",
        "white_rook", "white_queen", "white_king"
    ]

    print("\n棋盘状态管道：")
    for piece_type in piece_order:
        matrix = pipelines[piece_type]
        print(f"\n{piece_type.upper():<15}")
        for row in matrix:
            # 将数字转换为■和□显示
            visual_row = ["■" if cell else "□" for cell in row]
            print(" ".join(visual_row))


def get_board_pipelines(board):
    """
    生成12通道的棋盘状态矩阵
    返回字典格式：
    {
        "black_king": [[...], ...],
        "black_queen": [[...], ...],
        ...（共12个键）
    }
    """
    # 初始化12个8x8矩阵
    pipelines = {
        "black_king":    [[0]*8 for _ in range(8)],
        "black_queen":   [[0]*8 for _ in range(8)],
        "black_rook":    [[0]*8 for _ in range(8)],
        "black_knight":  [[0]*8 for _ in range(8)],
        "black_bishop":  [[0]*8 for _ in range(8)],
        "black_pawn":    [[0]*8 for _ in range(8)],
        "white_pawn":    [[0]*8 for _ in range(8)],
        "white_bishop":  [[0]*8 for _ in range(8)],
        "white_knight":  [[0]*8 for _ in range(8)],
        "white_rook":    [[0]*8 for _ in range(8)],
        "white_queen":   [[0]*8 for _ in range(8)],
        "white_king":    [[0]*8 for _ in range(8)],
    }

    # 遍历棋盘所有格子
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        # 转换坐标：chess库的square是a1=0, h8=63
        row = 7 - (square // 8)  # 棋盘第1行对应矩阵第7行
        col = square % 8

        # 根据棋子类型选择通道
        color = "black" if piece.color == chess.BLACK else "white"
        piece_type = None

        if piece.piece_type == chess.KING:
            piece_type = "king"
        elif piece.piece_type == chess.QUEEN:
            piece_type = "queen"
        elif piece.piece_type == chess.ROOK:
            piece_type = "rook"
        elif piece.piece_type == chess.KNIGHT:
            piece_type = "knight"
        elif piece.piece_type == chess.BISHOP:
            piece_type = "bishop"
        elif piece.piece_type == chess.PAWN:
            piece_type = "pawn"

        key = f"{color}_{piece_type}"
        pipelines[key][row][col] = 1

    return pipelines

def print_board(board):
    """打印带坐标的棋盘"""
    # 获取原始Unicode棋盘字符串
    board_str = board.unicode(invert_color=True, borders=False)
    
    # 添加坐标系统
    ranks = board_str.split('\n')
    max_rank_width = max(len(r) for r in ranks)
    
    # 构建带坐标的棋盘
    numbered_ranks = [f"{8 - i}  {r.ljust(max_rank_width)}" for i, r in enumerate(ranks)]
    coord_footer = "   " + " ".join(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    
    # 组合所有部分
    full_board = "\n".join([
        "\n" + "="*40,
        "\n".join(numbered_ranks),
        coord_footer,
        "="*40 + "\n"
    ])
    
    print(full_board)

def get_game_result(board):
    """获取详细对局结果"""
    if board.is_checkmate():
        winner = "黑方" if board.turn == chess.WHITE else "白方"
        return f"将军！{winner}获胜！"
    elif board.is_stalemate():
        return "和棋（僵局）"
    elif board.is_insufficient_material():
        return "和棋（子力不足）"
    elif board.is_seventyfive_moves():
        return "和棋（75步规则）"
    elif board.is_fivefold_repetition():
        return "和棋（五次重复）"
    return f"游戏结束：{board.result()}"
