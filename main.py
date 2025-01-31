import chess
from ai import SmartAI
from util_function import print_pipelines, get_board_pipelines, print_board, get_game_result
import torch
import time

def main():
    # 初始化棋盘和AI
    board = chess.Board()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 初始化双方AI（当前均为随机模式）
    black_ai = SmartAI(
        color=chess.BLACK,
        depth=3,
        device=device,
        mode='random'
    )
    
    white_ai = SmartAI(
        color=chess.WHITE,
        depth=3,
        device=device,
        mode='random'
    )

    print("国际象棋对弈开始！（Chess-V 0.1）")
    print_board(board)

    while not board.is_game_over():
        # 获取当前玩家
        current_player = "白方" if board.turn == chess.WHITE else "黑方"
        current_ai = white_ai if board.turn == chess.WHITE else black_ai

        # AI选择走法
        move = current_ai.choose_move(board)
        if move is None:
            print("错误：没有合法走法！")
            break

        # 记录走法信息
        san = board.san(move)  # 标准代数记法
        uci = move.uci()       # UCI格式

        # 执行走子
        board.push(move)
        # time.sleep(5)

        # 打印对局信息
        print(f"{current_player} 走子：{uci} ({san})")
        if board.is_check():
            print("!! 将军 !!")
            time.sleep(5)
        print_board(board)
        # 执行走子后添加：
        pipelines = get_board_pipelines(board)
        print_pipelines(pipelines)
        print("\n" + "="*40)

    # 显示最终结果
    print("\n" + "="*40)
    print(get_game_result(board))
    print("="*40)

if __name__ == "__main__":
    main()