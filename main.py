import chess
from ai.smart_ai import SmartAI
import torch

def print_board(board):
    # 打印Unicode棋盘
    print(board.unicode(invert_color=False))

def main():
    board = chess.Board()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    black_ai = SmartAI(chess.BLACK, model_file='model/model.pth', device=device, mode='search')
    white_ai = SmartAI(chess.WHITE, model_file='model/model.pth', device=device, mode='search')

    print("初始棋盘:")
    print_board(board)
    print("\n开始游戏...\n")

    while not board.is_game_over():
        current_color = board.turn
        ai_player = white_ai if current_color == chess.WHITE else black_ai
        move = ai_player.choose_move(board)

        if move is None:
            print("没有合法的走法。")
            break

        san = board.san(move)
        board.push(move)

        player = "White" if current_color == chess.WHITE else "Black"
        print(f"{player} moves: {move.uci()} ({san})")
        print_board(board)
        print("\n")

    print("游戏结束！")
    result = board.result()
    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        print(f"将死，{winner} 获胜！")
    elif board.is_stalemate():
        print("和棋（僵局）")
    elif board.is_insufficient_material():
        print("和棋（兵种不足）")
    elif board.is_seventyfive_moves():
        print("和棋（75回合规则）")
    elif board.is_fivefold_repetition():
        print("和棋（五次重复）")
    else:
        print(f"结果: {result}")

if __name__ == "__main__":
    main()
