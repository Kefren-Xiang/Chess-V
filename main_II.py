"""
main_II.py:
main_II是第一轮生成数据的文件，DFSAI是模拟深度优先的对战思路。慎点！
平均一秒生成十轮对局，100000轮约花费近3h时间。
"""

import chess
import csv
import random
import math
from ai import SmartAI
from tqdm import tqdm
from util_function import normalize_fen, get_white_result

class EnhancedChessTree:
    def __init__(self):
        self.nodes = {}  # {fen: {'W': float, 'N': int, 'parent': str, 'children': dict}}

    def register_move(self, parent_fen, move, child_fen):
        # 父节点初始化
        if parent_fen not in self.nodes:
            self.nodes[parent_fen] = {'W': 0.0, 'N': 1, 'parent': None, 'children': {}}
        
        # 子节点初始化
        if child_fen not in self.nodes:
            self.nodes[child_fen] = {
                'W': 0.0, 
                'N': 1,
                'parent': parent_fen,
                'children': {}
            }
        
        # 更新走法关系
        move_uci = move.uci()
        if move_uci not in self.nodes[parent_fen]['children']:
            self.nodes[parent_fen]['children'][move_uci] = {
                'child_fen': child_fen,
                'N': 1  # 走法访问次数
            }
        else:
            self.nodes[parent_fen]['children'][move_uci]['N'] += 1
            
class DFSAI_V7(SmartAI):
    def __init__(self, color, tree, exploration=0.3):
        super().__init__(color)
        self.tree = tree
        self.exploration = exploration

    def choose_move(self, board):
        legal_moves = list(board.legal_moves)
        # 如果压根没有合法走法，那说明要么将死，要么和棋
        if not legal_moves:
            return None

        current_fen = normalize_fen(board)
        current_node = self.tree.nodes.get(current_fen, {'children': {}})

        # 先做探索还是保留参数
        if random.random() < self.exploration:
            move = self._choose_new_move(board, current_node, legal_moves)
        else:
            move = self._choose_least_visited(current_node, legal_moves)

        # 兜底：如果所有分支都在树里出现过，就 `_choose_new_move` 可能返回 None
        # 那我们这里再做一次容错，随便选一个合法走法都可以
        if move is None:
            move = random.choice(legal_moves)
        return move

    def _choose_new_move(self, board, current_node, legal_moves):
        new_moves = []
        for move in legal_moves:
            board.push(move)
            next_fen = normalize_fen(board)
            board.pop()
            # 如果这个 next_fen 没在 current_node['children'] 里出现过，就算"新招法"
            if not any(m['child_fen'] == next_fen for m in current_node['children'].values()):
                new_moves.append(move)

        # 如果存在新招法，就随机选一个返回
        if new_moves:
            return random.choice(new_moves)
        else:
            # 如果没有新招法可选，就返回 None（之后在 choose_move 再去兜底）
            return None

    def _choose_least_visited(self, current_node, legal_moves):
        move_visits = []
        for move in legal_moves:
            move_uci = move.uci()
            if move_uci in current_node['children']:
                visits = current_node['children'][move_uci]['N']
            else:
                # 该走法没在子节点出现过，令访问次数为 0
                visits = 0
            move_visits.append( (move, visits) )

        if not move_visits:
            return None  # 理论上不会出现，因为 legal_moves 不空
        # 返回访问次数最小的走法
        return min(move_visits, key=lambda x: x[1])[0]

    def _choose_least_visited(self, current_node, legal_moves):
        move_visits = []
        for move in legal_moves:
            move_uci = move.uci()
            if move_uci in current_node['children']:
                visits = current_node['children'][move_uci]['N']
                move_visits.append( (move, visits) )
        return min(move_visits, key=lambda x: x[1])[0] if move_visits else None
    
def run_enhanced_simulation(num_games=100000):
    tree = EnhancedChessTree()
    global_train_step = 0

    with tqdm(total=num_games, desc="增强模拟进度") as pbar:
        for _ in range(num_games):
            board = chess.Board()
            initial_fen = normalize_fen(board)

            if initial_fen not in tree.nodes:
                tree.nodes[initial_fen] = {
                    'W': 0.0,
                    'N': 1,
                    'parent': None,
                    'children': {}
                }

            path = [initial_fen]
            ai_white = DFSAI_V7(chess.WHITE, tree)
            ai_black = DFSAI_V7(chess.BLACK, tree)

            move_count = 0
            max_moves = 300  # 防止无限循环或过长对局
            while not board.is_game_over() and move_count < max_moves:
                current_ai = ai_white if board.turn == chess.WHITE else ai_black
                prev_fen = normalize_fen(board)

                move = current_ai.choose_move(board)
                # 理论上只会在无合法走法时返回 None，这时往往 board.is_game_over()==True，
                # 但我们还是做一个判定
                if move is None:
                    # 强行结束本局
                    break

                board.push(move)
                next_fen = normalize_fen(board)
                path.append(next_fen)
                tree.register_move(prev_fen, move, next_fen)
                move_count += 1

            # 如果超过 max_moves, 也视为和棋
            if not board.is_game_over() and move_count >= max_moves:
                # 这里可以手动设置半点给白方(0.5)，你也可以更复杂判断
                # board.is_game_over() = False，但实际就当和棋对待
                result = 0.5
            else:
                # 使用现成的 get_white_result()（或者 board.outcome()）
                result = get_white_result(board)

            # 将结果反向回传给路径中的节点
            for fen in reversed(path):
                if fen not in tree.nodes:
                    tree.nodes[fen] = {
                        'W': result,
                        'N': 1,
                        'parent': None,
                        'children': {}
                    }
                else:
                    tree.nodes[fen]['W'] += result
                    tree.nodes[fen]['N'] += 1

            global_train_step += 1
            pbar.update(1)

    return tree, global_train_step


def calculate_metrics(tree, global_train_step):
    """计算所有节点的Q/P/U值"""
    for fen in tree.nodes:
        node = tree.nodes[fen]
        
        # 计算Q值
        node['Q'] = node['W'] / node['N']
        
        # 计算P值
        parent_fen = node['parent']
        if parent_fen and parent_fen in tree.nodes:
            parent = tree.nodes[parent_fen]
            total_N = sum(child['N'] for child in parent['children'].values())
            node['P'] = node['N'] / total_N if total_N else 0
        else:
            node['P'] = 0.0
        
        # 计算U值
        if node['P'] > 0 and parent_fen:
            parent = tree.nodes[parent_fen]
            total_N = sum(child['N'] for child in parent['children'].values())
            k = 3 / ((global_train_step % 10000)**1.5 + 0.5) + 1
            node['U'] = k * node['P'] * math.sqrt(total_N) / node['N']
        else:
            node['U'] = 0.0

def save_full_report(tree, filename="Chess_Information_I.csv"):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['FEN', 'W', 'N', 'Q', 'P', 'U'])
        for fen, data in tree.nodes.items():
            writer.writerow([
                fen,
                round(data['W'], 2),
                data['N'],
                round(data.get('Q', 0), 4),
                round(data.get('P', 0), 4),
                round(data.get('U', 0), 4)
            ])

if __name__ == "__main__":
    # 运行十万次对局
    chess_tree, train_steps = run_enhanced_simulation(num_games=100000)
    
    # 计算所有指标
    calculate_metrics(chess_tree, train_steps)
    
    # 保存完整报告
    save_full_report(chess_tree)
    
    # 打印统计信息
    print(f"\n最终统计：")
    print(f"总训练次数：{train_steps}")
    print(f"唯一节点数：{len(chess_tree.nodes)}")
    print(f"平均Q值：{sum(n['Q'] for n in chess_tree.nodes.values())/len(chess_tree.nodes):.2f}")
