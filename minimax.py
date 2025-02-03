import numpy as np
import copy

# オセロのボードのサイズ
BOARD_SIZE = 8

# 方向ベクトル（縦・横・斜め）
DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),         (0, 1),
    (1, -1), (1, 0), (1, 1)
]

# 盤面の初期化
def initial_board():
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    mid = BOARD_SIZE // 2
    board[mid - 1][mid - 1] = board[mid][mid] = 1  # 白
    board[mid - 1][mid] = board[mid][mid - 1] = -1  # 黒
    return board

# 指定した位置に石を置けるか判定
def is_valid_move(board, row, col, player):
    if board[row][col] != 0:
        return False
    
    for dr, dc in DIRECTIONS:
        r, c = row + dr, col + dc
        flipped = False
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == -player:
            r += dr
            c += dc
            flipped = True
        
        if flipped and 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
            return True
    return False

# 可能な手のリストを取得
def get_valid_moves(board, player):
    return [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if is_valid_move(board, r, c, player)]

# 石を置く
def apply_move(board, row, col, player):
    new_board = copy.deepcopy(board)
    new_board[row][col] = player
    
    for dr, dc in DIRECTIONS:
        r, c = row + dr, col + dc
        flipped_positions = []
        
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and new_board[r][c] == -player:
            flipped_positions.append((r, c))
            r += dr
            c += dc
        
        if flipped_positions and 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and new_board[r][c] == player:
            for fr, fc in flipped_positions:
                new_board[fr][fc] = player
    return new_board

# 評価関数（基本的な評価）
def evaluate(board, player):
    return np.sum(board) * player

# ミニマックス法（α-β枝刈りあり）
def minimax(board, depth, player, alpha, beta):
    valid_moves = get_valid_moves(board, player)
    
    if depth == 0 or not valid_moves:
        return evaluate(board, player), None
    
    best_move = None
    if player == 1:  # 白（最大化）
        max_eval = float('-inf')
        for move in valid_moves:
            new_board = apply_move(board, move[0], move[1], player)
            eval, _ = minimax(new_board, depth - 1, -player, alpha, beta)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:  # 黒（最小化）
        min_eval = float('inf')
        for move in valid_moves:
            new_board = apply_move(board, move[0], move[1], player)
            eval, _ = minimax(new_board, depth - 1, -player, alpha, beta)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

# AI の手を決定
def get_ai_move(board, player, depth=3):
    _, move = minimax(board, depth, player, float('-inf'), float('inf'))
    return move

# メイン処理
def main():
    board = initial_board()
    player = -1  # 黒が先手
    
    while True:
        valid_moves = get_valid_moves(board, player)
        if not valid_moves:
            player *= -1
            if not get_valid_moves(board, player):
                break
            continue
        
        if player == -1:
            move = get_ai_move(board, player)
        else:
            move = get_ai_move(board, player)
        
        if move:
            board = apply_move(board, move[0], move[1], player)
        player *= -1
    
    print("Game Over")
    print("Final Board:")
    print(board)

if __name__ == "__main__":
    main()
