import numpy as np
import copy
import time
import random

# オセロのボードのサイズ
BOARD_SIZE = 8

# 方向ベクトル
DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),         (0, 1),
    (1, -1), (1, 0), (1, 1)
]

# ボードの初期化
def initial_board():
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    mid = BOARD_SIZE // 2
    board[mid - 1][mid - 1] = board[mid][mid] = 1  # 白
    board[mid - 1][mid] = board[mid][mid - 1] = -1  # 黒
    return board

# 有効な手を取得
def get_valid_moves(board, player):
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

# モンテカルロ法のAI
def monte_carlo_ai(board, player, simulations=100):
    valid_moves = get_valid_moves(board, player)
    if not valid_moves:
        return None
    
    move_scores = {move: 0 for move in valid_moves}
    for move in valid_moves:
        for _ in range(simulations):
            sim_board = apply_move(board, move[0], move[1], player)
            current_player = -player
            while get_valid_moves(sim_board, current_player):
                random_move = random.choice(get_valid_moves(sim_board, current_player))
                sim_board = apply_move(sim_board, random_move[0], random_move[1], current_player)
                current_player *= -1
            move_scores[move] += np.sum(sim_board) * player
    
    return max(move_scores, key=move_scores.get)

# ミニマックス法のAI
def minimax(board, depth, player, alpha, beta):
    valid_moves = get_valid_moves(board, player)
    if depth == 0 or not valid_moves:
        return np.sum(board) * player, None
    
    best_move = None
    if player == 1:
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
    else:
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

def get_minimax_move(board, player, depth=3):
    _, move = minimax(board, depth, player, float('-inf'), float('inf'))
    return move

def play_game():
    board = initial_board()
    player = -1  # 黒が先手
    start_time = time.time()
    
    while True:
        valid_moves = get_valid_moves(board, player)
        if not valid_moves:
            player *= -1
            if not get_valid_moves(board, player):
                break
            continue
        
        if player == -1:
            move = monte_carlo_ai(board, player, simulations=100)
        else:
            move = get_minimax_move(board, player, depth=3)
        
        if move:
            board = apply_move(board, move[0], move[1], player)
        player *= -1
    
    end_time = time.time()
    
    print("Game Over")
    print("Final Board:")
    print(board)
    score = np.sum(board)
    if score > 0:
        print("Minimax AI (White) Wins!")
    elif score < 0:
        print("Monte Carlo AI (Black) Wins!")
    else:
        print("It's a Draw!")
    print(f"Game Duration: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    play_game()
