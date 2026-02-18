import numpy as np
import json
import random


def hash_board(board):
    """Generate a consistent, readable string representation of the board."""
    return ''.join(board.flatten())


class Game:
    def __init__(self, play_mode='RANDOM', output_mode='SILENT', states_dict=None,
                 epsilon=0.1, unknown_score=0.5, discount_factor=0.9,
                 win_score=1.0, loss_score=0.0):
        self.play_mode = play_mode
        self.output_mode = output_mode
        self.states_dict = states_dict or {}
        self.epsilon = epsilon
        self.unknown_score = unknown_score
        self.discount_factor = discount_factor
        self.win_score = win_score
        self.loss_score = loss_score
        self.board = np.full((5, 5), ' ')
        self.current_player = 'X'
        self.board_history = []
        self.outcome = 'ONGOING'
        self.unknown_count = 0
        self.total_moves = 0

    def reset_game(self):
        """Reset the board and all state for a fresh game."""
        self.board = np.full((5, 5), ' ')
        self.current_player = 'X'
        self.board_history = []
        self.outcome = 'ONGOING'
        self.unknown_count = 0
        self.total_moves = 0

    def play(self):
        """Main game loop. Returns (scored_boards_dict, unknown_rate)."""
        self.board_history = []
        self.unknown_count = 0
        self.total_moves = 0

        while self.outcome == 'ONGOING':
            board_hash = hash_board(self.board)
            self.board_history.append(board_hash)
            self.total_moves += 1
            if board_hash not in self.states_dict:
                self.unknown_count += 1

            if self.current_player == 'X':
                if self.play_mode == 'GREEDY':
                    self.perform_greedy_agent_move()
                elif self.play_mode == 'HEURISTIC':
                    self.perform_heuristic_agent_move()
                else:
                    self.perform_random_agent_move()
            else:
                self.perform_random_agent_move()

            self.outcome = self.check_win()
            # Always switch player after a move (even if game ended)
            self.current_player = 'O' if self.current_player == 'X' else 'X'

        if self.output_mode != 'SILENT':
            self.print_board()
            self.print_result()

        unknown_rate = self.unknown_count / self.total_moves if self.total_moves > 0 else 0
        return self.score_boards(), unknown_rate

    # ── Agent move methods ──────────────────────────────────────────────

    def perform_random_agent_move(self):
        """Pick a random valid edge position and a random valid direction."""
        positions = self.get_valid_positions()
        row, col = random.choice(positions)
        directions = self.get_valid_directions(row, col)
        direction = random.choice(directions)
        self.make_move(row, col, direction)

    def perform_greedy_agent_move(self):
        """
        Evaluate all possible moves using the dictionary.
        With probability epsilon pick a random move; otherwise pick the best.
        """
        positions = self.get_valid_positions()
        move_scores = []
        for row, col in positions:
            for direction in self.get_valid_directions(row, col):
                board_copy = self.board.copy()
                self.make_move(row, col, direction)
                board_hash = hash_board(self.board)
                score = self.states_dict.get(board_hash, [self.unknown_score, 1])[0]
                move_scores.append(((row, col, direction), score))
                self.board = board_copy

        if random.random() < self.epsilon:
            move = random.choice(move_scores)[0]
        else:
            move_scores.sort(key=lambda x: x[1], reverse=True)
            move = move_scores[0][0]
        self.make_move(move[0], move[1], move[2])

    def perform_heuristic_agent_move(self):
        """
        Heuristic agent (epsilon = 0.5 recommended for training).
        Priority:
          1. Make a winning move if one exists.
          2. Block the opponent's winning move.
          3. Fall back to greedy/random with strategic position bonus.
        """
        my_positions = self.get_valid_positions()
        all_moves = []
        for row, col in my_positions:
            for direction in self.get_valid_directions(row, col):
                all_moves.append((row, col, direction))

        # 1. Check for a winning move
        for move in all_moves:
            board_copy = self.board.copy()
            self.make_move(*move)
            if self.check_win() == 'VICTORY_X':
                # Board already has the winning move applied – keep it
                return
            self.board = board_copy

        # 2. Check for blocking moves
        #    Temporarily switch to opponent to find their valid positions & moves
        saved_player = self.current_player
        self.current_player = 'O'
        opp_positions = self.get_valid_positions()  # Opponent's valid picks
        opponent_can_win = False
        for orow, ocol in opp_positions:
            for odir in self.get_valid_directions(orow, ocol):
                board_copy = self.board.copy()
                self.make_move(orow, ocol, odir)
                if self.check_win() == 'VICTORY_O':
                    opponent_can_win = True
                self.board = board_copy
                if opponent_can_win:
                    break
            if opponent_can_win:
                break
        self.current_player = saved_player

        if opponent_can_win:
            # Try each of our moves; pick one where opponent can no longer win
            for move in all_moves:
                board_copy = self.board.copy()
                self.make_move(*move)
                # Check opponent's options on the new board
                still_wins = False
                self.current_player = 'O'
                opp_positions2 = self.get_valid_positions()
                for orow, ocol in opp_positions2:
                    for odir in self.get_valid_directions(orow, ocol):
                        board_copy2 = self.board.copy()
                        self.make_move(orow, ocol, odir)
                        if self.check_win() == 'VICTORY_O':
                            still_wins = True
                        self.board = board_copy2
                        if still_wins:
                            break
                    if still_wins:
                        break
                self.current_player = saved_player
                if not still_wins:
                    # This move blocks – keep it (board already has the move applied)
                    return
                self.board = board_copy

        # 3. Greedy logic with strategic position bonus
        #    Evaluate all moves using dictionary, but give a small bonus to
        #    strategic positions (corners) when the dictionary score is unknown.
        strategic = {(0, 0), (0, 4), (4, 0), (4, 4)}
        STRATEGIC_BONUS = 0.05  # Small bonus for strategic positions

        move_scores = []
        for move in all_moves:
            board_copy = self.board.copy()
            self.make_move(*move)
            board_hash = hash_board(self.board)
            entry = self.states_dict.get(board_hash, None)
            if entry is not None:
                score = entry[0]
            else:
                # Unknown board – use unknown_score + strategic bonus
                score = self.unknown_score
                if (move[0], move[1]) in strategic:
                    score += STRATEGIC_BONUS
            move_scores.append((move, score))
            self.board = board_copy

        if random.random() < self.epsilon:
            move = random.choice(move_scores)[0]
        else:
            move_scores.sort(key=lambda x: x[1], reverse=True)
            move = move_scores[0][0]
        self.make_move(*move)

    # ── Board helpers ───────────────────────────────────────────────────

    def get_valid_positions(self):
        """Return edge positions the current player can pick (empty or own piece)."""
        return [(i, j) for i in range(5) for j in range(5)
                if (i == 0 or i == 4 or j == 0 or j == 4)
                and self.board[i, j] in {self.current_player, ' '}]

    def get_valid_directions(self, row, col):
        """Return the valid push directions for a given edge position."""
        directions = []
        if row == 0:
            directions.append("down")
        if row == 4:
            directions.append("up")
        if col == 0:
            directions.append("right")
        if col == 4:
            directions.append("left")
        return directions

    def make_move(self, row, col, direction):
        """Push a cube from (row, col) in the given direction."""
        piece = self.current_player
        if direction == "down":
            for r in range(row, 4):
                self.board[r, col] = self.board[r + 1, col]
            self.board[4, col] = piece
        elif direction == "up":
            for r in range(row, 0, -1):
                self.board[r, col] = self.board[r - 1, col]
            self.board[0, col] = piece
        elif direction == "right":
            for c in range(col, 4):
                self.board[row, c] = self.board[row, c + 1]
            self.board[row, 4] = piece
        elif direction == "left":
            for c in range(col, 0, -1):
                self.board[row, c] = self.board[row, c - 1]
            self.board[row, 0] = piece

    def check_win(self):
        """Check if any player has won."""
        for symbol in ['X', 'O']:
            for row in range(5):
                if all(self.board[row, col] == symbol for col in range(5)):
                    return 'VICTORY_X' if symbol == 'X' else 'VICTORY_O'
            for col in range(5):
                if all(self.board[row, col] == symbol for row in range(5)):
                    return 'VICTORY_X' if symbol == 'X' else 'VICTORY_O'
            if all(self.board[i, i] == symbol for i in range(5)):
                return 'VICTORY_X' if symbol == 'X' else 'VICTORY_O'
            if all(self.board[i, 4 - i] == symbol for i in range(5)):
                return 'VICTORY_X' if symbol == 'X' else 'VICTORY_O'
        return 'ONGOING'

    def score_boards(self):
        """Score all board states from this game based on outcome and discount."""
        scores = {}
        n = len(self.board_history)
        if self.outcome == 'VICTORY_X':
            final_score = self.win_score
        else:
            final_score = self.loss_score
        for i, board_hash in enumerate(self.board_history):
            score = (self.discount_factor ** (n - i - 1)) * final_score
            scores[board_hash] = score
        return scores

    def print_board(self):
        for row in self.board:
            print("|" + "|".join(row) + "|")

    def print_result(self):
        if self.outcome == 'VICTORY_X':
            print("X Wins!")
        elif self.outcome == 'VICTORY_O':
            print("O Wins!")
        else:
            print("Game ongoing.")
