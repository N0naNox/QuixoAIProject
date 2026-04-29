import numpy as np
import json
import random
from quixoNet import load_network, predict_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split


def other_player(player):
    return 'O' if player == 'X' else 'X'


def victory_for(player):
    return 'VICTORY_X' if player == 'X' else 'VICTORY_O'


def hash_board(board, current_player=None):
    """Generate a state key for the board, optionally including the player to move."""
    board_state = ''.join(board.flatten())
    if current_player is None:
        return board_state
    return f"{current_player}:{board_state}"


class Game:
    def __init__(self, play_mode='RANDOM', output_mode='SILENT', states_dict=None,
                 opponent_play_mode=None,
                 epsilon=0.1, unknown_score=0.5, discount_factor=0.9,
                 win_score=1.0, loss_score=0.0, draw_score=0.5, model=None):
        self.play_mode = play_mode
        self.opponent_play_mode = opponent_play_mode or play_mode
        self.output_mode = output_mode
        self.states_dict = states_dict or {}
        self.epsilon = epsilon
        self.model = model
        self.unknown_score = unknown_score
        self.discount_factor = discount_factor
        self.win_score = win_score
        self.loss_score = loss_score
        self.draw_score = draw_score
        self.board = np.full((5, 5), ' ')
        self.current_player = 'X'
        self.board_history = []
        self.outcome = 'ONGOING'
        self.unknown_count = 0
        self.total_moves = 0

    def get_active_play_mode(self):
        return self.play_mode if self.current_player == 'X' else self.opponent_play_mode

    def get_model_device(self):
        if self.model is None:
            return torch.device("cpu")
        return next(self.model.parameters()).device

    def lookup_state_entry(self, board, player_to_move):
        state_key = hash_board(board, player_to_move)
        entry = self.states_dict.get(state_key)
        if entry is not None:
            return entry
        return self.states_dict.get(hash_board(board))

    def reset_game(self):
        """Reset the board and all state for a fresh game."""
        self.board = np.full((5, 5), ' ')
        self.current_player = 'X'
        self.board_history = []
        self.outcome = 'ONGOING'
        self.unknown_count = 0
        self.total_moves = 0

    def play(self, max_moves=None):
        """Main game loop. Returns (scored_boards_dict, unknown_rate)."""
        self.board_history = []
        self.unknown_count = 0
        self.total_moves = 0

        while self.outcome == 'ONGOING':
            if max_moves is not None and self.total_moves >= max_moves:
                self.outcome = 'DRAW_MAX_MOVES'
                break

            board_hash = hash_board(self.board, self.current_player)
            self.board_history.append(board_hash)
            self.total_moves += 1
            if self.lookup_state_entry(self.board, self.current_player) is None:
                self.unknown_count += 1

            active_play_mode = self.get_active_play_mode()
            if active_play_mode == 'GREEDY':
                self.perform_greedy_agent_move()
            elif active_play_mode == 'NN':
                self.perform_nn_agent_move()
            elif active_play_mode == 'HEURISTIC':
                self.perform_heuristic_agent_move()
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
                next_player = other_player(self.current_player)
                entry = self.lookup_state_entry(self.board, next_player)
                score = entry[0] if entry is not None else self.unknown_score
                move_scores.append(((row, col, direction), score))
                self.board = board_copy

        if random.random() < self.epsilon:
            move = random.choice(move_scores)[0]
        else:
            move_scores.sort(key=lambda x: x[1], reverse=self.current_player == 'X')
            move = move_scores[0][0]
        self.make_move(move[0], move[1], move[2])

    def perform_heuristic_agent_move(self):
        """
                Heuristic agent with win/block/greedy priorities.
        Priority:
          1. Make a winning move if one exists.
          2. Block the opponent's winning move.
          3. Fall back to greedy/random with strategic position bonus.
        """
        current_player = self.current_player
        opponent = other_player(current_player)
        my_positions = self.get_valid_positions()
        all_moves = []
        for row, col in my_positions:
            for direction in self.get_valid_directions(row, col):
                all_moves.append((row, col, direction))

        # 1. Check for a winning move
        for move in all_moves:
            board_copy = self.board.copy()
            self.make_move(*move)
            if self.check_win() == victory_for(current_player):
                # Board already has the winning move applied – keep it
                return
            self.board = board_copy

        # 2. Check for blocking moves
        #    Temporarily switch to opponent to find their valid positions & moves
        saved_player = self.current_player
        self.current_player = opponent
        opp_positions = self.get_valid_positions()  # Opponent's valid picks
        opponent_can_win = False
        for orow, ocol in opp_positions:
            for odir in self.get_valid_directions(orow, ocol):
                board_copy = self.board.copy()
                self.make_move(orow, ocol, odir)
                if self.check_win() == victory_for(opponent):
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
                self.current_player = opponent
                opp_positions2 = self.get_valid_positions()
                for orow, ocol in opp_positions2:
                    for odir in self.get_valid_directions(orow, ocol):
                        board_copy2 = self.board.copy()
                        self.make_move(orow, ocol, odir)
                        if self.check_win() == victory_for(opponent):
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
            score = self.unknown_score 
            move_scores.append((move, score))
            self.board = board_copy

        if random.random() < self.epsilon:
            move = random.choice(move_scores)[0]
        else:
            move_scores.sort(key=lambda x: x[1], reverse=current_player == 'X')
            move = move_scores[0][0]
        self.make_move(*move)


    def perform_nn_agent_move(self):
        if self.model is None:
            self.perform_greedy_agent_move()
            return

        current_player = self.current_player
        opponent = other_player(current_player)
        model_device = self.get_model_device()
        my_positions = self.get_valid_positions()
        all_moves = []
        for row, col in my_positions:
            for direction in self.get_valid_directions(row, col):
                all_moves.append((row, col, direction))

        # 1. Check for a winning move
        for move in all_moves:
            board_copy = self.board.copy()
            self.make_move(*move)
            if self.check_win() == victory_for(current_player):
                # Board already has the winning move applied – keep it
                return
            self.board = board_copy

        # 2. Check for blocking moves
        #    Temporarily switch to opponent to find their valid positions & moves
        saved_player = self.current_player
        self.current_player = opponent
        opp_positions = self.get_valid_positions()  # Opponent's valid picks
        opponent_can_win = False
        for orow, ocol in opp_positions:
            for odir in self.get_valid_directions(orow, ocol):
                board_copy = self.board.copy()
                self.make_move(orow, ocol, odir)
                if self.check_win() == victory_for(opponent):
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
                self.current_player = opponent
                opp_positions2 = self.get_valid_positions()
                for orow, ocol in opp_positions2:
                    for odir in self.get_valid_directions(orow, ocol):
                        board_copy2 = self.board.copy()
                        self.make_move(orow, ocol, odir)
                        if self.check_win() == victory_for(opponent):
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
            
            score = predict_score(
                self.model,
                hash_board(self.board, other_player(self.current_player)),
                model_device,
            )

            move_scores.append((move, score))
            self.board = board_copy

        if random.random() < self.epsilon:
            move = random.choice(move_scores)[0]
        else:
            move_scores.sort(key=lambda x: x[1], reverse=current_player == 'X')
            move = move_scores[0][0]
        self.make_move(*move)
        

    # ── Board helpers ───────────────────────────────────────────────────

    def get_valid_positions(self):
        """Return edge positions the current player can pick (empty or own piece)."""
        return [(i, j) for i in range(5) for j in range(5)
                if (i == 0 or i == 4 or j == 0 or j == 4)
                and self.board[i, j] in {self.current_player, ' '}]

    def get_valid_directions(self, row, col):
        """Return the valid push directions for a given edge position.

        A piece can be pushed in any direction EXCEPT back towards the edge it
        already sits on.  This gives:
          • corners          → 2 directions
          • non-corner edges → 3 directions
        """
        directions = []
        if row > 0:   directions.append("up")
        if row < 4:   directions.append("down")
        if col > 0:   directions.append("left")
        if col < 4:   directions.append("right")
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
        elif self.outcome == 'VICTORY_O':
            final_score = self.loss_score
        else:
            final_score = self.draw_score
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
            print(f"Game ended without a winner: {self.outcome}")
