import json
import os
import tkinter as tk

from game import Game
from view import QuixoGameView
from quixoNet import load_network, predict_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split


# ==============================================================================
# CONTROLLER
# ==============================================================================

class MyGameController:
    """
    Connects QuixoGameView ↔ Game.

    Turn order: X (AI) always goes first.
    Human plays O.

    Click flow for the human:
      1st click – select a valid perimeter cell  (empty or 'O').
      • The possible positions where the cube can move to are highlighted in pink.
      • 2nd click – click on a highlighted position to execute the move.
    """

    def __init__(self, model: Game, view: QuixoGameView):
        self._model = model
        self._view = view
        self._selected = None   # (row, col) of the cell the human has tapped first
        self._possible_positions = {}  # (row, col) -> direction for highlighted positions

        self._connect_signals()
        self.show_opening_screen()

    # ── Wiring ─────────────────────────────────────────────────────────────────

    def _connect_signals(self):
        self._view.set_click_callback(self._handle_human_move)
        self._view.set_reset_callback(self.start_new_game)
        self._view.set_start_callback(self._handle_start_game)

    def show_opening_screen(self):
        """Show the opening screen."""
        self._view.show_opening_screen()

    def _handle_start_game(self, agent_type: str):
        """Handle start game with selected agent type."""
        # Update the model's play mode
        self._model.play_mode = agent_type
        # Hide opening screen and start the game
        self._view.hide_opening_screen()
        self.start_new_game()
        self._view.set_start_callback(self._handle_start_game)

    # ── Game lifecycle ─────────────────────────────────────────────────────────

    def start_new_game(self):
        """Reset model and view, then let X (AI) take the first move."""
        self._selected = None
        self._possible_positions.clear()
        self._model.reset_game()          # board empty, current_player = 'X'
        self._view.reset_view()           # clears all buttons, status → 'Your turn (O)'
        self._view.disable_board()        # lock the board while AI thinks
        
        # Update status based on agent type
        agent_names = {
            'NN': 'Neural Network',
            'HEURISTIC': 'Heuristic Agent', 
            'GREEDY': 'Greedy Agent',
            'RANDOM': 'Random Agent'
        }
        agent_name = agent_names.get(self._model.play_mode, self._model.play_mode)
        self._view.set_status(f"{agent_name} is thinking…  (X)")
        
        # Give tkinter a moment to render before the AI blocks
        self._view.root.after(600, self._handle_ai_move)

    # ── AI turn ────────────────────────────────────────────────────────────────

    def _handle_ai_move(self):
        """Run one AI move, sync the view, then hand control to the human."""
        self._model.current_player = 'X'

        if self._model.play_mode == 'NN':
            self._model.perform_nn_agent_move()
        elif self._model.play_mode == 'HEURISTIC':
            self._model.perform_heuristic_agent_move()
        elif self._model.play_mode == 'GREEDY':
            self._model.perform_greedy_agent_move()
        else:
            self._model.perform_random_agent_move()

        self._sync_board_view()

        if self._check_game_over():
            return

        self._model.current_player = 'O'
        self._view.enable_board()
        self._view.set_status("Your move  (O)")

    # ── Human turn ─────────────────────────────────────────────────────────────

    def _handle_human_move(self, row: int, col: int):
        """Handle a click on the board while it is the human's turn."""
        # Check if clicking on a highlighted possible position
        if (row, col) in self._possible_positions:
            direction = self._possible_positions[(row, col)]
            self._execute_human_move(self._selected[0], self._selected[1], direction)
            return

        # Inner cells are never valid in Quixo
        if not (row in (0, 4) or col in (0, 4)):
            self._view.show_warning("Invalid move", "Only perimeter cells can be selected.")
            return

        # Cannot pick the AI's pieces
        if self._model.board[row, col] == 'X':
            self._view.show_warning("Invalid move", "You cannot pick a cell occupied by X.")
            return

        # Clicking the same cell a second time deselects it
        if self._selected == (row, col):
            self._selected = None
            self._view.unhighlight_all()
            self._view.unhighlight_possible()
            return

        # Clear any previous selection highlight
        if self._selected is not None:
            self._view.unhighlight_all()
            self._view.unhighlight_possible()

        # Select this cell
        self._selected = (row, col)
        self._view.highlight_button(row, col)

        dirs = self._model.get_valid_directions(row, col)
        # Highlight possible positions
        self._possible_positions = {}
        positions = []
        for d in dirs:
            if d == "up":
                pos = (0, col)
            elif d == "down":
                pos = (4, col)
            elif d == "left":
                pos = (row, 0)
            elif d == "right":
                pos = (row, 4)
            self._possible_positions[pos] = d
            positions.append(pos)
        self._view.highlight_possible(positions)

    def _show_direction_picker(self, row: int, col: int, dirs: list):
        """Modal popup with a directional-pad layout for choosing a push direction."""
        popup = tk.Toplevel(self._view.root)
        popup.title("Push direction")
        popup.resizable(False, False)
        popup.configure(bg="#0f0f23")
        popup.grab_set()   # make it modal

        tk.Label(
            popup,
            text="Choose a direction to push:",
            font=("Helvetica", 11),
            bg="#0f0f23",
            fg="#e0e0e0",
        ).pack(pady=(14, 6), padx=20)

        # D-pad grid: up=row0/col1, left=row1/col0, right=row1/col2, down=row2/col1
        DPAD = {
            "up":    (0, 1, "↑"),
            "left":  (1, 0, "←"),
            "right": (1, 2, "→"),
            "down":  (2, 1, "↓"),
        }

        pad_frame = tk.Frame(popup, bg="#0f0f23")
        pad_frame.pack(padx=24, pady=(4, 16))

        def pick(direction):
            popup.destroy()
            self._execute_human_move(row, col, direction)

        for d in dirs:
            grid_row, grid_col, symbol = DPAD[d]
            tk.Button(
                pad_frame,
                text=symbol,
                font=("Helvetica", 16, "bold"),
                width=3,
                height=1,
                bg="#00d4ff",
                fg="#ffffff",
                activebackground="#0088aa",
                relief="flat",
                cursor="hand2",
                command=lambda d=d: pick(d),
            ).grid(row=grid_row, column=grid_col, padx=4, pady=4)

        def on_close():
            """User closed the popup without choosing – deselect the cell."""
            self._selected = None
            self._view.unhighlight_all()
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", on_close)

    def _execute_human_move(self, row: int, col: int, direction: str):
        """Apply the validated human move, then schedule the AI's response."""
        self._selected = None
        self._possible_positions.clear()
        self._view.unhighlight_all()
        self._view.unhighlight_possible()

        self._model.current_player = 'O'
        self._model.make_move(row, col, direction)
        self._sync_board_view()

        if self._check_game_over():
            return

        self._model.current_player = 'X'
        self._view.disable_board()
        self._view.set_status("AI is thinking…  (X)")
        # Short delay so the board re-renders before the AI computation
        self._view.root.after(400, self._handle_ai_move)

    # ── Shared helpers ─────────────────────────────────────────────────────────

    def _sync_board_view(self):
        """Push the entire model board state into the view (necessary after any slide move)."""
        for r in range(5):
            for c in range(5):
                mark = self._model.board[r, c]   # ' ', 'X', or 'O'
                self._view.update_button(r, c, mark if mark != ' ' else "")

    def _check_game_over(self) -> bool:
        """Return True and show a result message if the game has ended."""
        outcome = self._model.check_win()
        if outcome == 'ONGOING':
            return False
        if outcome == 'VICTORY_X':
            self._view.set_status("Agent (X) wins!")
            self._view.show_message("Game Over", "Agent (X) wins! 🤖")
        else:
            self._view.set_status("You (O) win! 🎉")
            self._view.show_message("Game Over", "You (O) win! 🎉")
        self._view.disable_board()
        return True


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":

    #Load the strongest available dictionary
    states_dict = {}
    for filename in ('states_heuristic.json', 'states_greedy.json', 'states_random.json'):
        if os.path.exists(filename):
            with open(filename) as f:
                raw = json.load(f)
            # JSON keys are strings; values may be [score, count] or just score
            for key, val in raw.items():
                score = val[0] if isinstance(val, list) else val
                states_dict[key] = [score, 1]
            print(f"Loaded {len(states_dict)} board states from {filename}")
            break

    root = tk.Tk()

    game_model = Game(
        play_mode='NN',   # Default, will be changed by user selection
        output_mode='SILENT',
        states_dict=states_dict,
        model=load_network('quixo_model.pth', device=torch.device("cpu"))
    )
    game_view = QuixoGameView(root)

    controller = MyGameController(model=game_model, view=game_view)

    root.mainloop()