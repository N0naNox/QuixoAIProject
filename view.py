# ==============================================================================
# GRAPHIC VIEW  (tkinter)
# ==============================================================================

import tkinter as tk
from tkinter import messagebox
from typing import Callable, List, Optional

from game import Game

# ── Colour palette ─────────────────────────────────────────────────────────────
BG_WINDOW   = "#1e1e2e"   # outer window background
BG_BOARD    = "#313244"   # board frame background
BG_PERIM    = "#45475a"   # default colour for clickable perimeter cells
BG_INNER    = "#585b70"   # default colour for inner (non-clickable) cells
BG_X        = "#89b4fa"   # light-blue for X pieces
BG_O        = "#a6e3a1"   # light-green for O pieces
FG_DARK     = "#1e1e2e"   # text on coloured cells
FG_DOT      = "#cdd6f4"   # dot / empty cell text colour
BTN_RESET   = "#f38ba8"   # reset button


class QuixoGameView:
    """
    Pure-tkinter view for the Quixo board.

    Public interface (used by the controller):
      • reset_view()
      • update_button(row, col, player_mark)   – player_mark: "X", "O", or ""
      • set_click_callback(fn)                 – fn(row, col)
      • set_reset_callback(fn)
      • show_message(title, text)
      • show_warning(title, text)
      • disable_board()  /  enable_board()
      • show_opening_screen()
      • hide_opening_screen()
      • root  – the tk.Tk root window (call root.mainloop() from outside)
    """

    def __init__(self, root: Optional[tk.Tk] = None):
        # Allow the caller to pass in an existing Tk root, or create one here.
        self.root = root if root is not None else tk.Tk()
        self.root.title("VLadimir's Quixo – Agent (X) vs Manual (O)")
        self.root.resizable(False, False)
        self.root.configure(bg=BG_WINDOW)

        self._click_callback = None   # set externally via set_click_callback()
        self._reset_callback = None   # set externally via set_reset_callback()
        self._start_callback = None   # set externally via set_start_callback()
        self._agent_var = tk.StringVar(value="NN")  # selected agent type

        self.buttons: List[List[tk.Button]] = []
        self.opening_frame = None
        self.game_frame = None

        self._build_opening_screen()
        self._build_game_ui()

    # ── Opening Screen ────────────────────────────────────────────────────────

    def _build_opening_screen(self):
        """Build the opening screen with title, instructions, and agent selection."""
        self.opening_frame = tk.Frame(self.root, bg=BG_WINDOW)
        
        # Title
        tk.Label(
            self.opening_frame,
            text="Q U I X O",
            font=("Helvetica", 32, "bold"),
            bg=BG_WINDOW,
            fg="#cdd6f4",
        ).pack(pady=(40, 20))

        tk.Label(
            self.opening_frame,
            text="AI Agent vs Human",
            font=("Helvetica", 16),
            bg=BG_WINDOW,
            fg="#a6adc8",
        ).pack(pady=(0, 40))

        # Instructions button
        tk.Button(
            self.opening_frame,
            text="📖 How to Play",
            font=("Helvetica", 12, "bold"),
            bg="#89b4fa",
            fg="#1e1e2e",
            activebackground="#74c7ec",
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2",
            command=self._show_instructions,
        ).pack(pady=(0, 30))

        # Agent selection
        tk.Label(
            self.opening_frame,
            text="Choose AI Agent:",
            font=("Helvetica", 14, "bold"),
            bg=BG_WINDOW,
            fg="#cdd6f4",
        ).pack(pady=(0, 10))

        agent_frame = tk.Frame(self.opening_frame, bg=BG_WINDOW)
        agent_frame.pack(pady=(0, 30))

        agents = [
            ("Neural Network (Strongest)", "NN"),
            ("Heuristic Agent", "HEURISTIC"), 
            ("Greedy Agent", "GREEDY"),
            ("Random Agent (Easiest)", "RANDOM")
        ]

        for text, value in agents:
            tk.Radiobutton(
                agent_frame,
                text=text,
                variable=self._agent_var,
                value=value,
                font=("Helvetica", 11),
                bg=BG_WINDOW,
                fg="#cdd6f4",
                selectcolor=BG_WINDOW,
                activebackground=BG_WINDOW,
                activeforeground="#89b4fa",
                command=self._on_agent_change,
            ).pack(anchor="w", padx=20, pady=2)

        # Start game button
        tk.Button(
            self.opening_frame,
            text="🎮 Start Game",
            font=("Helvetica", 14, "bold"),
            bg="#a6e3a1",
            fg="#1e1e2e",
            activebackground="#94e2cd",
            relief="flat",
            padx=30,
            pady=12,
            cursor="hand2",
            command=self._on_start_game,
        ).pack(pady=(20, 40))

    def _build_game_ui(self):
        """Build the game UI (board and controls)."""
        self.game_frame = tk.Frame(self.root, bg=BG_WINDOW)
        
        self._build_title(self.game_frame)
        self._build_board(self.game_frame)
        self._build_controls(self.game_frame)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        """Build all UI elements."""
        self._build_title()
        self._build_board()
        self._build_controls()

    def _build_title(self, parent=None):
        """Header label shown above the board."""
        parent = parent or self.root
        tk.Label(
            parent,
            text="Q U I X O",
            font=("Helvetica", 22, "bold"),
            bg=BG_WINDOW,
            fg="#cdd6f4",
        ).pack(pady=(14, 4))

        self.status_label = tk.Label(
            parent,
            text="Your turn  (O)",
            font=("Helvetica", 11),
            bg=BG_WINDOW,
            fg="#a6adc8",
        )
        self.status_label.pack(pady=(0, 8))

    def _build_board(self, parent=None):
        """5×5 grid of buttons inside a padded frame."""
        parent = parent or self.root
        board_frame = tk.Frame(parent, bg=BG_BOARD, padx=8, pady=8, relief="flat")
        board_frame.pack(padx=16, pady=4)

        for row in range(5):
            row_buttons: List[tk.Button] = []
            for col in range(5):
                is_perimeter = (row in (0, 4) or col in (0, 4))
                btn = tk.Button(
                    board_frame,
                    text="•" if is_perimeter else "",
                    width=4,
                    height=2,
                    font=("Helvetica", 18, "bold"),
                    bg=BG_PERIM if is_perimeter else BG_INNER,
                    fg=FG_DOT,
                    activebackground="#585b70",
                    relief="flat",
                    cursor="hand2" if is_perimeter else "arrow",
                    command=lambda r=row, c=col: self.handle_button_click(r, c),
                )
                btn.grid(row=row, column=col, padx=3, pady=3)
                row_buttons.append(btn)
            self.buttons.append(row_buttons)

    def _build_controls(self, parent=None):
        """Reset button below the board."""
        parent = parent or self.root
        ctrl = tk.Frame(parent, bg=BG_WINDOW)
        ctrl.pack(pady=12)

        self.reset_button = tk.Button(
            ctrl,
            text="↺  New Game",
            font=("Helvetica", 11, "bold"),
            bg=BTN_RESET,
            fg="#1e1e2e",
            activebackground="#eb6f92",
            relief="flat",
            padx=14,
            pady=6,
            cursor="hand2",
            command=self._on_reset,
        )
        self.reset_button.pack()

    # ── Callback wiring ────────────────────────────────────────────────────────

    def set_click_callback(self, fn):
        """Register a function to be called with (row, col) on every cell click."""
        self._click_callback = fn

    def set_reset_callback(self, fn):
        """Register a function to be called when the reset button is pressed."""
        self._reset_callback = fn

    def set_start_callback(self, fn):
        """Register a function to be called when start game is pressed, with selected agent."""
        self._start_callback = fn

    def _on_start_game(self):
        """Handle start game button click."""
        if self._start_callback:
            agent = self._agent_var.get()
            self._start_callback(agent)

    def _on_agent_change(self):
        """Handle agent selection change."""
        pass  # Could add visual feedback if needed

    def _on_reset(self):
        self.reset_view()
        if self._reset_callback:
            self._reset_callback()

    def _show_instructions(self):
        """Show game instructions in a popup."""
        instructions = """QUIXO GAME RULES:

• 5x5 board game for two players (X and O)
• Players take turns moving pieces on the board
• You can only select pieces from the perimeter (edges) that are either empty or marked with your symbol
• After selecting a piece, push it back into the board from a perpendicular edge
• The row/column slides in the direction you choose
• Win by getting 5 of your pieces in a row (horizontal, vertical, or diagonal)
• NO TIES - the game continues until someone gets 5 in a row

HOW TO PLAY:
1. Click on a perimeter cell (edge of the board)
2. If it's a corner, choose which direction to push
3. If it's an edge (not corner), it will automatically push in the only available direction
4. The AI (X) always goes first
5. You play as O

AI AGENTS:
• Neural Network: Advanced AI trained with reinforcement learning
• Heuristic: Uses strategic evaluation of board positions  
• Greedy: Chooses moves based on learned board values
• Random: Makes completely random moves (easiest opponent)"""
        
        # Create a scrollable text popup
        popup = tk.Toplevel(self.root)
        popup.title("How to Play Quixo")
        popup.resizable(False, False)
        popup.configure(bg=BG_WINDOW)
        popup.grab_set()

        # Instructions text in a scrollable frame
        text_frame = tk.Frame(popup, bg=BG_WINDOW)
        text_frame.pack(padx=20, pady=20)

        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=("Helvetica", 10),
            bg=BG_WINDOW,
            fg="#cdd6f4",
            height=20,
            width=60,
            relief="flat",
        )
        text_widget.insert(tk.END, instructions)
        text_widget.config(state=tk.DISABLED)  # Make it read-only
        
        scrollbar = tk.Scrollbar(text_frame, command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Close button
        tk.Button(
            popup,
            text="Close",
            font=("Helvetica", 11, "bold"),
            bg="#f38ba8",
            fg="#1e1e2e",
            activebackground="#eb6f92",
            relief="flat",
            padx=20,
            pady=8,
            cursor="hand2",
            command=popup.destroy,
        ).pack(pady=(0, 20))

    # ── Screen management ─────────────────────────────────────────────────────

    def show_opening_screen(self):
        """Show the opening screen and hide the game."""
        if self.game_frame:
            self.game_frame.pack_forget()
        self.opening_frame.pack(expand=True, fill=tk.BOTH)

    def hide_opening_screen(self):
        """Hide the opening screen and show the game."""
        self.opening_frame.pack_forget()
        self.game_frame.pack(expand=True, fill=tk.BOTH)

    # ── Public view-update API ─────────────────────────────────────────────────

    def update_button(self, row: int, col: int, player_mark: str):
        """
        Refresh one cell's appearance.
        player_mark: "X"  → blue
                     "O"  → green
                     ""   → default perimeter / inner colour
        """
        btn = self.buttons[row][col]
        is_perimeter = (row in (0, 4) or col in (0, 4))

        if player_mark == "X":
            btn.config(text="X", bg=BG_X, fg=FG_DARK)
        elif player_mark == "O":
            btn.config(text="O", bg=BG_O, fg=FG_DARK)
        else:
            btn.config(
                text="•" if is_perimeter else "",
                bg=BG_PERIM if is_perimeter else BG_INNER,
                fg=FG_DOT,
            )

    def reset_view(self):
        """Return the board to its initial empty state."""
        for row in range(5):
            for col in range(5):
                self.update_button(row, col, "")
        self.set_status("Your turn  (O)")

    def set_status(self, text: str):
        """Update the status line beneath the title."""
        self.status_label.config(text=text)

    def disable_board(self):
        """Grey-out all buttons (e.g. while the AI is thinking)."""
        for row in self.buttons:
            for btn in row:
                btn.config(state=tk.DISABLED)

    def enable_board(self):
        """Re-enable only the perimeter buttons for the human player."""
        for r, row in enumerate(self.buttons):
            for c, btn in enumerate(row):
                if r in (0, 4) or c in (0, 4):
                    btn.config(state=tk.NORMAL)

    def highlight_button(self, row: int, col: int):
        """Visually mark a cell as selected (gold outline effect)."""
        self.buttons[row][col].config(bg="#f9e2af", fg="#1e1e2e")

    def unhighlight_all(self):
        """Remove any selection highlight, restoring each cell's natural colour."""
        for r, row_btns in enumerate(self.buttons):
            for c, btn in enumerate(row_btns):
                # Only touch cells that are still in their default state
                current_text = btn.cget("text")
                if current_text not in ("X", "O"):
                    is_perimeter = r in (0, 4) or c in (0, 4)
                    btn.config(
                        bg=BG_PERIM if is_perimeter else BG_INNER,
                        fg=FG_DOT,
                    )

    def show_message(self, title: str, text: str):
        """Display an info popup."""
        messagebox.showinfo(title, text, parent=self.root)

    def show_warning(self, title: str, text: str):
        """Display a warning popup."""
        messagebox.showwarning(title, text, parent=self.root)

    # ── Internal click dispatcher ──────────────────────────────────────────────

    def handle_button_click(self, row: int, col: int):
        """Dispatch a cell click to the registered callback (or show a demo popup)."""
        if self._click_callback:
            self._click_callback(row, col)
        else:
            # Stand-alone demo behaviour
            if row in (0, 4) or col in (0, 4):
                self.show_message("Cell clicked", f"Perimeter cell ({row}, {col})")
            else:
                self.show_warning("Invalid move", "Only perimeter cells can be selected.")


# ── Stand-alone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    view = QuixoGameView(root)
    root.mainloop()