# ==============================================================================
# GRAPHIC VIEW  (tkinter)
# ==============================================================================

import tkinter as tk
from tkinter import messagebox
from typing import Callable, List, Optional

from game import Game

# ── Colour palette ─────────────────────────────────────────────────────────────
BG_WINDOW   = "#0f0f23"   # deep dark blue outer window background
BG_BOARD    = "#1a1a2e"   # dark navy board frame background
BG_PERIM    = "#16213e"   # dark blue for clickable perimeter cells
BG_INNER    = "#0f3460"   # darker blue for inner (non-clickable) cells
BG_X        = "#00d4ff"   # neon cyan for X pieces
BG_O        = "#ff006e"   # neon pink for O pieces
BG_POSSIBLE = "#ffbe0b"   # neon yellow for possible move positions
FG_DARK     = "#ffffff"   # white text on coloured cells
FG_DOT      = "#e0e0e0"   # light gray dot / empty cell text colour
BTN_RESET   = "#ff006e"   # neon pink for reset button


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
        self._possible_positions = set()  # set of (row, col) currently highlighted as possible

        self.buttons: List[List[tk.Button]] = []
        self.opening_frame = None
        self.game_frame = None
        self.instructions_frame = None

        # Build game UI first to determine window size
        self._build_game_ui()
        self._setup_window_size()
        self._build_opening_screen()
        self._build_instructions_screen()
        
        # All frames now overlay; opening is shown initially
        self.show_opening_screen()

    # ── Setup window size based on game frame ────────────────────────────────

    def _setup_window_size(self):
        """Calculate and set window size based on game frame content."""
        self.root.update_idletasks()
        width = self.root.winfo_reqwidth()
        height = self.root.winfo_reqheight()
        self.root.geometry(f"{width}x{height}")

    # ── Opening Screen ────────────────────────────────────────────────────────

    def _build_opening_screen(self):
        """Build the opening screen with title, instructions, and agent selection."""
        self.opening_frame = tk.Frame(self.root, bg=BG_WINDOW)
        self.opening_frame.place(x=0, y=0, relwidth=1.0, relheight=1.0)

        # Centered card container with content
        card = tk.Frame(self.opening_frame, bg=BG_BOARD, padx=28, pady=28)
        card.pack(expand=True, pady=20)

        # Title
        tk.Label(
            card,
            text="Q U I X O",
            font=("Helvetica", 32, "bold"),
            bg=BG_BOARD,
            fg=FG_DOT,
        ).pack(pady=(8, 16))

        tk.Label(
            card,
            text="AI Agent vs Human",
            font=("Helvetica", 16),
            bg=BG_BOARD,
            fg=FG_DOT,
        ).pack(pady=(0, 24))

        # Instructions button
        tk.Button(
            card,
            text="📖 How to Play",
            font=("Helvetica", 12, "bold"),
            bg=BG_X,
            fg=FG_DARK,
            activebackground="#0088aa",
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2",
            command=self.show_instructions_screen,
        ).pack(pady=(0, 28))

        # Agent selection
        tk.Label(
            card,
            text="Choose AI Agent:",
            font=("Helvetica", 14, "bold"),
            bg=BG_BOARD,
            fg=FG_DOT,
        ).pack(pady=(0, 12))

        agent_frame = tk.Frame(card, bg=BG_BOARD)
        agent_frame.pack(pady=(0, 26))

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
                bg=BG_BOARD,
                fg=FG_DOT,
                selectcolor=BG_BOARD,
                activebackground=BG_BOARD,
                activeforeground=BG_X,
                command=self._on_agent_change,
            ).pack(anchor="w", padx=20, pady=2)

        # Start game button
        tk.Button(
            card,
            text="🎮 Start Game",
            font=("Helvetica", 14, "bold"),
            bg=BG_O,
            fg=FG_DARK,
            activebackground="#cc0055",
            relief="flat",
            padx=30,
            pady=12,
            cursor="hand2",
            command=self._on_start_game,
        ).pack(pady=(20, 18))

    def _build_game_ui(self):
        """Build the game UI (board and controls)."""
        self.game_frame = tk.Frame(self.root, bg=BG_WINDOW)
        self.game_frame.pack()  # Pack initially to measure size
        
        self._build_title(self.game_frame)
        self._build_board(self.game_frame)
        self._build_controls(self.game_frame)

    # ── UI construction ────────────────────────────────────────────────────────

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
            fg=FG_DOT,
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
                    activebackground="#00d4ff",
                    relief="flat",
                    cursor="hand2" if is_perimeter else "arrow",
                    command=lambda r=row, c=col: self.handle_button_click(r, c),
                )
                btn.grid(row=row, column=col, padx=3, pady=3)
                row_buttons.append(btn)
            self.buttons.append(row_buttons)

    def _build_controls(self, parent=None):
        """Reset button and menu button below the board."""
        parent = parent or self.root
        ctrl = tk.Frame(parent, bg=BG_WINDOW)
        ctrl.pack(pady=12)

        self.reset_button = tk.Button(
            ctrl,
            text="↺  New Game",
            font=("Helvetica", 11, "bold"),
            bg=BTN_RESET,
            fg=FG_DARK,
            activebackground="#cc0055",
            relief="flat",
            padx=14,
            pady=6,
            cursor="hand2",
            command=self._on_reset,
        )
        self.reset_button.pack(side=tk.LEFT, padx=6)

        tk.Button(
            ctrl,
            text="☰  Menu",
            font=("Helvetica", 11, "bold"),
            bg="#00d4ff",
            fg=FG_DARK,
            activebackground="#0088aa",
            relief="flat",
            padx=14,
            pady=6,
            cursor="hand2",
            command=self.show_menu_from_game,
        ).pack(side=tk.LEFT, padx=6)

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
        pass  

    def _on_reset(self):
        self.reset_view()
        if self._reset_callback:
            self._reset_callback()

    def _build_instructions_screen(self):
        """Build the instructions screen as an overlay."""
        self.instructions_frame = tk.Frame(self.root, bg=BG_WINDOW)
        self.instructions_frame.place(x=0, y=0, relwidth=1.0, relheight=1.0)

        # Centered card container
        card = tk.Frame(self.instructions_frame, bg=BG_BOARD, padx=20, pady=20)
        card.pack(expand=True, pady=10, padx=20, fill=tk.BOTH)

        # Title
        tk.Label(
            card,
            text="How to Play QUIXO",
            font=("Helvetica", 18, "bold"),
            bg=BG_BOARD,
            fg=FG_DOT,
        ).pack(pady=(0, 16))

        # Instructions text in a scrollable frame
        text_frame = tk.Frame(card, bg=BG_BOARD)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        instructions = """QUIXO GAME RULES:

• 5x5 board game for two players (X and O)
• Players take turns moving pieces
• Only select pieces from the perimeter (edges) that are empty or marked with your symbol
• After selecting, push the piece back into the board from a perpendicular edge
• The row/column slides in the direction you choose
• Win by getting 5 of your pieces in a row (horizontal, vertical, or diagonal)
• NO TIES - the game continues until someone wins

HOW TO PLAY:
1. Click on a perimeter cell (edge of the board)
2. If it's a corner, choose which direction to push
3. If it's an edge (not corner), it automatically pushes in the only available direction
4. The AI (X) always goes first
5. You play as O

AI AGENTS:
• Neural Network: Advanced AI trained with reinforcement learning
• Heuristic: Uses strategic evaluation of board positions  
• Greedy: Chooses moves based on learned board values
• Random: Makes completely random moves (easiest opponent)"""

        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=("Helvetica", 9),
            bg=BG_WINDOW,
            fg=FG_DOT,
            height=16,
            relief="flat",
            bd=0,
        )
        text_widget.insert(tk.END, instructions)
        text_widget.config(state=tk.DISABLED)
        
        scrollbar = tk.Scrollbar(text_frame, command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Back button
        tk.Button(
            card,
            text="← Back to Menu",
            font=("Helvetica", 11, "bold"),
            bg=BTN_RESET,
            fg=FG_DARK,
            activebackground="#cc0055",
            relief="flat",
            padx=20,
            pady=8,
            cursor="hand2",
            command=self.show_opening_screen,
        ).pack(pady=(0, 0))

    # ── Screen management ─────────────────────────────────────────────────────

    def show_opening_screen(self):
        """Show the opening screen and hide others."""
        self.opening_frame.tkraise()
        self.game_frame.pack_forget()
        self.instructions_frame.pack_forget()

    def show_instructions_screen(self):
        """Show the instructions screen and hide others."""
        self.instructions_frame.tkraise()
        self.game_frame.pack_forget()
        self.opening_frame.pack_forget()

    def hide_opening_screen(self):
        """Hide the opening screen and show the game."""
        self.game_frame.pack()
        self.game_frame.tkraise()
        self.opening_frame.pack_forget()
        self.instructions_frame.pack_forget()

    def show_menu_from_game(self):
        """Return to the opening menu from the game."""
        self.show_opening_screen()

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
        """Visually mark a cell as selected (neon yellow effect)."""
        self.buttons[row][col].config(bg="#ffbe0b", fg=FG_DARK)

    def unhighlight_all(self):
        """Remove any selection highlight, restoring each cell's natural colour."""
        for r, row_btns in enumerate(self.buttons):
            for c, btn in enumerate(row_btns):
                current_text = btn.cget("text")
                if current_text == "X":
                    btn.config(bg=BG_X, fg=FG_DARK)
                elif current_text == "O":
                    btn.config(bg=BG_O, fg=FG_DARK)
                else:
                    is_perimeter = r in (0, 4) or c in (0, 4)
                    btn.config(
                        bg=BG_PERIM if is_perimeter else BG_INNER,
                        fg=FG_DOT,
                    )

    def highlight_possible(self, positions):
        """Highlight positions that are possible moves with pink color."""
        self.unhighlight_possible()
        self._possible_positions = set(positions)
        for row, col in positions:
            btn = self.buttons[row][col]
            btn.config(bg=BG_POSSIBLE, fg=FG_DARK)

    def unhighlight_possible(self):
        """Remove highlights from possible positions, restoring their natural colors."""
        for row, col in self._possible_positions:
            btn = self.buttons[row][col]
            current_text = btn.cget("text")
            if current_text == "X":
                btn.config(bg=BG_X, fg=FG_DARK)
            elif current_text == "O":
                btn.config(bg=BG_O, fg=FG_DARK)
            else:
                is_perimeter = (row in (0, 4) or col in (0, 4))
                btn.config(
                    bg=BG_PERIM if is_perimeter else BG_INNER,
                    fg=FG_DOT,
                )
        self._possible_positions.clear()

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
            if row in (0, 4) or col in (0, 4):
                self.show_message("Cell clicked", f"Perimeter cell ({row}, {col})")
            else:
                self.show_warning("Invalid move", "Only perimeter cells can be selected.")


# ── Stand-alone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    view = QuixoGameView(root)
    root.mainloop()