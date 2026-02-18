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

        self.buttons: List[List[tk.Button]] = []

        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        """Build all UI elements."""
        self._build_title()
        self._build_board()
        self._build_controls()

    def _build_title(self):
        """Header label shown above the board."""
        tk.Label(
            self.root,
            text="Q U I X O",
            font=("Helvetica", 22, "bold"),
            bg=BG_WINDOW,
            fg="#cdd6f4",
        ).pack(pady=(14, 4))

        self.status_label = tk.Label(
            self.root,
            text="Your turn  (O)",
            font=("Helvetica", 11),
            bg=BG_WINDOW,
            fg="#a6adc8",
        )
        self.status_label.pack(pady=(0, 8))

    def _build_board(self):
        """5×5 grid of buttons inside a padded frame."""
        board_frame = tk.Frame(self.root, bg=BG_BOARD, padx=8, pady=8, relief="flat")
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

    def _build_controls(self):
        """Reset button below the board."""
        ctrl = tk.Frame(self.root, bg=BG_WINDOW)
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

    def _on_reset(self):
        self.reset_view()
        if self._reset_callback:
            self._reset_callback()

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