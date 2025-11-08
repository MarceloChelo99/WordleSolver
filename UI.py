import pygame
from pygame.locals import QUIT

class PygameGrid:
    """
    Minimal 5x6 grid renderer for Wordle-like displays.

    Public API
    ----------
    - set_row(row_index: int, word: str, scores: Sequence[int])
        Place `word` (left aligned) into row `row_index` and apply `scores` (0/1/2).
    - draw()
        Redraw the screen immediately.
    - pump_once()
        Process one event loop iteration (use in host app loop to keep window responsive).
    - close()
        Quit pygame and close the window.

    The class avoids any game logic or input handling beyond a quit event.
    """

    # configuration (tweak as needed)
    ROWS, COLS = 6, 5
    CELL_SIZE = 80
    GAP = 10
    TOP = 40
    LEFT = 40
    BG = (18, 18, 19)
    CELL_BG = (250, 250, 250)
    CELL_BORDER = (30, 30, 30)
    TEXT_COLOR = (0, 0, 0)
    SCORE_COLORS = {
        0: (120, 124, 126),   # grey
        1: (201, 180, 88),    # yellow
        2: (106, 170, 100),   # green
    }
    FONT_NAME = "arial"
    FONT_SIZE = 48

    def __init__(self, window_title="5x6 Grid"):
        pygame.init()
        width = self.LEFT * 2 + self.COLS * self.CELL_SIZE + (self.COLS - 1) * self.GAP
        height = self.TOP * 2 + self.ROWS * self.CELL_SIZE + (self.ROWS - 1) * self.GAP
        self._screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(window_title)
        self._font = pygame.font.SysFont(self.FONT_NAME, self.FONT_SIZE)

        # grid state
        self.letters = [["" for _ in range(self.COLS)] for _ in range(self.ROWS)]
        self.colors = [[self.CELL_BG for _ in range(self.COLS)] for _ in range(self.ROWS)]
        self._running = True

    def _cell_rect(self, r: int, c: int) -> pygame.Rect:
        x = self.LEFT + c * (self.CELL_SIZE + self.GAP)
        y = self.TOP + r * (self.CELL_SIZE + self.GAP)
        return pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)

    def set_row(self, row: int, word: str, scores):
        """
        Set letters and colors for `row`.
        - row: 0-based row index (0 <= row < 6)
        - word: string placed left-aligned (characters beyond COLS are ignored)
        - scores: sequence of ints (0/1/2), length must be >= len(word)
        """
        if not (0 <= row < self.ROWS):
            raise IndexError("row out of range")
        # fill letters and colors for this row
        for c in range(self.COLS):
            if c < len(word):
                ch = word[c]
                self.letters[row][c] = ch
                score = int(scores[c]) if c < len(scores) else None
                self.colors[row][c] = self.SCORE_COLORS.get(score, self.CELL_BG)
            else:
                self.letters[row][c] = ""
                self.colors[row][c] = self.CELL_BG

    def draw(self):
        """Immediate redraw of the whole grid."""
        self._screen.fill(self.BG)
        for r in range(self.ROWS):
            for c in range(self.COLS):
                rect = self._cell_rect(r, c)
                pygame.draw.rect(self._screen, self.colors[r][c], rect, border_radius=6)
                pygame.draw.rect(self._screen, self.CELL_BORDER, rect, 2, border_radius=6)
                ch = self.letters[r][c]
                if ch:
                    txt = self._font.render(ch.upper(), True, self.TEXT_COLOR)
                    txt_r = txt.get_rect(center=rect.center)
                    self._screen.blit(txt, txt_r)
        pygame.display.flip()

    def pump_once(self):
        """Process events once; keeps window responsive. Returns False if window closed."""
        for ev in pygame.event.get():
            if ev.type == QUIT:
                self._running = False
        return self._running

    def close(self):
        """Close window and quit pygame."""
        pygame.quit()

if __name__ == "__main__":
    # Quick demo when run directly
    grid = PygameGrid()
    grid.draw()

    grid.set_row(0, "crane", [2, 0, 1, 0, 2])
    grid.draw()
    input()
    grid.set_row(1, "slate", [0, 1, 0, 2, 0])
    grid.draw()

    # simple loop to keep window open until closed by user
    running = True
    while running:
        running = grid.pump_once()
    grid.close()