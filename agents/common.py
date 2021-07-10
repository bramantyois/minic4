from enum import Enum
from typing import Optional, Callable, Tuple
import numpy as np
from scipy.signal import convolve2d

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    return np.zeros(shape=(6, 7), dtype=BoardPiece)


def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output:
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """
    rows, cols = board.shape

    ret_str = '|' + '='*cols*2 + '|\n'

    for r in range(rows):
        cur_row = '|'
        for c in range(cols):
            if board[r, c] == NO_PLAYER:
                cur_row += NO_PLAYER_PRINT
            elif board[r, c] == PLAYER1:
                cur_row += PLAYER1_PRINT
            else:
                cur_row += PLAYER2_PRINT
            cur_row += ' '
        cur_row += '|\n'
        ret_str += cur_row

    ret_str += '|' + '=' * cols * 2 + '|\n'

    cols_str = np.arange(0, cols, 1).tolist()
    cols_num = '|'
    for s in cols_str:
        cols_num += str(s)
        cols_num += ' '
    cols_num += '|'
    ret_str += cols_num

    return ret_str


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    split = pp_board.splitlines()
    num_rows = len(split) - 3
    num_cols = int(0.5*(len(split[0])-2))

    ret = np.zeros((num_rows, num_cols))

    mat_split = split[1:-2]

    for row, r_str in enumerate(mat_split):
        splat = r_str[1:-1]

        for col in range(num_cols):
            c_str = splat[col*2]

            if c_str == NO_PLAYER_PRINT:
                ret[row, col] = NO_PLAYER
            elif c_str == PLAYER1_PRINT:
                ret[row, col] = PLAYER1
            else:
                ret[row, col] = PLAYER2

    return ret
    # raise NotImplementedError()


def apply_player_action(
    board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """

    if copy:
        ret = board.copy()
    else:
        ret = board

    valid = ret[0, action] == NO_PLAYER

    if valid:
        zero = np.max(np.where(ret[:, action] == NO_PLAYER))
        ret[zero, action] = player
        return ret
    else:
        return ret


def connected_four(
    board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    """

    board_bin = np.array(board == player, dtype=int)

    kernel_v = np.ones((4, 1), player)
    conv_v = convolve2d(kernel_v, board_bin, mode='valid')
    if (conv_v == 4).any():
        return True

    kernel_h = np.ones((1, 4), player)
    conv_h = convolve2d(kernel_h, board_bin, mode='valid')
    if (conv_h == 4).any():
        return True

    kernel_dl = np.zeros((4, 4), dtype=int)
    np.fill_diagonal(kernel_dl, val=1)
    conv_dl = convolve2d(kernel_dl, board_bin, mode='valid')
    if (conv_dl == 4).any():
        return True

    kernel_dr = np.flipud(kernel_dl)
    conv_dr = convolve2d(kernel_dr, board_bin, mode='valid')
    if (conv_dr == 4).any():
        return True

    return False


def check_end_state(
    board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    if connected_four(board, player, last_action):
        return GameState.IS_WIN
    if not (board == NO_PLAYER).any():
        return GameState.IS_DRAW

    return GameState.STILL_PLAYING

def check_valid_action(
        board: np.ndarray, action: PlayerAction) -> bool:
    """
    checking if given action is valid

    :param board: current board state
    :param action: action to be performed on the board
    :return: True if action is valid
    """

    return board[0, action] == NO_PLAYER
