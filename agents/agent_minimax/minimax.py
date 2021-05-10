import numpy as np
from typing import Optional, Tuple
from agents.common import BoardPiece, SavedState, PlayerAction, PLAYER1, PLAYER2, NO_PLAYER
from agents.common import apply_player_action, check_end_state
from agents.common import GameState
from scipy.signal import convolve2d
import math


def get_minimax_heuristic(board: np.ndarray) -> float:
    """
    :param board: current board state
    :return: heuristic value
    """
    board_tr = board.copy()
    board_tr[board_tr == PLAYER1] = 1
    board_tr[board_tr == PLAYER2] = -1

    kernel_v = np.ones((4, 1))
    conv_v = convolve2d(kernel_v, board_tr, mode='valid').sum()

    kernel_h = np.ones((1, 4))
    conv_h = convolve2d(kernel_h, board_tr, mode='valid').sum()

    kernel_dl = np.zeros((4, 4), dtype=int)
    np.fill_diagonal(kernel_dl, val=1)
    conv_dl = convolve2d(kernel_dl, board_tr, mode='valid').sum()

    kernel_dr = np.flipud(kernel_dl)
    conv_dr = convolve2d(kernel_dr, board_tr, mode='valid').sum()

    return conv_v + conv_h + conv_dl + conv_dr


def minimax(
        board: np.ndarray,
        depth: int,
        alpha: float,
        beta: float,
        player: BoardPiece) -> (float, PlayerAction):
    """

    :param board: board: current board state
    :param depth: depth of the node
    :param player: maximizing or minimizing player
    :return:

    Assuming player 1 always maximizing and player 2 minimizing
    """
    move = -1
    if player == PLAYER1:
        value = -math.inf
    else:
        value = math.inf

    if depth == 0 or (check_end_state(board, player) != GameState.STILL_PLAYING):
        return get_minimax_heuristic(board), move

    available_node = np.argwhere(board[0, :] == NO_PLAYER, ).tolist()

    if player == PLAYER1:
        for node in available_node:
            new_board = apply_player_action(board, node, player, True)
            new_val, _ = minimax(new_board, depth-1, alpha, beta, PLAYER2)
            if new_val > value:
                value = new_val
                move = node
            alpha = max(value, alpha)
            if alpha >= beta:
                break
    else:
        for node in available_node:
            new_board = apply_player_action(board, node, player, True)
            new_val, _ = minimax(new_board, depth-1, alpha, beta, PLAYER1)
            if new_val < value:
                value = new_val
                move = node
            beta = min(value, beta)
            if beta <= alpha:
                break

    return value, move


def generate_move_minimax(
        board: np.ndarray,
        player: BoardPiece,
        saved_state: Optional[SavedState],
        depth: int = 4) -> Tuple[PlayerAction, Optional[SavedState]]:
    """

    :param board:
    :param player:
    :param saved_state:
    :param depth: depth of the search tree
    :return:
    """

    _, action = minimax(board, depth, math.inf, -math.inf, player)
    return action, saved_state
