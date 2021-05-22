import numpy as np
from typing import Optional, Tuple
from agents.common import BoardPiece, SavedState, PlayerAction, PLAYER1, PLAYER2, NO_PLAYER
from agents.common import apply_player_action, check_end_state
from agents.common import GameState
from scipy.signal import convolve2d
import math

kernel_v = np.ones((4, 1))
kernel_h = np.ones((1, 4))
kernel_dl = np.zeros((4, 4), dtype=int)
np.fill_diagonal(kernel_dl, val=1)
kernel_dr = np.flipud(kernel_dl)


def compute_score(convolved_board: np.ndarray) -> float:
    """

    :param convolved_board: array of convolution values
    :return: score the heuristic respective to convolution kernel
    """
    v4 = np.sum(convolved_board == 4) * 9999
    v3 = np.sum(convolved_board == 3) * 999
    v2 = np.sum(convolved_board == 2) * 99
    v1 = np.sum(convolved_board == 1) * 9
    vm4 = -np.sum(convolved_board == -4) * 9999
    vm3 = -np.sum(convolved_board == -3) * 999
    vm2 = -np.sum(convolved_board == -2) * 99
    vm1 = -np.sum(convolved_board == -1) * 9

    return v4 + v3 + v2 + v1 + vm4 + vm3 + vm2 + vm1


def get_minimax_heuristic(board: np.ndarray) -> float:
    """
    :param board: current board state
    :return: heuristic value
    """
    board_tr = np.zeros_like(board)
    board_tr[board == PLAYER1] = 1
    board_tr[board == PLAYER2] = -1

    ret = 0
    conv_v = convolve2d(kernel_v, board_tr, mode='valid')
    ret += compute_score(conv_v)

    conv_h = convolve2d(kernel_h, board_tr, mode='valid')
    ret += compute_score(conv_h)

    conv_dl = convolve2d(kernel_dl, board_tr, mode='valid')
    ret += compute_score(conv_dl)

    conv_dr = convolve2d(kernel_dr, board_tr, mode='valid')
    ret += compute_score(conv_dr)

    return ret


def minimax(
        board: np.ndarray,
        depth: int,
        player: BoardPiece) -> (float, PlayerAction):
    """
    :param board: current board state
    :param depth: depth of the node
    :param player: maximizing or minimizing player
    :return:
    Assuming player 1 always maximizing and player 2 minimizing
    """
    move = -1
    if player == PLAYER1:
        value = -np.inf
    else:
        value = np.inf

    if depth == 0 or (check_end_state(board, player) != GameState.STILL_PLAYING):
        return get_minimax_heuristic(board), move

    available_node = np.where(board[0, :] == NO_PLAYER)[0]

    if player == PLAYER1:
        for node in available_node:
            new_board = apply_player_action(board, node, player, True)
            new_val, _ = minimax(new_board, depth - 1, PLAYER2)
            if new_val > value:
                value = new_val
                move = node
    else:
        for node in available_node:
            new_board = apply_player_action(board, node, player, True)
            new_val, _ = minimax(new_board, depth - 1, PLAYER1)
            if new_val < value:
                value = new_val
                move = node

    return value, move


def generate_move_minimax(
        board: np.ndarray,
        player: BoardPiece,
        saved_state: Optional[SavedState],
        depth: int = 2) -> Tuple[PlayerAction, Optional[SavedState]]:
    """

    :param board:
    :param saved_state:
    :param depth: depth of the search tree
    :return:
    """

    _, action = minimax(board, depth, player)
    return action, saved_state


def minimax_ab(
        board: np.ndarray,
        depth: int,
        alpha: float,
        beta: float,
        player: BoardPiece) -> (float, PlayerAction):
    """

    :param board:
    :param depth:
    :param alpha:
    :param beta:
    :param player:
    """

    move = -1

    if player == PLAYER1:
        value = -np.inf
    else:
        value = np.inf

    if depth == 0 or (check_end_state(board, player) != GameState.STILL_PLAYING):
        return get_minimax_heuristic(board), move

    available_node = np.where(board[0, :] == NO_PLAYER)[0]
    np.random.shuffle(available_node)

    if player == PLAYER1:
        for node in available_node:
            new_board = apply_player_action(board, node, player, True)
            new_val, _ = minimax_ab(new_board, depth - 1, alpha, beta, PLAYER2)
            if new_val > value:
                value = new_val
                move = node
            alpha = max(value, alpha)
            if alpha >= beta:
                break
    else:
        for node in available_node:
            new_board = apply_player_action(board, node, player, True)
            new_val, _ = minimax_ab(new_board, depth - 1, alpha, beta, PLAYER1)
            if new_val < value:
                value = new_val
                move = node
            beta = min(value, beta)
            if beta <= alpha:
                break

    return value, move


def generate_move_minimax_ab(
        board: np.ndarray,
        player: BoardPiece,
        saved_state: Optional[SavedState],
        depth: int = 2) -> Tuple[PlayerAction, Optional[SavedState]]:
    """

    :param board:
    :param player:
    :param saved_state:
    :param depth: depth of the search tree
    :return:
    """

    _, action = minimax_ab(board, depth, -np.inf, np.inf, player)
    return action, saved_state
