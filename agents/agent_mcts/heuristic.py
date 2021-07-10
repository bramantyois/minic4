import numpy as np

from agents.common import BoardPiece, PlayerAction, PLAYER1, PLAYER2
from agents.common import apply_player_action,  check_valid_action

from scipy.signal import convolve2d


def compute_score(convolved_board: np.ndarray) -> float:
    """
    compute the score of the convolution. The closer the pattern to 4, the bigger the score

    :param convolved_board: array of convolution values
    :return: score the heuristic respective to convolution kernel
    """

    v4 = np.sum(convolved_board == 4) * 9999
    v3 = np.sum(convolved_board == 3) * 999
    v2 = np.sum(convolved_board == 2) * 99
    v1 = np.sum(convolved_board == 1) * 9
    vm4 = np.sum(convolved_board == -4) * 9999
    vm3 = np.sum(convolved_board == -3) * 999
    vm2 = np.sum(convolved_board == -2) * 99
    vm1 = np.sum(convolved_board == -1) * 9

    return v4 + v3 + v2 + v1 - (vm4 + vm3 + vm2 + vm1)


def get_convolution_heuristic(board: np.ndarray, player: BoardPiece) -> float:
    """
    get the heuristic value based on convolutions of kernels

    :param board: current board state
    :param player: currently playing player
    :return: heuristic value
    """
    board_tr = np.zeros_like(board)

    kernel_v = np.ones((4, 1), dtype=int)
    kernel_h = np.ones((1, 4), dtype=int)

    kernel_dl = np.zeros((4, 4), dtype=int)
    np.fill_diagonal(kernel_dl, val=1)

    kernel_dr = np.flipud(kernel_dl)

    if player == PLAYER1:
        board_tr[board == PLAYER1] = 1
        board_tr[board == PLAYER2] = -1
    else:
        board_tr[board == PLAYER1] = -1
        board_tr[board == PLAYER2] = 1

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


def get_conv_action(board: np.ndarray, player: BoardPiece) -> PlayerAction:
    """
    get the action that returns the biggest heuristic

    :param board: current board state
    :param player: currently turning player
    :return: action that maximizes the convolution heuristic
    """

    actions = np.arange(7).tolist()

    h_val = -np.inf
    chosen_action = -1

    for action in actions:
        if check_valid_action(board, action):
            new_board = apply_player_action(board, action, player, copy=True)
            cur_h_val = get_convolution_heuristic(new_board, player)

            if cur_h_val > h_val:
                chosen_action = action
                h_val = cur_h_val

    return chosen_action



