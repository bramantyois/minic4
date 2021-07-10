import numpy as np

from agents.agent_mcts import get_conv_action, get_convolution_heuristic, compute_score
from agents.common import PLAYER1, PLAYER2, NO_PLAYER, apply_player_action, SavedState


def test_heuristic_score():
    """
    compute the heuristic score of the board. patterns that is almost 4 should return greater value
    """
    init_board = np.full((6, 7), NO_PLAYER)

    board_1 = init_board.copy()
    board_1[5, 0:3] = PLAYER1
    board_1[5, 3:5] = PLAYER2

    score_1 = get_convolution_heuristic(board_1, PLAYER1)

    board_2 = init_board.copy()
    board_2[5, 0:3] = PLAYER1
    board_2[5, 3] = PLAYER2
    board_2[4, 1] = PLAYER2

    score_2 = get_convolution_heuristic(board_2, PLAYER1)

    board_3 = init_board.copy()
    board_3[5, 0:2] = PLAYER1
    board_3[4, 1] = PLAYER2
    board_3[5, 3] = PLAYER2
    score_3 = get_convolution_heuristic(board_3, PLAYER1)

    assert (score_2 > score_1)
    assert (score_2 > score_3)


def test_get_heuristic_action():
    """
    Check if the heuristic can point out an action given a board that only requires one winning action
    """
    test_board = np.zeros((6, 7))
    test_board[3:, 0] = PLAYER1
    test_board[5, 1:3] = PLAYER2

    action = get_conv_action(test_board, PLAYER1)
    assert (action == 0)


