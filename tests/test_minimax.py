import numpy as np


def test_generate_minimax_move():
    """
    First checking whether the agent can return an action.
    Then it asserts the agent will producing valid move.
    Next, it will test if the agent can produce a winning move
    given a board state
    """
    from agents.agent_minimax import generate_move
    from agents.common import NO_PLAYER, PLAYER1, PLAYER2, BoardPiece

    test_board = np.full((6, 7), NO_PLAYER)
    action, _ = generate_move(test_board, PLAYER1, None)
    assert action

    test_board = np.full((6, 7), PLAYER2)
    test_board[0, 0] = NO_PLAYER
    action, _ = generate_move(test_board, PLAYER1, None)
    assert (action == 0)

    test_board = np.zeros((6, 7))
    test_board[3:, 0] = PLAYER1
    test_board[5, 1:3] = PLAYER2

    action, _ = generate_move(test_board, PLAYER1, None)
    assert (action == 0)

    action, _ = generate_move(test_board, PLAYER2, None)
    assert (action == 0)
