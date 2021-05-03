import numpy as np

def test_generate_random_move():
    from agents.agent_random import generate_move
    from agents.common import NO_PLAYER, PLAYER1, PLAYER2, BoardPiece

    test_board = np.full((6, 7), NO_PLAYER)
    action, _ = generate_move(test_board, PLAYER1, None)

    assert(action)
    test_board = np.full((6, 7), PLAYER2)
    test_board[0, 0] = NO_PLAYER
    action, _ = generate_move(test_board, PLAYER1, None)
    assert(action == 0)
