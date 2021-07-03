import numpy as np

from agents.agent_mcts import Connect4MCTS
from agents.common import PLAYER1, PLAYER2, NO_PLAYER, apply_player_action, SavedState


def test_mcts_set_player():
    agent = Connect4MCTS()
    agent.set_player(PLAYER1)
    assert (agent.get_player() == PLAYER1)

    agent.set_player(PLAYER2)
    assert (agent.get_player() == PLAYER2)


def test_mcts_set_current_board():
    init_board = np.full((6, 7), NO_PLAYER)
    init_board = apply_player_action(init_board, np.int8(0), PLAYER1)
    init_board = apply_player_action(init_board, np.int8(0), PLAYER2)

    agent = Connect4MCTS()
    agent.set_player(PLAYER1)
    agent.set_current_board(init_board)

    assert (agent.get_root_node().get_board() == init_board).all()


def test_mcts_expand():
    init_board = np.full((6, 7), NO_PLAYER)
    saved_state = SavedState()
    agent = Connect4MCTS()
    _ = agent.generate_move_mcts(init_board, PLAYER2, saved_state)

    children = agent.get_root_node().get_children()
    assert (children != [])

    for child in children:
        assert (child.get_board() != init_board).any()


def test_mcts_rollout():
    init_board = np.full((6, 7), NO_PLAYER)
    saved_state = SavedState()
    agent = Connect4MCTS()

    action, _ = agent.generate_move_mcts(init_board, PLAYER2, saved_state)
    init_board = apply_player_action(init_board, action, PLAYER2)

    rn_board = agent.get_root_node().get_board()

    init_board = apply_player_action(init_board, np.int8(0), PLAYER1)

    action, _ = agent.generate_move_mcts(init_board, PLAYER2, saved_state)

    assert (rn_board != agent.get_root_node().get_board()).any()


def test_run_iteration():
    pass
