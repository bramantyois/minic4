import numpy as np

from agents.agent_mcts import Connect4MCTS
from agents.common import PLAYER1, PLAYER2, NO_PLAYER, apply_player_action, SavedState


def test_mcts_set_player():
    """
    assert that the agent play the Player that is the argument of generate_move
    """
    agent = Connect4MCTS()
    agent.set_player(PLAYER1)
    assert (agent.get_player() == PLAYER1)

    agent.set_player(PLAYER2)
    assert (agent.get_player() == PLAYER2)


def test_mcts_set_current_board():
    """
    assert that the agent based its move by the current board
    """
    init_board = np.full((6, 7), NO_PLAYER)
    init_board = apply_player_action(init_board, np.int8(0), PLAYER1)
    init_board = apply_player_action(init_board, np.int8(0), PLAYER2)

    agent = Connect4MCTS()
    agent.set_player(PLAYER1)
    agent.set_current_board(init_board)

    assert (agent.get_root_node().get_board() == init_board).all()


def test_mcts_rollout():
    """
    asserting that after the rollout, the node has different board state
    :return:
    """
    init_board = np.full((6, 7), NO_PLAYER)
    saved_state = SavedState()
    agent = Connect4MCTS()

    action, _ = agent.generate_move_mcts(init_board, PLAYER2, saved_state)
    init_board = apply_player_action(init_board, action, PLAYER2)

    rn_board = agent.get_root_node().get_board()

    init_board = apply_player_action(init_board, np.int8(0), PLAYER1)

    action, _ = agent.generate_move_mcts(init_board, PLAYER2, saved_state)

    assert (rn_board != agent.get_root_node().get_board()).any()


def test_mcts_expand():
    """
    assert that the agent expand a desired node state
    """
    init_board = np.full((6, 7), NO_PLAYER)
    saved_state = SavedState()
    agent = Connect4MCTS()
    _ = agent.generate_move_mcts(init_board, PLAYER2, saved_state)

    children = agent.get_root_node().get_children()
    assert (children != [])

    for child in children:
        assert (child.get_board() != init_board).any()


def test_mcts_iterate():
    """
    assert that the agent perform iteration, in which it would expand and rollout
    """
    init_board = np.full((6, 7), NO_PLAYER)
    saved_state = SavedState()
    agent = Connect4MCTS()
    agent.set_player(PLAYER1)
    agent.iterate()

    cur_n = agent.get_root_node().get_n()

    assert (cur_n != 0)
    assert (len(agent.get_root_node().get_children()) != 0)

    agent.iterate()

    assert (agent.get_root_node().get_n() != cur_n)


def test_mcts_run_iteration():
    """
    Assert that mcts.run_iteration will be run for a certain time or number of iteration
    """
    from time import time
    max_t = 2

    init_board = np.full((6, 7), NO_PLAYER)

    agent = Connect4MCTS(max_t=max_t)
    agent.set_current_board(init_board)
    agent.set_player(PLAYER1)

    cur_time = time()
    agent.run_iteration()
    assert ((time() - cur_time) <= 1e2)

    max_iter = 10

    agent = Connect4MCTS(max_iter=max_iter, curb_iter_time=False)
    agent.set_current_board(init_board)
    agent.set_player(PLAYER1)

    agent.run_iteration()

    assert (agent.get_root_node().get_n() == max_iter)


def test_mcts_choose_action():
    """
    given a tree, choose an action that maximize score in the smallest step count
    """
    init_board = np.full((6, 7), NO_PLAYER)

    agent = Connect4MCTS()
    agent.set_current_board(init_board)
    agent.set_player(PLAYER1)

    agent.run_iteration()

    action = agent.choose_action()
    biggest_score = agent.get_root_node().get_children()[int(action)].get_score()
    for i in range(7):
        assert agent.get_root_node().get_children()[0].get_score() <= biggest_score


def test_mcts_generate_move():
    """
    given current board, assert that agent pick an action that maximizes simulation score
    :return:
    """
    init_board = np.full((6, 7), NO_PLAYER)
    saved_state = SavedState()

    agent = Connect4MCTS()
    action, saved_state = agent.generate_move_mcts(init_board, PLAYER1, saved_state)

    biggest_score = agent.get_root_node().get_children()[int(action)].get_score()
    for i in range(7):
        assert agent.get_root_node().get_children()[0].get_score() <= biggest_score

