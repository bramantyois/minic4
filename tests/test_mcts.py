import numpy as np


def test_backpropagation() -> None:
    """
    test procedure to assert the propagation of score and number of trial on a simple tree
    """
    from agents.agent_mcts import State
    from agents.common import NO_PLAYER

    test_board = np.full((6, 7), NO_PLAYER)

    parent_node = State(test_board)
    child_node_11 = State(test_board)
    child_node_12 = State(test_board)
    child_node_21 = State(test_board)

    scores = np.random.choice([0, 0.5, 1], 3)

    child_node_21.set_score(scores[0])
    child_node_12.set_score(scores[1])
    child_node_11.set_score(scores[2])

    child_node_12.add_child(child_node_21)

    parent_node.add_child(child_node_11)
    parent_node.add_child(child_node_12)

    parent_node.backpropagate()

    assert (parent_node.get_score() == np.sum(scores))
    assert (parent_node.get_n() == 3)


def test_state_find_child():
    from agents.agent_mcts import State
    from agents.common import NO_PLAYER, PLAYER1, PLAYER2
    from agents.common import apply_player_action

    test_board_1 = np.full((6, 7), NO_PLAYER)
    test_board_2 = apply_player_action(test_board_1, 0, PLAYER1, copy=True)
    test_board_3 = apply_player_action(test_board_2, 2, PLAYER2, copy=True)

    state_1 = State(test_board_1)

    ret = state_1.find_child(test_board_1)
    assert (ret[0])
    assert (ret[1].get_board() == test_board_1).all()

    state_2 = State(test_board_2)
    state_1.add_child(state_2)

    ret = state_1.find_child(test_board_1)
    assert (ret[0])
    assert (ret[1].get_board() == test_board_1).all()

    ret = state_1.find_child(test_board_2)
    assert (ret[0])
    assert (ret[1].get_board() == test_board_2).all()

    ret = state_1.find_child(test_board_3)
    assert (not ret[0])


def test_state_leaf_node():
    from agents.agent_mcts import State
    from agents.common import NO_PLAYER

    test_board = np.full((6, 7), NO_PLAYER)

    parent_node = State(test_board)
    child_node_11 = State(test_board)
    child_node_12 = State(test_board)
    child_node_21 = State(test_board)

    child_node_12.add_child(child_node_21)

    parent_node.add_child(child_node_11)
    parent_node.add_child(child_node_12)

    assert (not parent_node.is_leaf_node())
    assert (not child_node_12.is_leaf_node())
    assert (child_node_11.is_leaf_node())
    assert (child_node_21.is_leaf_node())


def test_mcts_rollout():
    from agents.agent_mcts import State
    from agents.agent_mcts import Connect4MCTS
    from agents.common import PLAYER1

    agent = Connect4MCTS(player=PLAYER1)


def test_mcts_expand():
    pass


def test_run_iteration():
    pass
