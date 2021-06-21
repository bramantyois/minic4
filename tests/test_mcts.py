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

    pass


def test_state_leaf_node():

    pass


def test_mcts_rollout():
    pass


def test_mcts_expand():
    pass


def test_run_iteration():
    pass
