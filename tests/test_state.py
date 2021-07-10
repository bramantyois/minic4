from agents.agent_mcts import State
from agents.common import PLAYER1, PLAYER2, NO_PLAYER, apply_player_action

import numpy as np


def test_backpropagation():
    """
    test procedure to assert the propagation of score and number of trial on a simple tree
    """

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

    parent_node.backpropagate([1, 0])

    assert (parent_node.get_score() == (scores[0]+scores[1]))
    # assert (parent_node.get_n() == 3)

    score = np.random.choice([0, 0.5, 1])

    child_node_13 = State(test_board)
    child_node_13.set_score(score)
    parent_node.add_child(child_node_13)

    parent_node.backpropagate([2])

    assert (parent_node.get_score() == (scores[0]+scores[1]+ score) )
    # assert (parent_node.get_n() == 4)


def test_state_find_child():
    """
    asserting adding children to a parent State
    """
    test_board_1 = np.full((6, 7), NO_PLAYER)
    test_board_2 = apply_player_action(test_board_1, np.int8(0), PLAYER1, copy=True)
    test_board_3 = apply_player_action(test_board_2, np.int8(2), PLAYER2, copy=True)

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
    """
    Assert the node to return its State status(lead node or no)
    """
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


def test_state_score_n():
    """
    Asserting the state keep the score that is set from outside and increment of n
    """
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

    assert (child_node_11.get_score() == scores[2])
    assert (child_node_11.get_n() == 1)

    assert (child_node_12.get_score() == scores[1])
    assert (child_node_12.get_n() == 1)

    assert (child_node_21.get_score() == scores[0])
    assert (child_node_21.get_n() == 1)


def test_state_board():
    """
    assert a state keeping board
    """
    test_board = np.full((6, 7), NO_PLAYER)

    action = np.random.choice(7)
    cur_board = apply_player_action(test_board, action, PLAYER1)

    test_state = State(cur_board)

    assert ((test_state.get_board() == cur_board).all())


def test_state_add_get_children():
    """
    Asserting adding child to and getting children out of a state
    """
    test_board = np.full((6, 7), NO_PLAYER)

    parent_node = State(test_board)

    action = np.random.choice(7)
    test_board_11 = apply_player_action(test_board, action, PLAYER1)
    action = np.random.choice(7)
    test_board_12 = apply_player_action(test_board, action, PLAYER1)
    action = np.random.choice(7)
    test_board_21 = apply_player_action(test_board, action, PLAYER1)

    child_node_11 = State(test_board_11)
    child_node_12 = State(test_board_12)
    child_node_21 = State(test_board_21)

    child_node_12.add_child(child_node_21)

    parent_node.add_child(child_node_11)
    parent_node.add_child(child_node_12)

    children = parent_node.get_children()

    assert (children[0].get_board() == test_board_11).all()
    assert (children[1].get_board() == test_board_12).all()
    assert (children[1].get_children()[0].get_board() == test_board_21).all()

