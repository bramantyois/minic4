import numpy as np
import copy


class State:
    """

    :param board: board representing the state current node
    :type _children: a list containing all the children node
    :type _n: number of trial performed for tree
    :type _score: score value of the tree
    :type _board: current state board
    """
    def __init__(self, board: np.ndarray):

        self._children = []
        self._score = 0
        self._n = 0
        self._board = board.copy()

    def backpropagate(self) -> None:
        """
        Backpropagation. Update the intrinsic parameters of the state.
        """

        total_score = 0
        total_n = 0
        for child in self._children:
            child.backpropagate()
            total_score += child.get_score()
            total_n += child.get_n()
        self._score += total_score
        self._n += total_n

    def find_child(self, board: np.ndarray):
        """
        Find the node that has the same board state
        :param board:
        :return: Tuple of bool and State. return none if no child having given board
        """
        if (self._board == board).all():
            return True, copy.deepcopy(self)

        ret = False, None
        for child in self._children:
            ret = child.find_child(board)
        return ret

    def is_leaf_node(self) -> bool:
        """
        checking if the state is a leaf node
        :return:
        """
        if not self._children:
            return True
        else:
            return False

    def get_n(self) -> int:
        return self._n

    def get_score(self) -> float:
        return self._score

    def set_n(self, n: int) -> None:
        self._n = n

    def set_score(self, score) -> None:
        if self._n <= 0:
            self._score = score
            self._n = 1

    def add_child(self, child_state):
        self._children.append(copy.deepcopy(child_state))

    def get_board(self):
        return self._board.copy()

    def get_children(self):
        return self._children
