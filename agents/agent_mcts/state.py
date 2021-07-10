import numpy as np
import copy


class State:
    def __init__(self, board: np.ndarray):
        """
        Class representing a node in a tree.

        :param board: board representing the state current node
        :type self._children: a list containing all the children node
        :type self._n: number of trial performed for tree
        :type self._score: score value of the tree
        :type self._board: current state board
        """
        self._children = []
        self._score = 0
        self._n = 0
        self._board = board.copy()

    def backpropagate(self, child_list) -> None:
        """
        Backpropagation. Update the intrinsic parameters of the state.

        :param child_list: list of child index. representing the back propagation path
        """
        total_score = 0
        total_n = 0
        if len(child_list) > 1:
            self._children[child_list[0]].backpropagate(child_list[1:])

        self._score += self._children[child_list[0]].get_score()
        self._n += self._children[child_list[0]].get_n()

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
            if ret[0]:
                break
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
        """
        Getter function returning number of simulation
        :return: number of elapsed simulation
        """
        return self._n

    def get_score(self) -> float:
        """
        Getter function returning the score
        :return: score
        """
        return self._score

    # def set_n(self, n: int) -> None:
    #     self._n = n

    def set_score(self, score: float) -> None:
        """
        Setting the score of the node. Automatically increment number of step n

        :param score: simulation score
        :return: None
        """
        if self._n <= 0:
            self._score = score
            self._n = 1

    def get_board(self) -> np.ndarray:
        """
        getter function to get the current board of the State
        :return: current board
        """
        return self._board.copy()

    def get_children(self):
        """
        getter function to get the children of the State
        :return: reference to the children
        """
        return self._children

    def add_child(self, child_state) -> None:
        """
        adding child state to the node
        :param child_state: child state, should be a State. Automatically deep copy the child_state

        """
        self._children.append(copy.deepcopy(child_state))
