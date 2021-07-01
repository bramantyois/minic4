import numpy as np
from agents.common import BoardPiece, SavedState, PlayerAction, PLAYER1, PLAYER2, NO_PLAYER
from agents.common import apply_player_action, check_end_state, check_valid_action
from agents.common import GameState
import math
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

class Connect4MCTS:
    """
    implementation of Monte-carlo tree search on connect 4
    """
    def __init__(self, player: BoardPiece, expansion_rate: int = 1):
        """

        :param player:
        :param expansion_rate:
        """
        self.expansion_rate = expansion_rate**1
        self.root_node = []
        self.player = player
        self.c = 2

        if player == PLAYER1:
            self.competing_player = PLAYER2
        else:
            self.competing_player = PLAYER1

    def set_current_board(self, board: np.ndarray):
        ret = self.root_node.find_child(board)

        if ret[0]:
            self.root_node = ret[1]
        else:
            self.root_node = State(board)

    def rollout(self, state: State):
        """

        :param state: root node in which rollout will be performed. State should have zero score and zero iteration
        :return:
        """
        cur_board = state.board
        cur_player = self.player
        score = 0

        while check_end_state(cur_board, cur_player) == GameState.STILL_PLAYING:
            while True:
                action = np.random.choice(7)
                if check_valid_action(cur_board, action):
                    break

            cur_board = apply_player_action(cur_board, action, cur_player)

            if cur_player == PLAYER1:
                cur_player = PLAYER2
            else:
                cur_player = PLAYER1

        end_game_state = check_end_state(cur_board, cur_player)
        if end_game_state == GameState.IS_WIN and cur_player == self.player:
            score = 1  # agent winning the game
        elif end_game_state == GameState.IS_DRAW:
            score = 0.5

        return score

    def expand(self, state: State):
        actions = np.random.arange(7)
        np.random.shuffle(actions)

        board = state.board.copy()

        count = 0
        for action in actions:
            if check_valid_action(board, action):
                new_board = apply_player_action(action, board, self.player, copy=True)
                count += 1
                for action2 in actions:
                    if check_valid_action(new_board, action2):
                        new_board_2 = apply_player_action(action, new_board, self.competing_player, copy=True)
                        new_child = State(new_board_2)
                        state.add_child(new_child.copy())
                        count += 1

                        if count >= self.expansion_rate:
                            break
                if count >= self.expansion_rate:
                    break

    def run_iteration(self, iteration_num: int = 1):
        cur_state = self.root_node

        for _ in range(iteration_num):
            while True:

                if cur_state.is_leaf_node():
                    if cur_state.get_num_n() == 0:
                        self.expand(cur_state)
                        cur_state = cur_state.children[0]
                        self.rollout(cur_state)
                        break
                    else:
                        self.rollout(cur_state)
                else:
                    idx = -1
                    ucb1 = -999
                    for i, child in enumerate(cur_state.children):
                        new_val = child.score + self.c * math.sqrt(cur_state.N / cur_state.n)  # check division by zero

                        if new_val > ucb1:
                            idx = i
                            ucb1 = new_val

                    cur_state = cur_state.children[idx]
