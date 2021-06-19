import numpy as np
from typing import Optional, Tuple
from agents.common import BoardPiece, SavedState, PlayerAction, PLAYER1, PLAYER2, NO_PLAYER
from agents.common import apply_player_action, check_end_state, check_valid_action
from agents.common import GameState
from scipy.signal import convolve2d
import math

import abc


class State:
    def __init__(self, board: np.ndarray):
        """

        :param board:
        """
        self.children = []
        self.score = 0
        self.n = 0
        self.board = board

    def backpropagate(self) -> None:
        """
        Backpropagation. Update the intrinsic parameters of the state.
        """

        total_score = 0
        total_n = 0
        for child in self.children:
            child.backpropagate()
            total_score += child.score
            total_n += child.n
        self.score += total_score
        self.n += total_n

    def find_child(self, board: np.ndarray):
        identical_child = None

        if (self.board == board).all():
            return self.children.copy(), self.score, self.n

        for child in self.children:
            identical_child = child.find_children(board)

        return identical_child

    @property
    def is_leaf_node(self) -> bool:
        """
        checking if the state is a leaf node
        :return:
        """
        if not self.children:
            return True
        else:
            return False

    def get_num_n(self) -> int:
        return self.n

    def add_child(self, child_state):
        self.children.append(child_state)


class Connect4MCTS:
    """
    implementation of Monte-carlo tree search on connect 4
    """

    def __init__(self, player: BoardPiece):
        self.root_node = []
        self.player = player
        self.c = 2

        if player == PLAYER1:
            self.competing_player = PLAYER2
        else:
            self.competing_player = PLAYER1

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
        actions = np.arange(7)

        board = state.board.copy()
        for action in actions:
            if check_valid_action(board, action):
                new_board = apply_player_action(action, board, self.player, copy=True)

                for action2 in actions:
                    if check_valid_action(new_board, action2):
                        new_board_2 = apply_player_action(action, new_board, self.competing_player, copy=True)
                        new_child = State(new_board_2)
                        state.add_child(new_child.copy())

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
