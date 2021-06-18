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


    def backpropagate(self)->None:
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

    def is_leaf_node(self)->bool:
        """
        checking if the state is a leaf node
        :return:
        """
        if self.children == []:
            return True
        else:
            return False

class C4MCTS:
    """
    implementation of Monte-carlo tree search on connect 4
    """
    def __init__(self):
        self.root_node = []
        self.player = NO_PLAYER

    def rollout(self, state: State):
        """

        :param state: root node in which rollout will be performed. State should have zero score and zero iteration
        :return:
        """
        cur_board = state.board
        cur_player = self.player
        score = 0

        cur_game_state = check_end_state(cur_board, cur_player)

        while cur_game_state == GameState.STILL_PLAYING:
            while True:
                action = np.random.choice(7)
                if check_valid_action(cur_board, action):
                    break

            cur_board = (cur_board, action, cur_player)

            if cur_player == PLAYER1:
                cur_player = PLAYER2
            else:
                cur_player = PLAYER1

            cur_game_state = check_end_state(cur_board, cur_player)

        if cur_game_state == GameState.IS_WIN and cur_player == self.player:
            score = 1  # agent winning the game
        elif cur_game_state == GameState.IS_DRAW:
            score = 0.5

        return score

    def USB1(self, state: State):
        pass

    def run_iteration(self, iteration_num : int = 1):
        cur_state = self.root_node

        for _ in range(iteration_num):
            while True:

                if not cur_state.is_leaf_node():
                    cur_state =







    def simulate(self, board, action):
        turn = self.player
        while check_end_state(board, self.player) == GameState.STILL_PLAYING:
            board = apply_player_action(board, action, self.player)
