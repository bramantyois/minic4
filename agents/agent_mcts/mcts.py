import numpy as np
from typing import Optional
from agents.common import BoardPiece, SavedState, PLAYER1, PLAYER2, NO_PLAYER
from agents.common import apply_player_action, check_end_state, check_valid_action
from agents.common import GameState

from agents.agent_mcts import State

import math
import time


class Connect4MCTS:
    """
    implementation of Monte-carlo tree search on connect 4
    """
    def __init__(
            self,
            expansion_rate: int = 1,
            curb_iter_time: bool = True,
            max_t: float = 2,
            max_iter: int = 10):
        """

        :param max_iter:
        :param expansion_rate:
        """
        self._expansion_rate = expansion_rate

        self._time_curb = curb_iter_time
        self._max_t = max_t
        self._max_iter = max_iter

        self._root_node = State(board=np.zeros((6, 7)))
        self._player = NO_PLAYER
        self._past_player = NO_PLAYER
        self._competing_player = NO_PLAYER
        self._c = 2

    def set_player(self, player: BoardPiece):
        self._player = player

        if player == PLAYER1:
            self._competing_player = PLAYER2
        else:
            self._competing_player = PLAYER1

        if self._past_player != self._player:
            self.flush_tree()
            self._past_player = self._player

    def set_current_board(self, board: np.ndarray):
        ret = self._root_node.find_child(board)

        if ret[0]:
            self._root_node = ret[1]
        else:
            self._root_node = State(board)

    def get_root_node(self):
        return self._root_node

    def get_player(self):
        return self._player

    def rollout(self, state: State):
        """

        :param state: root node in which rollout will be performed. State should have zero score and zero iteration
        :return:
        """
        cur_board = state.get_board()
        cur_player = self._player
        score = 0

        while check_end_state(cur_board, cur_player) == GameState.STILL_PLAYING:
            while True:
                action = np.random.choice(7)
                if check_valid_action(cur_board, action):
                    break

            cur_board = apply_player_action(cur_board, action, cur_player, copy=True)

            if cur_player == PLAYER1:
                cur_player = PLAYER2
            else:
                cur_player = PLAYER1

        end_game_state = check_end_state(cur_board, cur_player)
        if end_game_state == GameState.IS_WIN and cur_player == self._player:
            score = 1  # agent winning the game
        elif end_game_state == GameState.IS_DRAW:
            score = 0.5

        state.set_score(score)
        self._root_node.backpropagate()

    def expand(self, state: State):
        actions_1 = np.arange(7)
        actions_2 = np.arange(7)

        np.random.shuffle(actions_1)
        np.random.shuffle(actions_2)

        board = state.get_board()

        if check_end_state(board, self._player) == GameState.STILL_PLAYING:
            for action in actions_1:
                if check_valid_action(board, action):
                    new_board = apply_player_action(board, action, self._player, copy=True)

                    count_2 = 0
                    for action2 in actions_2:
                        if check_valid_action(new_board, action2):
                            new_board_2 = apply_player_action(new_board, action2, self._competing_player, copy=True)
                            new_child = State(new_board_2)
                            state.add_child(new_child)

                            count_2 += 1
                            if count_2 >= self._expansion_rate:
                                break

                    # count_1 += 1
                    # if count_1 >= self._expansion_rate:
                    #     break

    def iterate(self):
        cur_state = self._root_node
        while True:
            if cur_state.is_leaf_node():
                if cur_state.get_n() == 0:
                    self.rollout(cur_state)
                    break
                else:
                    self.expand(cur_state)
                    if len(cur_state.get_children()) != 0:
                        self.rollout(cur_state.get_children()[0])
                    break
            else:
                idx = -1
                ucb1 = -math.inf
                for i, child in enumerate(cur_state.get_children()):
                    n = child.get_n()

                    if n == 0:
                        idx = i
                        break
                    else:
                        new_val = child.get_score()/n
                        new_val += self._c * math.sqrt(math.log(self._root_node.get_n()) / n)

                    if new_val > ucb1:
                        idx = i
                        ucb1 = new_val

                cur_state = cur_state.get_children()[idx]

    def run_iteration(self):
        if self._time_curb:
            cur_time = time.time()
            while True:
                self.iterate()
                elapsed = time.time() - cur_time
                if elapsed > self._max_t:
                    break
        else:
            for _ in range(self._max_iter):
                self.iterate()

    def choose_action(self):
        max_score = -999
        child_idx = -1
        for i, child in enumerate(self._root_node.get_children()):
            score = child.get_score()
            if max_score < score:
                child_idx = i
                max_score = score

        winning_board = self._root_node.get_children()[child_idx].get_board() == self._player
        cur_board = self._root_node.get_board() == self._player

        bool_board = np.where(cur_board != winning_board)
        action = bool_board[1].astype(np.int8)

        return action

    def generate_move_mcts(self, board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]):
        self.set_player(player)
        self.set_current_board(board)
        self.run_iteration()

        action = self.choose_action()

        return action, saved_state

    def flush_tree(self):
        self._root_node = State(board=np.zeros((6, 7)))
