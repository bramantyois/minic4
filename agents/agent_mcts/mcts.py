import numpy as np
from typing import Optional, List, Tuple

from agents.common import BoardPiece, SavedState, PLAYER1, PLAYER2, NO_PLAYER, PlayerAction
from agents.common import apply_player_action, check_end_state, check_valid_action
from agents.common import GameState

from agents.agent_mcts import State
from agents.agent_mcts import get_conv_action

import math
import time


class Connect4MCTS:
    def __init__(
            self,
            use_heuristic: bool = True,
            expansion_rate: int = 1,
            curb_iter_time: bool = False,
            max_t: float = 2,
            max_iter: int = 100):
        """
        Implementation of a Monte-Carlo tree search agent on  game of connect 4

        :type use_heuristic: using convolution heuristic if True. Otherwise random action will be chosen when simulating.
            Note that using heuristic improves the 'intelligence' of the agent a lot
        :type expansion_rate: dictates the expansion corresponds the action that will be taken by the competing player.
            Only valid if use_heuristic is False
        :type max_t: maximum time in second. iteration stops when t > max_t
        :type max_iter: number of maximum iteration
        :type curb_iter_time: use time limit instead of iteration number
        """
        self._expansion_rate = expansion_rate

        self._use_heuristic = use_heuristic

        self._time_curb = curb_iter_time
        self._max_t = max_t
        self._max_iter = max_iter

        self._root_node = State(board=np.zeros((6, 7)))
        self._player = NO_PLAYER
        self._past_player = NO_PLAYER
        self._competing_player = NO_PLAYER

        self._c = 2

        self._back_propagation_path = []

    def set_player(self, player: BoardPiece) -> None:
        """
        Set which player the agent would play. Flush tree if the agent switches to another BoardPiece

        :param player: turning agent
        :return: None
        """
        self._player = player

        if player == PLAYER1:
            self._competing_player = PLAYER2
        else:
            self._competing_player = PLAYER1

        if self._past_player != self._player:
            self.flush_tree()
            self._past_player = self._player

    def set_current_board(self, board: np.ndarray) -> None:
        """
        Set the current board state in which agent would base the decision on.
        If board state has been simulated before, the 'knowledge' tree would be transferred

        :param board: current board state
        :return: None
        """
        ret = self._root_node.find_child(board)

        if ret[0]:
            self._root_node = ret[1]
        else:
            self._root_node = State(board)

    def get_root_node(self) -> State:
        """
        Getter function returning the root node

        :return: root node
        """
        return self._root_node

    def get_player(self) -> BoardPiece:
        """
        Getter function returning the current BoardPiece the agent is playing

        :return: player
        """
        return self._player

    def flush_tree(self) -> None:
        """
        Setting the root node to a blank board. Used this when changing player

        :return:None
        """
        self._root_node = State(board=np.zeros((6, 7)))

    def rollout(self, state: State, backprop_path: List) -> None:
        """
        Simulating a game starting from given state. Actions are taken randomly. When finally reaches end state,
        root node will start back-propagating according to backprop_path

        :param state: root node in which rollout will be performed. State should have zero score and zero iteration
        :param backprop_path: list containing children index relative to root node in which back propagation would be
            performed
        :return: None
        """
        cur_board = state.get_board()
        cur_player = self._player
        score = 0

        while ((check_end_state(cur_board, PLAYER1) == GameState.STILL_PLAYING) and
                (check_end_state(cur_board, PLAYER2) == GameState.STILL_PLAYING)):

            if self._use_heuristic:
                action = get_conv_action(cur_board, cur_player)
            else:
                while True:
                    action = np.random.choice(7)
                    if check_valid_action(cur_board, action):
                        break

            cur_board = apply_player_action(cur_board, action, cur_player, copy=True)

            if cur_player == PLAYER1:
                cur_player = PLAYER2
            else:
                cur_player = PLAYER1

        end_game_state = check_end_state(cur_board, self._player)
        if end_game_state == GameState.IS_WIN:
            score = 1  # agent winning the game
        elif end_game_state == GameState.IS_DRAW:
            score = 0.5

        state.set_score(score)
        self._root_node.backpropagate(backprop_path)

    def expand(self, state: State) -> None:
        """
        Expanding the tree by adding children to state. By default, the tree will expand at least to the number
        of possible moves can be taken by _player. _expansion_rate will determine how the tree will further
        expanded in respect to the actions that would be taken by _competing_player.

        :param state: tree node in which expansion would be performed
        :return: None
        """
        actions_1 = np.arange(7)
        actions_2 = np.arange(7)

        board = state.get_board()

        if check_end_state(board, self._player) == GameState.STILL_PLAYING:
            for action in actions_1:
                if check_valid_action(board, action):
                    new_board = apply_player_action(board, action, self._player, copy=True)

                    if self._use_heuristic:
                        action2 = get_conv_action(new_board, self._competing_player)
                        new_board_2 = apply_player_action(new_board, action2, self._competing_player, copy=True)
                        new_child = State(new_board_2)
                        state.add_child(new_child)
                    else:
                        count_2 = 0
                        np.random.shuffle(actions_2)
                        for action2 in actions_2:
                            if check_valid_action(new_board, action2):
                                new_board_2 = apply_player_action(new_board, action2, self._competing_player, copy=True)
                                new_child = State(new_board_2)
                                state.add_child(new_child)

                                count_2 += 1
                                if count_2 >= self._expansion_rate:
                                    break

    def iterate(self) -> None:
        """
        The mcts algorithm. perform rollout when reaching a leaf node with no simulation amd will expand otherwise.
        It will select the node that maximize the UCB1 value.

        :return: None
        """
        if len(self._root_node.get_children()) == 0:
            self.expand(self._root_node)

        cur_state = self._root_node
        back_propagation_path = []
        while True:
            if cur_state.is_leaf_node():
                if cur_state.get_n() == 0:
                    self.rollout(cur_state, back_propagation_path)
                    break
                else:
                    self.expand(cur_state)
                    if len(cur_state.get_children()) != 0:
                        back_propagation_path.append(0)
                        self.rollout(cur_state.get_children()[0], back_propagation_path)
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
                back_propagation_path.append(idx)

    def run_iteration(self) -> None:
        """
        Run iteration of mcts algorithm. Stop when whether time _max_t is up or the number of iteration is bigger than
        _max_iter

        :return: None
        """
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

    def choose_action(self) -> BoardPiece:
        """
        Choose action based on scores of the root node's children. Score will be scaled by the number of simulation
        performed to prefer immediate winning action.

        :return: action for the agent
        """
        max_score = -math.inf
        child_idx = -1
        for i, child in enumerate(self._root_node.get_children()):
            score = child.get_score() / child.get_n()
            if max_score < score:
                child_idx = i
                max_score = score

        winning_board = self._root_node.get_children()[child_idx].get_board() == self._player
        cur_board = self._root_node.get_board() == self._player

        bool_board = np.where(cur_board != winning_board)
        action = bool_board[1].astype(np.int8)

        return action

    def generate_move_mcts(self, board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) \
            -> Tuple[PlayerAction, SavedState]:
        """
        Generate action by mcts agent.

        :param board: current board state
        :param player: turning player
        :param saved_state: unused
        :return: tuple of chosen action and saved state
        """
        self.set_player(player)
        self.set_current_board(board)
        self.run_iteration()

        action = self.choose_action()

        return action, saved_state

