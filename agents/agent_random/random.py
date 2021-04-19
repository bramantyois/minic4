import numpy as np
from typing import Optional, Tuple
from agents.common import BoardPiece, SavedState, PlayerAction, NO_PLAYER


def generate_move_random(
        board: np.ndarray,
        player: BoardPiece,
        saved_state: Optional[SavedState]) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    action = -1
    ret_valid = False
    top_row = board[0, :] == NO_PLAYER
    while not ret_valid:
        action = np.random.choice(76, 1)
        if top_row[action]:
            ret_valid = True

    return action, saved_state