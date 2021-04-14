import numpy as np
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2
from agents.common import NO_PLAYER_Print, PLAYER1_Print, PLAYER2_Print


def test_initialize_game_state():
    from agents.common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)


def test_pretty_print_board():
    from agents.common import pretty_print_board

    test_str = '|====|\n'
    test_str += '|' + NO_PLAYER_Print + ' ' + PLAYER1_Print + ' |\n'
    test_str += '|' + PLAYER2_Print + ' ' + NO_PLAYER_Print + ' |\n'
    test_str += '|====|\n'
    test_str += '|0 1 |'

    test_arr = np.full((2, 2), NO_PLAYER, dtype=BoardPiece)
    test_arr[0, 1] = PLAYER1
    test_arr[1, 0] = PLAYER2

    ret = pretty_print_board(test_arr)

    assert isinstance(ret, str)
    assert ret == test_str


def test_string_to_board():
    from agents.common import string_to_board, pretty_print_board

    test_str = '|====|\n'
    test_str += '|' + NO_PLAYER_Print + ' ' + PLAYER1_Print + ' |\n'
    test_str += '|' + PLAYER2_Print + ' ' + NO_PLAYER_Print + ' |\n'
    test_str += '|====|\n'
    test_str += '|0 1 |'

    ret_arr = string_to_board(test_str)

    test_arr = np.full((2, 2), NO_PLAYER, dtype=BoardPiece)
    test_arr[0, 1] = PLAYER1
    test_arr[1, 0] = PLAYER2

    assert (ret_arr == test_arr).all()
