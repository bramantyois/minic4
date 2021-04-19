import numpy as np
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2
from agents.common import NO_PLAYER_PRINT, PLAYER1_PRINT, PLAYER2_PRINT


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
    test_str += '|' + NO_PLAYER_PRINT + ' ' + PLAYER1_PRINT + ' |\n'
    test_str += '|' + PLAYER2_PRINT + ' ' + NO_PLAYER_PRINT + ' |\n'
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
    test_str += '|' + NO_PLAYER_PRINT + ' ' + PLAYER1_PRINT + ' |\n'
    test_str += '|' + PLAYER2_PRINT + ' ' + NO_PLAYER_PRINT + ' |\n'
    test_str += '|====|\n'
    test_str += '|0 1 |'

    ret_arr = string_to_board(test_str)

    test_arr = np.full((2, 2), NO_PLAYER, dtype=BoardPiece)
    test_arr[0, 1] = PLAYER1
    test_arr[1, 0] = PLAYER2

    assert (ret_arr == test_arr).all()


def test_apply_player_action():
    from agents.common import apply_player_action

    test_arr = np.full((2, 2), NO_PLAYER, dtype=BoardPiece)
    test_arr[1, 1] = PLAYER1
    test_arr[1, 0] = PLAYER2

    ret = apply_player_action(test_arr, 0, PLAYER1)

    test_arr[0, 0] = PLAYER1

    assert (ret.dtype == BoardPiece)
    assert ((test_arr == ret).all())

def test_connected_four():
    from agents.common import connected_four

    test_arr = np.zeros((6, 7))
    test_arr[5, 0] = PLAYER1
    test_arr[5, 1] = PLAYER1
    test_arr[5, 2] = PLAYER1
    test_arr[5, 3] = PLAYER1

    test_arr[4, 0] = PLAYER2
    test_arr[4, 1] = PLAYER2
    test_arr[4, 2] = PLAYER2

    assert(connected_four(test_arr, PLAYER1))

    test_arr = np.zeros((6, 7))
    test_arr[2, 4] = PLAYER2
    test_arr[3, 4] = PLAYER2
    test_arr[4, 4] = PLAYER2
    test_arr[5, 4] = PLAYER2

    test_arr[3, 5] = PLAYER1
    test_arr[4, 5] = PLAYER1
    test_arr[5, 5] = PLAYER1

    assert(connected_four(test_arr, PLAYER2))

    test_arr = np.zeros((6, 7))
    test_arr[5, 0] = PLAYER1
    test_arr[4, 1] = PLAYER1
    test_arr[3, 2] = PLAYER1
    test_arr[2, 3] = PLAYER1

    test_arr[5, 6] = PLAYER2
    test_arr[5, 5] = PLAYER2
    test_arr[5, 4] = PLAYER2

    assert(connected_four(test_arr, PLAYER1))

    assert(connected_four(np.flipud(test_arr), PLAYER1))