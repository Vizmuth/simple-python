import os
import ctypes
import re
from typing import Tuple
import numpy as np
from scipy.ndimage import convolve, label

# enable ANSI code for window cmd
kernel32 = ctypes.windll.kernel32
kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

# maximum map size & total cells / mines ratio
MAX_SIZE = 100
MAX_MINE_FACTOR = 2

# set row number width from maximum size
NUM_WIDTH = len(str(MAX_SIZE))

# convolution/labeling range
RANGE = np.ones((3, 3), dtype=np.int8)

# ANSI coded characters for display
cell = {
    0: '\u001b[48;5;240m 0 \u001b[0m',
    1: '\u001b[48;5;088m\u001b[38;5;202m 1 \u001b[0m',
    2: '\u001b[48;5;130m\u001b[38;5;214m 2 \u001b[0m',
    3: '\u001b[48;5;136m\u001b[38;5;220m 3 \u001b[0m',
    4: '\u001b[48;5;064m\u001b[38;5;148m 4 \u001b[0m',
    5: '\u001b[48;5;022m\u001b[38;5;040m 5 \u001b[0m',
    6: '\u001b[48;5;024m\u001b[38;5;075m 6 \u001b[0m',
    7: '\u001b[48;5;018m\u001b[38;5;063m 7 \u001b[0m',
    8: '\u001b[48;5;053m\u001b[38;5;165m 8 \u001b[0m',
    ' ': '\u001b[48;5;235m   \u001b[0m',
    'F': '\u001b[48;5;250m\u001b[38;5;236m F \u001b[0m',
    -1: '\u001b[48;5;215m\u001b[31;1m\u001b[1m M \u001b[0m',
}


# Exceptions
class Quit(Exception):
    pass


class Restart(Exception):
    pass


class GameOver(Exception):
    pass


class Win(Exception):
    pass


def clear_screen():
    """
    Clearing screen for various types of OS.
    """
    os.system("cls" if os.name == 'nt' else 'clear')


def int_to_letters(value: int):
    """
    Converts int number value to alphabetized column letters. (e.g. Microsoft Excel)

    Args:
        value: int
            Column value.

    Returns:
        letter: str
            Alphabetic column letters.
    """
    letter = str()

    # value must be >= 1, but we subtract 1 to count from 0. (0 -> A, 1 -> B, ..., 25 -> Z)
    while value > 0:
        r = (value-1) % 26
        letter = chr(r+65) + letter
        value = (value-1) // 26
    return letter


def letters_to_int(letters: str):
    """
    Converts alphabetized column letters to int. (e.g. Microsoft Excel)

    Args:
        letters: str
            Alphabetic column letters.

    Returns:
        value: int
            Column value.
    """
    value = 0
    for i in range(len(letters)):
        char = letters[len(letters) - i - 1]
        if 65 <= ord(char) < 91:
            value += (ord(char) - 64) * (26 ** i)
        elif 97 <= ord(char) < 123:
            value += (ord(char) - 96) * (26 ** i)
    return value - 1


def draw_map(value: np.ndarray, visible: np.ndarray):
    """
    Draws map from value & visible numpy array.

    Args:
        value: np.ndarray
            Current map's actual value.
        visible: np.ndarray
            Represents whether cell is covered, revealed or flagged.
    """
    clear_screen()
    print('')  # enter
    print(" "*NUM_WIDTH, end=' ')
    for column in range(value.shape[1]):
        print("{0:^{1}}".format(int_to_letters(column+1), NUM_WIDTH), end='')
    print('')
    for row in range(value.shape[0]):
        print("{0:>3}".format(row+1), end=' ')
        for column in range(value.shape[1]):
            if visible[row, column] == 1:
                print(cell[value[row, column]], end='')
            elif visible[row, column] == 0:
                print(cell[' '], end='')
            elif visible[row, column] == -1:
                print(cell['F'], end='')
        print("{0:>3}".format(row + 1))
    print(" "*NUM_WIDTH, end=' ')
    for column in range(value.shape[1]):
        print("{0:^{1}}".format(int_to_letters(column+1), NUM_WIDTH), end='')
    print('')


def init_map(width: int, height: int, mines: int, avoid: Tuple[int, int]):
    """
    Initializes map from the given inputs.

    Args:
        width: int
            Width of the map.
        height: int
            Height of the map.
        mines: int
            Number of mines to be generated.
        avoid: Tuple[int, int]
            Position of the very first clicked cell in (row, column) order.
            This cell and adjacent 8 cells must not contain any mines.

    Returns:
        map: np.ndarray
            Generated map.
    """
    new_map = np.zeros((height, width), dtype=np.int8)
    while True:
        v = avoid[0] * width + avoid[1]
        avoids = [v - width - 1, v - width, v - width + 1,
                  v - 1, v, v + 1,
                  v + width - 1, v + width, v + width + 1]  # cells to avoid
        pos = np.random.permutation(np.setdiff1d(np.arange(width*height), avoids))[:mines]  # generate random values
        for row in range(height):
            for column in range(width):
                if row * width + column in pos:
                    new_map[row, column] = 1
        return new_map


def get_mapinfo():
    """
    Function for getting map size and number of mines from player.

    Returns:
        width: int
            Width of the map.
        height: int
            Height of the map.
        mines: int
            Number of mines to be generated.
    """
    while True:
        l = input("Please input width, height and number of mines without comma: ").split()
        if len(l) != 3:
            print("Number of inputs do not match.")
        try:
            width = int(l[0])
            height = int(l[1])
            mines = int(l[2])
            if 0 < width <= MAX_SIZE and 0 < height <= MAX_SIZE:
                if mines >= 1:
                    if width * height > mines * MAX_MINE_FACTOR and mines <= width * height - 3 * min(3, width, height):
                        return width, height, mines
                    else:
                        print("Too many mines.")
                else:
                    print("At least 1 mine must be present.")
            else:
                print("Dimension is too big or too small.")
        except ValueError:
            print("Please input integers.")


def get_value(map: np.ndarray):
    """
    Function for generating value map from normal map.
    Use convolution to calculate number of adjacent mines.
    Then set value -1 where mines are present.

    Args:
        map: np.ndarray
            Normal map that only has position of mines.

    Returns:
        value: np.ndarray
            Valued map that additionally includes number of surrounding mines.
    """
    value = convolve(map, RANGE, mode='constant')
    value[map == 1] = -1
    return value


def update_visible(value: np.ndarray, visible: np.ndarray, pos: Tuple[int, int], flag: bool):
    """
    Updates map visible by input data.
    If flag is True, flag/unflag corresponding cell.
    Else, check if value is 0 or not.
    It just ignores when flagged cell is selected.
    If value is 0, floodfill near 0s plus 1 width border to be visible.
    When value is not 0 and not flagged, only selected cell reveals.

    Args:
        value: np.ndarray
            Current map's actual value.
        visible: np.ndarray
            Represents whether cell is covered, revealed or flagged.
        pos: Tuple[int, int]
            Position of selected cell.
        flag: bool
            True if flag option is selected.

    Returns:
        visible: np.ndarray
            Represents whether cell is covered, revealed or flagged.
    """
    if flag:
        if visible[pos[0], pos[1]] == 0:
            visible[pos[0], pos[1]] = -1
            return visible
        elif visible[pos[0], pos[1]] == -1:
            visible[pos[0], pos[1]] = 0
            return visible
        else:
            return visible
    else:
        if visible[pos[0], pos[1]] == -1:
            return visible
        elif value[pos[0], pos[1]] == 0:
            val_temp = value.copy()

            # interchanges non-zeros and zeros, because label function only takes account to non-zero values
            val_temp[val_temp == 0] = 9
            val_temp[np.logical_and(val_temp != 0, val_temp < 9)] = 0

            # label each 'groups' in value
            labeled, _ = label(val_temp, structure=RANGE)

            # set visible to True if label is equal to the label of selected cell
            vis_temp = np.zeros_like(visible, dtype=np.int8)
            vis_temp[labeled == labeled[pos[0], pos[1]]] = 1

            # also set visible to True for one additional width
            vis_temp = convolve(vis_temp, RANGE, mode='constant')

            # set visible to 1 where vis_temp was True
            visible[vis_temp >= 1] = 1
            return visible
        else:
            visible[pos[0], pos[1]] = 1
            return visible


def check_end(value: np.ndarray, visible: np.ndarray, pos: Tuple[int, int]):
    """
    Check if game must be over.

    Args:
        value: np.ndarray
            Current map's actual value.
        visible: np.ndarray
            Represents whether cell is covered, revealed or flagged.
        pos: Tuple[int, int]
            Position of selected cell.

    Returns:
        state: int
            -1 if game still goes on.
            0 if player clicked a mine, so game is over.
            1 if all non-mine cells are visible, so player won.
    """
    if value[pos[0], pos[1]] == -1 and visible[pos[0], pos[1]] == 1:
        return 0
    elif not np.any(np.logical_xor(value != -1, visible == 1)):
        return 1
    else:
        return -1


def get_position(width: int, height: int):
    """
    Get input while playing.
    Not only position, but also can get q for quit, r for restart and h for help.

    Args:
        width: int
            Width of the map.
        height: int
            Height of the map.

    Returns:
        row:
            Selected row.
        column:
            Selected column.
        flag: bool
            True if flag option is selected.
    """
    while True:
        pos = input("Please input grid position (h for help): ")
        if pos in ['Q', 'q']:
            raise Quit
        elif pos in ['R', 'r']:
            raise Restart
        elif pos in ['H', 'h']:
            print("Input position as [column][row] form. (e.g. A3, aC17)\n"
                  "To flag or unflag a cell, type [column][row]f. (e.g. h2f)\n"
                  "Input q to quit, and r to reset.\n")
        else:
            result = re.fullmatch('([A-Za-z]+)([0-9]+)([fF])?', pos)  # Check if input matches format (column)(row)(f)?
            if result is None:
                print("Invalid position.\n")
            else:
                column_str, row_str, is_flag = result.groups()
                column = letters_to_int(column_str)
                row = int(row_str) - 1
                if is_flag is not None:
                    flag = True
                else:
                    flag = False
                if 0 <= column < width and 0 <= row < height:
                    return row, column, flag
                else:
                    print("Position out of range.\n")


def run_game():
    """
    The main scope of the game.
    """
    try:
        while True:
            clear_screen()
            print('')
            width, height, mines = get_mapinfo()

            value = None
            visible = None
            try:
                while True:
                    # draw empty map if map is not generated yet; else draw map
                    if value is None:
                        empty_map = np.zeros((height, width))
                        draw_map(empty_map, np.zeros_like(empty_map))
                    else:
                        draw_map(value, visible)

                    row, column, flag = get_position(width, height)

                    # set value and visible if value is None
                    if value is None:
                        map = init_map(width, height, mines, (row, column))
                        value = get_value(map)
                        visible = np.zeros_like(map, dtype=np.int8)

                    visible = update_visible(value, visible, (row, column), flag)

                    end = check_end(value, visible, (row, column))
                    if end == 0:
                        raise GameOver
                    elif end == 1:
                        visible = np.ones_like(value)
                        raise Win
            except Restart:
                continue
            except (GameOver, Win) as e:
                ans = ''
                while ans not in ['Y', 'y', 'N', 'n']:
                    draw_map(value, visible)
                    ans = input("{} Do you want to play again? (y/n): ".format("Game Over." if isinstance(e, GameOver) else "You Win!"))
                    if ans in ['Y', 'y']:
                        pass
                    elif ans in ['N', 'n']:
                        print('Goodbye.')
                        raise Quit
                    else:
                        print("Please reply only with y or n.")
    except Quit:
        pass


run_game()
