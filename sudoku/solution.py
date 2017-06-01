#!/usr/bin/env python3
import itertools
assignments = []

digits = '123456789'
rows = 'ABCDEFGHI'
cols = digits

def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [a+b for a in A for b in B]

# variables for regular sudoku
squares = cross(rows, cols)
row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
unitlist = row_units + column_units + square_units
units = dict((s, [u for u in unitlist if s in u]) for s in squares)
peers = dict((s, set(sum(units[s],[]))-set([s]))for s in squares)

# variables for diagonal sudoku
diag_1 = [a[0]+a[1] for a in zip(rows, cols)]
diag_2 = [a[0]+a[1] for a in zip(rows, cols[::-1])]
diag_unitlist = unitlist + [diag_1, diag_2]
diag_units = dict((s, [u for u in diag_unitlist if s in u]) for s in squares)
diag_peers = dict((s, set(sum(diag_units[s],[]))-set([s]))for s in squares)

# from classroom lesson "Encoding the board"
def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}
    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """
    # Find all instances of naked twins
    for unit in unitlist:
        pairs = [box for box in unit if len(values[box]) == 2]
        possible_twins = [list(pair) for pair in itertools.combinations(pairs, 2)]
        # Eliminate the naked twins as possibilities for their peers
        for pair in possible_twins:
            if values[pair[0]] == values[pair[1]]:
                for box in unit:
                    if box != pair[0] and box != pair[1]:
                        for digit in values[pair[1]]:
                            values[box] = values[box].replace(digit,'')
    return values


# from my solution to improved grid values in "Strategy 1: Elimination" lesson
def grid_values(grid, show_dots=False):
    """
    Pulled from Udacity project
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    values = []
    all_numbers = '123456789'
    for value in grid:
        if value == '.':
            values.append(all_numbers)
        elif value in all_numbers:
            values.append(value)
    return {k: v for k, v in zip(squares, values)}

# from lesson "Strategy 1: Elimination"
def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    width = 1+max(len(values[s]) for s in squares)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    print()

# from my solution to Implement eliminate() in "Strategy 1: Elimination"
def eliminate(values):
    '''
    Goes through all the boxes. If a box has only one available value,
    it will remove this value from all the peers of this box.
    '''

    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        digit = values[box]
        for peer in diag_peers[box]:
            values[peer] = values[peer].replace(digit,'')
            assign_value(values, peer, values[peer])
    return values

# from my solution to "Strategy 2: Only Choice" lesson
def only_choice(values):
    '''
    Goes through all the units u. If a unit has a certain value d that will only
    fit in one box of u, it will assign d to this box.
    '''
    for unit in unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                values[dplaces[0]] = digit
    return values

# from my solution to "Constraint Propogation" lesson
def reduce_puzzle(values):
    '''
    It will apply eliminate and only_choice repeatedly.
    If at any point, there is a box with zero available values, it will return False.
    Otherwise, the loop will stop whenever the sudoku stays the same during one iteration.
    '''
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        # Use the Eliminate Strategy
        values =  eliminate(values)
        # Use the Only Choice Strategy
        values =  only_choice(values)
        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values

# from my solution to "Coding the Solution" lession on search
def search(values):
    '''
    Using depth-first search and propagation, try all possible values.
    At any given point, it picks the box with fewer available values
    (if there is more than one, it will pick some box), and propagate over that box.
    '''
    values = reduce_puzzle(values)
    if values is False:
        return False
    if all(len(values[s]) == 1 for s in squares):
        return values
    # Choose one of the unfilled squares with the fewest possibilities
    n,s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)
    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False), return that answer!
    for value in values[s]:
        new_sudoku = values.copy()
        new_sudoku[s] = value
        attempt = search(new_sudoku)
        if attempt:
            return attempt

def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    values = grid_values(grid)
    solved = search(values)
    if solved:
        return solved
    else:
        return False

if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
