assignments = []

digits   = '123456789'
rows     = 'ABCDEFGHI'
cols     = digits

def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [a+b for a in A for b in B]

# variables for typical sudoku
squares = cross(rows, cols)
unitlist = ([cross(rows, c) for c in cols] +
            [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')])
units = dict((s, [u for u in unitlist if s in u]) for s in squares)
peers = dict((s, set(sum(units[s],[]))-set([s]))for s in squares)

# variables for diagonal sudoku
diag_1 = [a[0]+a[1] for a in zip(rows, cols)]
diag_2 = [a[0]+a[1] for a in zip(rows, cols[::-1])]
diag_unitlist = unitlist + [diag_1, diag_2]
diag_units = dict((s, [u for u in diag_unitlist if s in u]) for s in squares)
diag_peers = dict((s, set(sum(diag_units[s],[]))-set([s]))for s in squares)

def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """
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
    # identify potential naked twins
    p_twins = [b for b in values.keys() if len(values[b]) == 2]
    n_twins = [[b1,b2] for b1 in p_twins for b2 in peers[b1] if set(values[b1])==set(values[b2]) ]

    for i in range(len(n_twins)):
        b_1 = n_twins[i][0]
        b_2 = n_twins[i][1]

        # find peers
        peers1 = set(peers[b_1])
        peers2 = set(peers[b_2])
        peers_int = peers1 & peers2

        # remove values from peers
        for peer_val in peers_int:
            if len(values[peer_val])>2:
                for rm_val in values[b_1]:
                    values = assign_value(values, peer_val, values[peer_val].replace(rm_val,''))
    return values

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
    chars = []
    for c in grid:
        if c in digits:
            chars.append(c)
        if c == '.':
            if show_dots:
                chars.append('.')
            else:
                chars.append(digits)
    assert len(chars) == 81
    return dict(zip(squares, chars))

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

def only_choice(values):
    '''
    Goes through all the units u. If a unit has a certain value d that will only
    fit in one box of u, it will assign d to this box.
    '''

    for unit in diag_unitlist:
        for digit in digits:
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                values[dplaces[0]] = digit
                assign_value(values, dplaces[0], values[dplaces[0]])
    return values

def reduce_puzzle(values):
    '''
    It will apply eliminate and only_choice repeatedly.
    If at any point, there is a box with zero available values, it will return False.
    Otherwise, the loop will stop whenever the sudoku stays the same during one iteration.
    '''

    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    stalled = False
    while not stalled:
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        values = eliminate(values)
        values = only_choice(values)
        values = naked_twins(values)
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        stalled = solved_values_before == solved_values_after
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values

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
    n,s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)
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
    values = search(values)
    return values

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
