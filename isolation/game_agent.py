"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def check_win_loss(game, player):
    """ Check if a player has won or lost the game """
    # determine if player has lost the game
    if game.is_loser(player):
        return float("-inf")
    # determine if player has won the game
    elif game.is_winner(player):
        return float("inf")
    else:
        return None

def check_time(time_left, timer_threshold):
    """ Check time remaining against the timer threshold """
    if time_left < timer_threshold:
        raise SearchTimeout()

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player. This heuristic determines the difference between the
    available moves for each player, with a penalty for moves into a corner
    position before 2/3 of the board is played.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # determine if the player has won or lost
    win_or_loss_bool = check_win_loss(game, player)
    if win_or_loss_bool:
        return win_or_loss_bool

    # create a current state weight variable, which is increased near end game
    if len(game.get_blank_spaces()) < game.width * game.height / 3.:
        current_state = 4
    else:
        current_state = 1

    # define game board corners
    last_col = game.width - 1
    last_row = game.height - 1
    corners = [(0, 0), (0, last_col),
               (last_row, 0), (last_row, last_col)]

    # get a list of available moves for each player
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    # get a list of available corner moves for each player
    own_in_corner = [move for move in own_moves if move in corners]
    opp_in_corner = [move for move in opp_moves if move in corners]

    # Weight score by state and corner moves
    own_moves_weighted = len(own_moves) - (current_state * len(own_in_corner))
    opp_moves_weighted = len(opp_moves) + (current_state * len(opp_in_corner))

    # return difference between the player and opponent's weighted moves
    return float(own_moves_weighted - opp_moves_weighted)

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player. This score is equal to the square of the distance from
    the center of the board to the position of the player. Code for this
    score is based on the center_score method in sample_players.py.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # determine if the player has won or lost
    win_or_loss_bool = check_win_loss(game, player)
    if win_or_loss_bool:
        return win_or_loss_bool

    # calculate a score equal to square of the distance from the center of the
    # board to the position of the player.
    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    return float((h - y)**2 + (w - x)**2)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player. This score is equal to the negative equivalent of the
    opponents available moves. This score will optimize toward minimizing the
    opponent's available moves.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # determine if the player has won or lost
    win_or_loss_bool = check_win_loss(game, player)
    if win_or_loss_bool:
        return win_or_loss_bool

    # calculate number of moves available to the opponent and return the
    # negative equivalent
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(opp_moves) * -1


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.
    ********************  DO NOT MODIFY THIS CLASS  ********************
    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)
    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.
    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        # determine remaining time
        self.time_left = time_left

        # intialize best move
        best_move = (-1, -1)

        try:
            # evaluate best move using minimax
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            # break the loop if the search times out
            # and return the initial best move
            pass
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # raise error if time has run out
        check_time(self.time_left(), self.TIMER_THRESHOLD)
        return self.minimax_move(game, depth)[0]

    def configure_player(self, game):
        """ Return function and value for player """
        if game.active_player == self:
            return max, float("-inf")
        else:
            return min, float("inf")

    def minimax_move(self, game, depth):
        # raise error if time has run out
        check_time(self.time_left(), self.TIMER_THRESHOLD)

        if not depth:
            return (game.get_player_location(self), self.score(game, self))

        # initialize best_move
        best_move =  (-1, -1)

        # configure function for players
        func, value = self.configure_player(game)

        # use iterative deepening to search for the best move up to the given depth
        for move in game.get_legal_moves():
            next_ply = game.forecast_move(move)
            score = self.minimax_move(next_ply, depth - 1)[1]
            if func(value, score) == score:
                best_move = move
                value = score

        return (best_move, value)


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        move = (-1, -1)
        for i in range(1, 10000):
            try:
                move = self.alphabeta(game, i)
            except SearchTimeout:
                return move
        return move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        return self.ab_move(game, depth)[0]

    def configure_player(self, game):
        """ Return function, value, is_alpha for player """
        if game.active_player == self:
            return float("-inf"), max, True
        else:
            return float("inf"), min, False

    def ab_move(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        # raise error if time has run out
        check_time(self.time_left(), self.TIMER_THRESHOLD)

        # initalize best_move
        best_move = (-1, -1)

        if depth == 0:
            return best_move, self.score(game, self)

        # configure values and function for maximizing_player
        value, func, is_alpha = self.configure_player(game)

        # use iterative deepening to search for the best move up to the given depth
        for move in game.get_legal_moves():
            next_ply = game.forecast_move(move)
            score = self.ab_move(next_ply, depth - 1, alpha, beta)[1]
            if score == func(value, score):
                best_move = move
                value = score
            if is_alpha:
                if value >= beta:
                    return best_move, value
                else:
                    alpha = max(value, alpha)
            else:
                if value <= alpha:
                    return best_move, value
                else:
                    beta = min(value, beta)

        return best_move, value
