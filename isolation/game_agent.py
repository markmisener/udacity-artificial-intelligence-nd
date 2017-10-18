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
    of the given player. This score is equal to the ratio of moves available to
    the player to the moves available to the opponent. This heuristic will
    maximize legal moves for the player while minmizing moves for the opponent.

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

    # get all legal moves
    my_moves = len(game.get_legal_moves())
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    total_moves = my_moves + opp_moves

    # calculate the ratio of my moves available to the total moves available
    return float(my_moves)/total_moves


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

        # Initialize best move
        best_move = (-1, -1)

        # Initialize search depth
        search_depth = 1

        # search for best move, or until time expires
        while True:
            try:
                best_move = self.alphabeta(game, search_depth)
                search_depth = search_depth + 1
            except SearchTimeout:
                break
        return best_move


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
        # raise error if time has run out
        check_time(self.time_left(), self.TIMER_THRESHOLD)

        return self.alphabeta_move(game, depth, alpha, beta)[1]

    def alphabeta_move(self, game, depth, alpha, beta, maximize_layer=True):
        """This is a helper function to  implement
        alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"))
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
        tuple
            (score, move)
        """
        # raise error if time has run out
        check_time(self.time_left(), self.TIMER_THRESHOLD)

        # Get legal moves
        legal_moves = game.get_legal_moves()

        # return if no legal moves
        if not legal_moves:
            return game.utility(self), (-1, -1)

        # score the game state if we reach depth of 0
        if depth == 0:
            return self.score(game, self), (-1, -1)

        # initialize best move
        best_move = None

        if maximize_layer:
            # initialize best score
            best_score = float("-inf")
            for move in legal_moves:
                # forecast moves
                next_ply = game.forecast_move(move)
                # calcualte score of the next ply
                score, _ = self.alphabeta_move(next_ply, depth - 1, alpha, beta, False)
                # update best score if the score is greater
                if score > best_score:
                    best_score, best_move = score, move
                if best_score >= beta:
                    break
                # set alpha score
                alpha = max(alpha, best_score)
        else:
            # initialize best score
            best_score = float("inf")
            for move in legal_moves:
                # forecast moves
                next_ply = game.forecast_move(move)
                # calculate score of the next ply
                score, _ = self.alphabeta_move(next_ply, depth - 1, alpha, beta, True)
                # update best score if the score is less than best score
                if score < best_score:
                    best_score, best_move = score, move
                if best_score <= alpha:
                    break
                # set beta score
                beta = min(beta, best_score)
        return best_score, best_move
