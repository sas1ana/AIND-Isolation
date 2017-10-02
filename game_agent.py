"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import sys


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass
def max_dis(game, player):
    """Maximizing the distance between the players by returning the absolute
    difference between the sum of individual's location vectors.
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
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    p1_pos = game.get_player_location(game.get_opponent(player))
    p2_pos = game.get_player_location(player)
    if p1_pos == None or p2_pos == None:
        return 0.
    return float(abs(sum(p1_pos) - sum(p2_pos)))

def diff_legal_moves(game,player):

    """Returns the difference in the number of moves available to the two players.
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
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    p1_moves = game.get_legal_moves(player)
    p2_moves = game.get_legal_moves(game.get_opponent(player))
    return float(len(p1_moves)- len(p2_moves))


def move_outer(game, player):
    """Returns the difference in the number of moves available to the
    two players while penalizing the moves for the maximizing player against
    the wall and rewarding the moves for the opponenet player against the wall. Not submitted.
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
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
  

    p1_moves = game.get_legal_moves(player)        
    p2_moves = game.get_legal_moves(game.get_opponent(player))
    p1_out_moves = [move for move in p1_moves if move[0] == 0
                                             or move[0] == (game.height - 1)
                                             or move[1] == 0
                                             or move[1] == (game.width - 1)]

    
    p2_out_moves = [move for move in p2_moves if move[0] == 0
                                             or move[0] == (game.height - 1)
                                             or move[1] == 0
                                             or move[1] == (game.width - 1)]
    
    # Rewarding/Penalising on the basis of legal moves in outer layer of the board.
    return float(len(p1_moves) - len(p1_out_moves)
                 - len(p2_moves) + len(p2_out_moves))


def custom_score(game, player):
    # Calling move_outer function as heuristic function
    return move_outer(game, player)

                  
class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

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

        
        if not legal_moves:
            return (-1, -1)
        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        result = None
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                if self.method == "minimax":
                    for depth in range(sys.maxsize):
                        _, move = self.minimax(game, depth)
                        result = move
                if self.method == "alphabeta":
                    for depth in range(sys.maxsize):
                        _, move = self.alphabeta(game, depth)
                        result = move
            else:
                if self.method == "minimax":
                    _, result = self.minimax(game, self.search_depth)
                if self.method == "alphabeta":
                    _, result = self.alphabeta(game, self.search_depth)

        except Timeout:
            return result

        # Return the best move from the last completed search iteration
        return result

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
            
        legal_moves = game.get_legal_moves()       # Getting the legal moves for active player

        if not legal_moves or depth == 0:          # Whether game is over or search depth is over
            return self.score(game, self), (-1, -1)
        
        new_move = None
        if maximizing_player:                # For maximizing player
            new_score = float("-inf")        # Best Score for maximizing is highest score
            for move in legal_moves:
                next_state = game.forecast_move(move)      # Switched the players over here
                score, _ = self.minimax(next_state, depth - 1, False)    # Recurring over next move to get the score 
                if score > new_score:        # Getting the better score
                    new_score, new_move = score, move
                    
        else:                                # For minimizing player
            
            new_score = float("inf")         # Best Score for minimizing is lowest score
            for move in legal_moves:
                next_state = game.forecast_move(move)      # Switched the players over here
                score, _ = self.minimax(next_state, depth - 1, True)    # Recurring over next move to get the score 
                if score < new_score:        # Getting the better score
                    new_score, new_move = score, move
        return new_score, new_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

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

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()            
        
        legal_moves = game.get_legal_moves()  # Getting the legal moves for active player

        if not legal_moves or depth == 0:     # Whether game is over or search depth is over 
            return self.score(game, self), (-1, -1)
        
        # Alpha is the maximum lower bound of possible solutions
        # Alpha is the highest score so far ("worst" highest score is -inf)
        
        # Beta is the minimum upper bound of possible solutions
        # Beta is the lowest score so far ("worst" lowest score is +inf)
        new_move = None
        
        if maximizing_player:           # For maximizing player
            new_score = float("-inf")   # Best Score for maximizing is highest score
            for move in legal_moves:
                next_state = game.forecast_move(move)  # Switched the players over here
                score, _ = self.alphabeta(next_state, depth - 1, alpha, beta, False)  # Recurring over next move to get the score 
                if score > new_score:   # Getting the better score
                    new_score, new_move = score, move
                if new_score >= beta:   # Pruning if possible
                    return new_score, new_move
                alpha = max(alpha, new_score)  # Alpha is updated if possible
        else:                           # For minimizing player
            new_score = float("inf")    # Best Score for minimizing is lowest score
            for move in legal_moves:
                next_state = game.forecast_move(move)  # Switched the players over here like before
                score, _ = self.alphabeta(next_state, depth - 1, alpha, beta, True)   # Recurring over next move to get the score
                if score < new_score:   # Getting the better score
                    new_score, new_move = score, move
                if new_score <= alpha:  # Pruning if possible
                    return new_score, new_move
                beta = min(beta, new_score)    # Beta is updated if possible    
        return new_score, new_move
