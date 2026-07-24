# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        food_list = newFood.asList()
        min_dist = float('inf')
        if len(food_list) > 0:
            for food in food_list: 
                distance = manhattanDistance(newPos, food)
                min_dist = min(min_dist, distance)
        food_score = 1.0 / (min_dist + 0.1)

        ghost_score = 0
        for ghost in newGhostStates:
            ghost_pos = ghost.getPosition()
            ghost_dist = manhattanDistance(newPos, ghost_pos)
            if ghost.scaredTimer == 0:
                if ghost_dist <= 1:
                    ghost_score -= 10000
                else:
                    pass
            else:
                pass

        return successorGameState.getScore() + food_score + ghost_score
 
def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            best_score = float('-inf')
            legalMoves = state.getLegalActions(0)
            for action in legalMoves:
                next_state = state.generateSuccessor(0, action)
                score = min_value(next_state, depth, 1)
                if score > best_score:
                    best_score = score     
            return best_score
        
        def min_value(state, depth, agent_index):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            min_score = float('inf')
            legalMoves = state.getLegalActions(agent_index)
            for action in legalMoves:
                next_state = state.generateSuccessor(agent_index, action)
                if(agent_index == state.getNumAgents() - 1):
                    score = max_value(next_state, depth + 1)
                else:
                    score = min_value(next_state, depth, agent_index + 1)
                min_score = min(score, min_score)
            return min_score

        best_action = None
        best_score = float('-inf')
        legalMoves = gameState.getLegalActions(0)
        for action in legalMoves:
            state = gameState.generateSuccessor(0, action)
            score = min_value(state, 0, 1)

            if score > best_score:
                best_score = score
                best_action = action
        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            best_score = float('-inf')
            legalMoves = state.getLegalActions(0)
            for action in legalMoves:
                next_state = state.generateSuccessor(0, action)
                score = min_value(next_state, depth, 1, alpha, beta)
                best_score = max(best_score, score)
                if best_score > beta:
                    return best_score
                alpha = max(alpha, best_score) 
            return best_score
        
        def min_value(state, depth, agent_index, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            min_score = float('inf')
            legalMoves = state.getLegalActions(agent_index)
            for action in legalMoves:
                next_state = state.generateSuccessor(agent_index, action)
                if(agent_index == state.getNumAgents() - 1):
                    score = max_value(next_state, depth + 1, alpha, beta)
                else:
                    score = min_value(next_state, depth, agent_index + 1, alpha, beta)
                min_score = min(score, min_score)
                if min_score < alpha:
                    return min_score
                beta = min(beta, min_score)

            return min_score

        best_action = None
        best_score = float('-inf')
        legalMoves = gameState.getLegalActions(0)
        alpha = float('-inf')
        beta = float('inf')
        for action in legalMoves:
            state = gameState.generateSuccessor(0, action)
            score = min_value(state, 0, 1, alpha, beta)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)
        return best_action
        

    

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            best_score = float('-inf')
            legalMoves = state.getLegalActions(0)
            for action in legalMoves:
                next_state = state.generateSuccessor(0, action)
                score = exp_value(next_state, depth, 1)
                if score > best_score:
                    best_score = score     
            return best_score
        
        def exp_value(state, depth, agent_index):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            total_score = 0
            legalMoves = state.getLegalActions(agent_index)
            probability = 1.0 / len(legalMoves)
            for action in legalMoves:
                next_state = state.generateSuccessor(agent_index, action)
                if(agent_index == state.getNumAgents() - 1):
                    score = max_value(next_state, depth + 1)
                else:
                    score = exp_value(next_state, depth, agent_index + 1)
                total_score += score * probability
            return total_score
        best_action = None
        best_score = float('-inf')
        legalMoves = gameState.getLegalActions(0)
        for action in legalMoves:
            state = gameState.generateSuccessor(0, action)
            score = exp_value(state, 0, 1)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action





def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    food_list = newFood.asList()
    min_dist = float('inf')
    if len(food_list) > 0:
        for food in food_list: 
            distance = manhattanDistance(newPos, food)
            min_dist = min(min_dist, distance)
    food_score = 1.0 / (min_dist + 0.1)
    food_score -= 20 * len(food_list)

    ghost_score = 0
    for ghost in newGhostStates:
        ghost_pos = ghost.getPosition()
        ghost_dist = manhattanDistance(newPos, ghost_pos)
        if ghost.scaredTimer == 0:
            ghost_score -= 10.0 / (ghost_dist + 0.1)
        else:
            if ghost.scaredTimer > ghost_dist: 
                ghost_score += 2.0 / (ghost_dist + 0.1)
    capsule_list = currentGameState.getCapsules()
    min_dist = float('inf')
    if len(capsule_list) > 0:
        for capsule in capsule_list:
            distance = manhattanDistance(newPos, capsule)
            min_dist = min(min_dist, distance)
    capsule_score = 1.0 / (min_dist + 0.1)
    capsule_score -= 50 * len(capsule_list)

    
    return currentGameState.getScore() + food_score + ghost_score + capsule_score
    
   

# Abbreviation
better = betterEvaluationFunction
