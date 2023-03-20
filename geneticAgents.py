from pacman import SCARED_TIME
#from game import Agent
from pacmanNN import PacmanControllerModel as pcm
from searchAgents import AnyFoodSearchProblem, AnyGhostSearchProblem, AnyScaredGhostSearchProblem
import search
import random
from game import Agent, Directions, Actions
import util
import numpy as np
import nn


class GeneticAgent(Agent):
    """
    A Pacman Agent which will use a Neural Network to decide its action. Trained
    via a Genetic Algorithm (GA). The Neural Network is referenced via the `policy`
    member variable. The network represents a policy function. 
    """
    def __init__(self, index=0, policy=pcm(), verbose=False):
        self.policy = policy
        # Need to calibrate on first received state
        self.need_calibration = True
        self.powerpill_active = False
        self.powerpill_time_consumed = 0
        self.verbose = verbose
        super().__init__(index)
    
    def calibrate(self, state):
        self.num_food_total = state.getNumFood()
        # 1/2 Perimeter of map
        self.max_path_length = len(state.getWalls().data) + len(state.getWalls().data[0])
        self.need_calibration = False

    def getAction(self, state):
        # Calibrate # food and max path length
        if self.need_calibration:
            self.calibrate(state)
        assert(not self.need_calibration)
            
        # Track powerpill 
        if self.powerpill_active:
            self.powerpill_time_consumed += 1
            if (self.powerpill_time_consumed >= SCARED_TIME):
                self.powerpill_active = False
                self.powerpill_time_consumed = 0
        # Check if we just consumed a powerpill
        for gs in state.getGhostStates():
            if gs.scaredTimer == SCARED_TIME:
                self.powerpill_active = True
                self.powerpill_time_consumed = 0
                break

        features = np.array([[self.calcPillsLeftFeature(state),
                              self.calcPowerLeftFeature(state),
                              self.calcPillLocationFeature(state, Directions.NORTH),
                              self.calcPillLocationFeature(state, Directions.EAST),
                              self.calcPillLocationFeature(state, Directions.SOUTH),
                              self.calcPillLocationFeature(state, Directions.WEST),
                              self.calcGhostLocationFeature(state, Directions.NORTH),
                              self.calcGhostLocationFeature(state, Directions.EAST),
                              self.calcGhostLocationFeature(state, Directions.SOUTH),
                              self.calcGhostLocationFeature(state, Directions.WEST),
                              self.calcScaredGhostLocationFeature(state, Directions.NORTH),
                              self.calcScaredGhostLocationFeature(state, Directions.EAST),
                              self.calcScaredGhostLocationFeature(state, Directions.SOUTH),
                              self.calcScaredGhostLocationFeature(state, Directions.WEST),
                              self.calcMaintainActionFeature(state, Directions.NORTH),
                              self.calcMaintainActionFeature(state, Directions.EAST),
                              self.calcMaintainActionFeature(state, Directions.SOUTH),
                              self.calcMaintainActionFeature(state, Directions.WEST),
                              self.calcAvoidWallsFeature(state, Directions.NORTH),
                              self.calcAvoidWallsFeature(state, Directions.EAST),
                              self.calcAvoidWallsFeature(state, Directions.SOUTH),
                              self.calcAvoidWallsFeature(state, Directions.WEST),
                              ]])
        if (self.verbose):
            print("Got feature vector:")
            print(features)
        # Run features through policy NN to get scores for each direction
        run_result = self.policy.run(nn.Constant(data=features))
        # Softmax to get log probabilities over actions
        softmax_result = nn.SoftmaxLoss.log_softmax(run_result.data)
        # print(softmax_result)
        action_idx = np.argmax(softmax_result)  
        # Map array indices to actions
        if (action_idx == 0):
            action_to_take = Directions.NORTH
        if (action_idx == 1):
            action_to_take = Directions.EAST
        if (action_idx == 2):
            action_to_take = Directions.SOUTH
        if (action_idx == 3):
            action_to_take = Directions.WEST
        if (self.verbose):
            print("Agent wants to take action: ", action_to_take)
        if (action_to_take in state.getLegalPacmanActions()):
            return action_to_take
        else:
            return Directions.STOP
        
    def calcPillsLeftFeature(self, state):
        """
        remove
        
        Returns:
            val = (total # pills - # pills remaining) / total # pills
        """
        return (self.num_food_total - state.getNumFood()) / self.num_food_total
    
    def calcPowerLeftFeature(self, state):
        """
        remove
        
        Returns: 
            IF (powerpill active):
                val = (total duration of powerpill - time consumed of powerpill) / total duration of powerpill
            ELSE:
                val = 0
            
        (NOTE Ryan) No way to immediately read the [time consumed of powerpill]. 
        Instead, this agent tracks this itself. Whenever this agent notices the
        SCARE timers of a ghost resets, it interprets that we must have just 
        eaten a powerpill.
        """
        if self.powerpill_active:
            return (SCARED_TIME - self.powerpill_time_consumed) / SCARED_TIME
        else:
            return 0
    
    def calcPillLocationFeature(self, state, direction):
        """
        Returns:
            val = (max possible path length* - shortest length to a pill for direction) / max possible path length
            
            * For simplicity, I am using the perimeter size of the map layout
        """
        shortest_len = self.shortestLengthForProblem(AnyFoodSearchProblem(state), direction)
        if (shortest_len == -1):
            return 0
        return (self.max_path_length - shortest_len) / self.max_path_length
    
    def calcGhostLocationFeature(self, state, direction):
        """
        Returns:
            val = (max possible path length* - shortest length to a ghost for direction) / max possible path length
            
            * For simplicity, I am using the perimeter size of the map layout
        """
        shortest_len = self.shortestLengthForProblem(AnyGhostSearchProblem(state), direction)
        if (shortest_len == -1):
            return 0
        return (self.max_path_length - shortest_len) / self.max_path_length
    
    def calcScaredGhostLocationFeature(self, state, direction):
        """
        remove
        
        Returns:
            val = (max possible path length* - shortest length to a scared ghost for direction) / max possible path length
            
            * For simplicity, I am using the perimeter size of the map layout
        """
        # If no powerpill, there is no path to scared ghost
        if not self.powerpill_active:
            return 0
        
        shortest_len = self.shortestLengthForProblem(AnyScaredGhostSearchProblem(state), direction)
        if (shortest_len == -1):
            return 0
        
        return (self.max_path_length - shortest_len) / self.max_path_length
    
    def calcEntrapmentFeature(self, state, direction):
        return 0
    
    def calcMaintainActionFeature(self, state, direction):
        """
        remove
        
        Returns:
            IF (given direction is pacman's current direction):
                val = 1
            ELSE:
                val = 0
        """
        if state.getPacmanState().getDirection() == direction:
            return 1
        else:
            return 0
        
    def calcAvoidWallsFeature(self, state, direction):
        """
        remove
        
        Returns:
            IF (given direction is a wall):
                val = 1
            ELSE:
                val = 0
        """
        my_pos = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(direction)

        nextx, nexty = int(my_pos[0] + dx), int(my_pos[1] + dy)

        # If the direction is valid for a move, simulate the move for search
        if state.getWalls()[nextx][nexty]:
            return 1
        else:
            return 0
    
    def shortestLengthForProblem(self, problem, direction):
        """
        Helper method to get shortest path, within a search problem, for a 
        given test direction.
        """
        my_pos = problem.startState
        # print("my current pos")
        # print(my_pos)
        # print("proposed direction")
        # print(direction)
        dx, dy = Actions.directionToVector(direction)
        # print(dx)
        # print(dy)
        nextx, nexty = int(my_pos[0] + dx), int(my_pos[1] + dy)
        # print(nextx)
        # print(nexty)
        # print(problem.walls)
        # If the direction is valid for a move, simulate the move for search
        if problem.walls[nextx][nexty]:
            return -1
        my_pos = (nextx, nexty)
        problem.startState = my_pos
        path = search.ucs(problem)
        # print(path)
        if path != None:
            return len(search.ucs(problem))
        else:
            return -1
    
    def getPolicyModel(self):
        return self.policy
