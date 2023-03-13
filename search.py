# search.py
# ---------
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

"""
SOURCE: https://github.com/janluke/cs188/blob/master/cs188/p1_search/search.py
"""


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import abc
from collections import deque
from typing import Callable, Union, Any, NamedTuple, List, Iterator

import util


class SearchProblem(abc.ABC):
    """
    NOTE: I added abc.ABC
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
    You do not need to change anything in this class, ever.
    """

    @abc.abstractmethod
    def getStartState(self):
        """
        Returns the start state for the search problem.
        """

    @abc.abstractmethod
    def isGoalState(self, state) -> bool:
        """
          state: Search state
        Returns True if and only if the state is a valid goal state.
        """

    @abc.abstractmethod
    def getSuccessors(self, state):
        """
          state: Search state
        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """

    @abc.abstractmethod
    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take
        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


# === MY CODE === #

class SearchNode(NamedTuple):
    state: Any
    cost: Union[int, float]
    previous: 'SearchNode'
    action: Any  # the action that bring us to this node from the previous node

    @classmethod
    def root(cls, state, cost=0):
        return SearchNode(state, cost, None, None)

    def reversed_path(self) -> Iterator['SearchNode']:
        node = self
        while node:
            yield node
            node = node.previous

    def plan(self) -> List:
        """ Returns the action plan that leads to this node """
        reversed_plan = [v.action for v in self.reversed_path()]
        return reversed_plan[-2::-1]  # last action is None (start node)


class Fringe(abc.ABC):
    """
    Determines the strategy used for deciding which node to expand next.
    
    Data structure class.
    """

    @abc.abstractmethod
    def popNext(self) -> SearchNode:
        pass

    @abc.abstractmethod
    def push(self, node: SearchNode):
        pass

    @abc.abstractmethod
    def isEmpty(self) -> bool:
        pass

    def isNotEmpty(self) -> bool:
        return not self.isEmpty()


class StackFringe(Fringe):
    def __init__(self):
        self._stack = []

    def popNext(self) -> SearchNode:
        return self._stack.pop()

    def push(self, node: SearchNode):
        self._stack.append(node)

    def isEmpty(self) -> bool:
        return len(self._stack) == 0


class QueueFringe(Fringe):
    """
    The implementation provided by util is actually inefficient, since
    list.insert(0, item) is O(n). Better using a deque.
    """

    def __init__(self):
        self._queue = deque()

    def popNext(self) -> SearchNode:
        return self._queue.popleft()

    def push(self, node: SearchNode):
        return self._queue.append(node)

    def isEmpty(self) -> bool:
        return not bool(self._queue)


class PriorityFringe(Fringe):
    def __init__(self):
        self.fringe = util.PriorityQueue()

    def popNext(self) -> SearchNode:
        return self.fringe.pop()

    def push(self, node: SearchNode):
        self.fringe.push(node, node.cost)

    def isEmpty(self):
        return self.fringe.isEmpty()


AStarHeuristic = Callable[[Any, SearchProblem], Union[int, float]]


class AStarFringe(Fringe):
    def __init__(self,
                 heuristic: AStarHeuristic,
                 problem: SearchProblem):
        self.fringe = util.PriorityQueue()
        self.heuristic = heuristic
        self.problem = problem

    def popNext(self):
        return self.fringe.pop()

    def push(self, node):
        estimatedForwardCost = node.cost + self.heuristic(node.state, self.problem)
        self.fringe.push(node, estimatedForwardCost)

    def isEmpty(self):
        return self.fringe.isEmpty()


def graphSearch(problem: SearchProblem, fringe: Fringe) -> SearchNode:
    """
    Generic graph search algorithm. The fringe determines which specific search
    is implemented (dfs, bsf, ucs or astar).
    Returns the goal node, which is a structure that can be used to obtain every
    useful piece of information about the problem solution (cost, path, actions).
    """
    visited = set()
    startNode = SearchNode.root(problem.getStartState())
    fringe.push(startNode)
    while fringe.isNotEmpty():
        node = fringe.popNext()
        state = node.state

        if problem.isGoalState(state):
            return node

        if state not in visited:
            visited.add(state)

            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:
                    neighNode = SearchNode(
                        successor, cost=node.cost + stepCost,
                        previous=node, action=action
                    )
                    fringe.push(neighNode)
    return None


def findPlanFor(problem: SearchProblem, fringe: Fringe) -> List:
    """ Just a wrapper of graphSearch that returns the action plan. """
    goalNode = graphSearch(problem, fringe)
    return goalNode.plan() if goalNode else None


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    return findPlanFor(problem, StackFringe())


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    return findPlanFor(problem, QueueFringe())


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    return findPlanFor(problem, PriorityFringe())


def nullHeuristic(state, problem: SearchProblem = None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem. This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic: AStarHeuristic = nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    return findPlanFor(problem, AStarFringe(heuristic, problem))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch