# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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
from collections import defaultdict
from typing import Dict, Any, Set, Tuple

#from my_util import getIndexOfMax
import util
from mdp import MarkovDecisionProcess

from learningAgents import ValueEstimationAgent


def _computeQValue(
        mdp: MarkovDecisionProcess,
        values: Dict, discount: float,
        state, action):
    """
    $ Run a step of Q-value iteration: it computes Q[k+1](s, a) given
    the last computed values v[k].
    """
    return sum(
        prob * (mdp.getReward(state, action, nextState) + discount * values[nextState])
        for nextState, prob in mdp.getTransitionStatesAndProbs(state, action)
    )


def _computeQValues(
        mdp: MarkovDecisionProcess,
        values: Dict, discount: float,
        state):
    return [
        (_computeQValue(mdp, values, discount, state, action), action)
        for action in mdp.getPossibleActions(state)
    ]


def _computeValueAndGreedyActions(
        mdp: MarkovDecisionProcess,
        values: Dict, discount: float,
        state) -> Tuple[float, Tuple]:
    """
    Returns the value of a state (the max_a Q(s, a)) and the greedy action(s)
    (argmax_a Q(s, a)).
    """
    qValues = _computeQValues(mdp, values, discount, state)
    if not qValues:
        return 0, ()
    maxQValue, _ = max(qValues)
    maxActions = tuple(action for qValue, action in qValues if qValue == maxQValue)
    return maxQValue, maxActions


class ValueIterationAgent(ValueEstimationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A ValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs value iteration
    for a given number of iterations using the supplied
    discount factor.

    #§ This class could be computed more efficiently, even without
    #§ storing all Q-values in memory, just by keeping track of the
    #§ optimal action in a state when updating its value.
    """
    def __init__(self, mdp: MarkovDecisionProcess, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = defaultdict(float)   #§ don't need util.Counter
        self.policy = None                 #§ cached policy
        self.runValueIteration()

    def runValueIteration(self):
        """
        § Runs value iteration for self.iterations iterations:

            v[k](s) = max[a] Q[k](s, a)
            Q[k+1](s, a) = sum[s'] T(s, a, s') * [r(s, a, s') + discount*v[k](s')]

        and returns the resulting policy.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        discount = self.discount
        states = mdp.getStates()
        prevValues = defaultdict(int)
        currValues = self.values
        for k in range(self.iterations):
            prevValues, currValues = currValues, prevValues
            for state in states:
                qValues = _computeQValues(mdp, prevValues, discount, state)
                if not qValues:
                    currValues[state] = 0
                else:
                    maxQValue = max(qValues)[0]
                    currValues[state] = maxQValue
        self.values = currValues

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        return _computeQValue(
            self.mdp, self.values, self.discount, state, action)

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #§ Note: the policy could be computed
        if self.policy is None:
            self.policy = policy = dict()
            for state in self.mdp.getStates():
                qValues = _computeQValues(self.mdp, self.values, self.discount, state)
                policy[state] = max(qValues)[1] if qValues else None
        return self.policy[state]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp: MarkovDecisionProcess, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        states = mdp.getStates()
        for k in range(self.iterations):
            state = states[k % len(states)]
            possibleActions = mdp.getPossibleActions(state)
            if not possibleActions:
                continue
            else:
                qValues = _computeQValues(mdp, self.values, self.discount, state)
                maxQValue = max(qValues)[0]
                self.values[state] = maxQValue


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A PrioritizedSweepingValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs prioritized sweeping value iteration
    for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
        Your prioritized sweeping value iteration agent should take an mdp on
        construction, run the indicated number of iterations,
        and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        predecessors = self._getPredecessors(self.mdp)

        queue = util.PriorityQueue()   #§ min priority queue
        for state in states:
            qValues = _computeQValues(
                self.mdp, self.values, self.discount, state)
            if not qValues:
                continue
            new_predecessor_value = max(qValues)[0]
            absError = abs(self.values[state] - new_predecessor_value)
            queue.push(state, -absError)   #§ bigger the error, higher the priority

        for t in range(self.iterations):
            if queue.isEmpty():
                break
            state = queue.pop()
            self.values[state] = _computeValueAndGreedyActions(
                self.mdp, self.values, self.discount, state)[0]
            for predecessor in predecessors[state]:
                qValues = _computeQValues(
                    self.mdp, self.values, self.discount, predecessor)
                if not qValues:
                    continue
                new_predecessor_value = max(qValues)[0]
                absError = abs(self.values[predecessor] - new_predecessor_value)
                if absError > self.theta:
                    #§ bigger the error, higher the priority
                    queue.update(predecessor, -absError)

    @staticmethod
    def _getPredecessors(mdp: MarkovDecisionProcess) -> Dict[Any, Set]:
        states = mdp.getStates()
        predecessors = {s: set() for s in states}
        for state in states:
            for action in mdp.getPossibleActions(state):
                for successor, prob in mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessors[successor].add(state)
        return predecessors
