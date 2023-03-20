# -*- coding: utf-8 -*-
import nn
import numpy as np

class PacmanControllerModel(object):
    """
    A model for controlling Pacman's actions,
    
    An input to this model will be the state of the pacman game, described in
    a 22-dimesnional vector for the purposes of this model. Each entry in the 
    vector is a floating point number between 0 and 1. 
    
    The output of this model will be a 4-dimensional vector corresponding to the
    classes of possible actions that Pacman can take at any moment: LEFT, FORWARD,
    BACKWARD, RIGHT. The goal is then for this model to learn a policy by which 
    it can guide Pacman successfully through a game. At each step, the model will
    dictate Pacman's action based on the state of the game.
    
    The dimensionality of this model's input and hidden layers is customizable
    via parameters in the constructor. Weights can be set via the constructor as well.
    
    Inputs:
        in_dim: Integer representing number of units (dimensionality) for the 
                input layer.
        hidden_dim: Integer representing number of units (dimensionality) for 
                the hidden layer.
        weights: A 1-dimensional numpy array representing weights to be used 
                for the interlayer connections. Format of the array is expected
                to match the returned format from `getWeights()`
                
    """
    def __init__(self, in_dim=12, hidden_dim=24, weights=np.array([])):
        assert(in_dim > 0)
        assert(hidden_dim > 0)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        # Initialize model parameters
        self.W1 = nn.Parameter(in_dim,hidden_dim)
        self.b1 = nn.Parameter(1,hidden_dim)
        self.W2 = nn.Parameter(hidden_dim,4)
        self.b2 = nn.Parameter(1,4)
        # If custom weights are provided, divide and set them
        if weights.size > 0:
            weights_length = in_dim * hidden_dim + hidden_dim + hidden_dim * 4 + 4
            assert(weights.size == weights_length)
            # Slice out respective weights. Make sure to make a copy
            new_W1 = weights[:(in_dim * hidden_dim)].copy()
            new_W1 = new_W1.reshape((in_dim, hidden_dim))
            new_b1 = weights[(in_dim * hidden_dim):(in_dim * hidden_dim) + hidden_dim].copy()
            # Biases need to have second dimension added
            new_b1 = np.array([new_b1])
            new_W2 = weights[(in_dim * hidden_dim) + hidden_dim: (in_dim * hidden_dim) + hidden_dim + hidden_dim * 4].copy()
            new_W2 = new_W2.reshape((hidden_dim, 4))
            new_b2 = weights[-4:].copy()
            new_b2 = np.array([new_b2])
            # (NOTE Ryan): Setting data member variables directly. Not ideal, but I would like
            # to avoid editing nn.py
            self.W1.data = new_W1
            self.b1.data = new_b1 # np.array([[0,0,0,0]]) # new_b1
            self.W2.data = new_W2
            self.b2.data = new_b2 # np.array([[0,0,0,0]]) # new_b2
        

    def run(self, x):
        """
        Runs a forward pass of the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x input_dimension)
        Output:
            A node with shape (batch_size x 4) containing predicted scores
                (logits)
        """
        # print("weights 1: ")
        # print(self.W1)
        # print("bias 1: ", self.b1)
        # print(self.b1)
        Z1 = nn.AddBias(nn.Linear(x,self.W1),self.b1)
        # print("Z1")
        A1 = nn.ReLU(Z1);
        Z2 = nn.AddBias(nn.Linear(A1,self.W2),self.b2)
        return Z2
    
    def getWeights(self):
        """
        Get the weights of this model as a single vector. Weights are 
        concatenated in a "top down, left to right" fashion. 
        
        More specifically, the return value is a numpy array representing a 
        row-vector of all weights in the network. Weights are concatenated as
        [W1_0 ... W1_n b1 W2_0 ... W2_n b2], where W1_0, for example, refers to
        the 1 x hidden_dim vector of weights for the first input node. 
        """
        first_layer = np.concatenate((self.W1.data, self.b1.data), axis=None)
        out_layer = np.concatenate((self.W2.data, self.b2.data), axis=None)
        
        return np.concatenate((first_layer, out_layer), axis=None)
    
    def initFlatWeights(self):
        """
        Helper to initialize all weights and biases in a NN. Uses a rough 
        Xavier initialization where all biases are set to 0.
        """
        w1 = np.random.uniform(low=-np.sqrt(1/(self.in_dim)), high=np.sqrt((1/(self.in_dim))), size=self.in_dim * self.hidden_dim).round(3)
        b1 = np.zeros(self.hidden_dim)
        w2 = np.random.uniform(low=-np.sqrt(1/(self.hidden_dim)), high=np.sqrt((1/(self.hidden_dim))), size=self.hidden_dim * 4).round(3)
        b2 = np.zeros(4)
        
        weights = np.concatenate((w1, b1, w2, b2), axis=None).round(3)
        return weights
    
    def initSingleWeight(self):
        """
        Helper to initialize a single weight in a NN
        """
        return round(np.random.uniform(low=-np.sqrt(1/self.hidden_dim), high=np.sqrt(1/self.hidden_dim)), 3)
        
    
    
    
# ------------ TEST BED ---------------#
"""
custom_weights = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
myPacman = PacmanControllerModel(in_dim=2, hidden_dim=4, weights=custom_weights)
print("W1")
print(myPacman.W1.data)
print(myPacman.b1.data)
#print("weights:")
#print(myPacman.getWeights())
run_result = myPacman.run(nn.Constant(data=np.array([[1.0, 2.0]])))
print(run_result.data)
softmax_result = nn.SoftmaxLoss.log_softmax(run_result.data)
print(softmax_result)
print(np.exp(softmax_result))
print(np.sum(np.exp(softmax_result)))
print(np.argmax(np.exp(softmax_result)))
"""




