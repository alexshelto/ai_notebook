'''math'''
import numpy as np
'''optimization'''
from scipy import optimize
'''plotting/showing data'''
import matplotlib.pyplot as plt

#todo:
# K and L
# input layer size, hidden layer size
class Classifier:
    def __init__(self, dataset, epsilon=1.7):
        print('Loading data set')
        self.dataset = np.genfromtxt(dataset, delimiter=',')
        self.y = self.dataset[:, 0]  # labels 
        self.X = self.dataset[:, 1:] # Input data
        self.m =  len(self.y)     # number of training examples 
        self.lam = 0 # lambda
        self.input_layer_size = len(self.X[0])
        self.hidden_layer_size = 25
        self.num_labels = 26
        print('complete')

        # creating theta params. the (+1) in the thetas is to account for fitting the bias
        self.theta1 = np.random.rand(self.hidden_layer_size, self.input_layer_size + 1) # theta for calculating L=2. default 25 units in layer 2
        self.theta2 = np.random.rand(self.hidden_layer_size + 1, self.num_labels)         # theta for output layer. default 25 + 1 units, 26 total classes a-z
        self.params = np.r_[self.theta1.T.flatten(), self.theta2.T.flatten()]

    def show(self, x, name):
        print(f'Size of {name}: {np.shape(x)}')

    def sigmoid(self, z):
        return ( (1 / (1 + np.exp(-z))) )

    def feed_forward(self):
        '''propogate forward calculating a2, z2, a3, z3'''
        # A1 is correct
        a1 = np.c_[np.ones((self.m, 1)), self.X] # (examples x input pixels + 1(bias))
        self.show(a1, 'a1')
        self.show(self.theta1, 'theta1')
        
        # layer 2: hidden layer
        z2 = a1.dot(self.theta1.T)
        a2 = self.sigmoid(z2)
        a2 = np.c_[np.ones((np.shape(a2)[0], 1)), a2] # bias for hidden layer
      
        # layer 3: output layer
        self.show(self.theta2, 'theta2')
        z3 = a2.dot(self.theta2.T)
        a3 = self.sigmoid(z3) # a3 = h_theta(x)
        return (a1, a2, a3, z2, z3)




    def cost_function(self):
        # FWD Propogate
        a1, a2, a3, z2, z3 = self.feed_forward()
        # Compute Cost
        J = 
        # Back Propogate
