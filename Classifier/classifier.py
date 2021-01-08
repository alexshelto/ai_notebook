'''math'''
import numpy as np
'''optimization'''
from scipy import optimize
'''plotting/showing data'''
import matplotlib.pyplot as plt


class Classifier:
    def __init__(self, dataset):
        print('Loading data set')
        self.dataset = np.genfromtxt(dataset, delimiter=',')
        self.y = self.dataset[:, 0]  # labels 
        self.X = self.dataset[:, 1:] # Input data
        self.m = len(self.y)     # number of training examples 
        print('complete')
        self.theta1 = np.zeros((25,self.m)) # theta for calculating L=2. default 25 units in layer 2
        self.theta2 = np.zeros((26, 26))         # theta for output layer. default 25 + 1 units, 26 total classes a-z
        
