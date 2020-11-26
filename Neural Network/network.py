# Simple vectorized Feed Forward Neural Network using numpy

import numpy as np
import matplotlib.pyplot as plt


# NN with tanh and sigmoid activation functions for binary classification

class NeuralNetwork:
    ''' Simple FFNN for binary classification
        
        ---Arguments---
        X - feature matrix
        Y - output vector
        h_u - number of hidden units in layer 1    
        ---
        '''

    def __init__(self, X, Y, h_u):

        self.X = X
        self.Y = Y
        self.h_u = h_u


    # Output layer activation function
    @staticmethod
    def sigmoid(z):
        return 1/(1 + np.exp(-z))
    
    
    # Hidden layer activation function
    @staticmethod
    def tanh(z):        
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    
    
    
    def weight_initializer(self):
        ''' Randomly initializes weights to be used in the network '''
        
        i_u = self.X.shape[0]   # Input units
        h_u = self.h_u          # Hidden units
        
        weights = {}
        
        np.random.seed(0)
        
        # Multiply by 0.01 to minimize Z and make GD converge faster
        W1 = np.random.randn(h_u, i_u) * 0.01
        B1 = np.zeros((h_u, 1))
        W2 = np.random.randn(1, h_u) * 0.01
        B2 = np.zeros((1,1))
        
        weights['W1'] = W1
        weights['B1'] = B1
        weights['W2'] = W2
        weights['B2'] = B2
        
        return weights


    @staticmethod
    def forward_prop(X, weights):
        ''' Forward propagate in the network '''

        cache = {} # To store variables that can be of later use
        
        W1 = weights['W1']
        B1 = weights['B1']
        W2 = weights['W2']
        B2 = weights['B2']
        
        Z1 = np.dot(W1, X) + B1
        A1 = NeuralNetwork.tanh(Z1) #hidden units of network
        Z2 = np.dot(W2, A1) + B2
        A2 = NeuralNetwork.sigmoid(Z2) #output unit of network
        
        cache['Z1'] = Z1
        cache['A1'] = A1
        cache['Z2'] = Z2
        cache['A2'] = A2
        
        return cache
    
    
    @staticmethod
    def cost(Y, A2):
        ''' Calculates cost function of the model '''
        
        m = Y.shape[1]

        loss = Y * np.log(A2) + (1 - Y) * np.log(1 - A2)
        
        return -1/m * np.sum(loss)
    
    
    @staticmethod
    def backward_prop(X, Y, weights, cache):
        ''' Backpropagates in the network to calculate the derivatives '''
        
        m = Y.shape[1]
        W2 = weights['W2']
        A2 = cache['A2']
        A1 = cache['A1']
        Z1 = cache['Z1']
        
        dZ2 = Y - A2
        dW2 = -1/m * np.dot(dZ2, A1.T)
        dB2 = -1/m* np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = (dZ2 * W2.T) * (1 - NeuralNetwork.tanh(Z1)**2)
        dW1 = -1/m * np.dot(dZ1, X.T)
        dB1 = -1/m * np.sum(dZ1, axis=1, keepdims=True)
        
        derivs = {} # Store calculated derivatives
        
        derivs['dW2'] = dW2
        derivs['dB2'] = dB2
        derivs['dW1'] = dW1
        derivs['dB1'] = dB1
        
        return derivs
    
    
    @staticmethod
    def update_weights(derivs, weights, lr):
        ''' Updates weights of the model using gradient descent '''
        
        W2 = weights['W2']
        B2 = weights['B2']
        W1 = weights['W1']
        B1 = weights['B1']
        
        dW2 = derivs['dW2']
        dB2 = derivs['dB2']
        dW1 = derivs['dW1']
        dB1 = derivs['dB1']
        
        # Updating weigths using Gradient Descent
        W2 = W2 - lr * dW2
        B2 = B2 - lr * dB2
        W1 = W1 - lr * dW1
        B1 = B1 - lr * dB1
        
        # Replace previous weights with updated ones
        weights['W2'] = W2
        weights['B2'] = B2
        weights['W1'] = W1
        weights['B1'] = B1
        
        return weights
    
    
    
    def train_model(self, iters=10000, lr = 0.3, 
                            print_cost=True, plot_cost=False):
        ''' Train neural network 
        
        ---Arguments---
        iters - Number of iterations to train
        lr - Learning rate to be used in training
        print_cost - Print cost function each 1000 iterations
        plot_cost - Plot cost function vs iterations in the end
        ---
        '''
        
        X = self.X
        Y = self.Y
        weights = self.weight_initializer()
        
        # Creating an array to save values of the cost for further plotting
        save_cost = np.empty((iters, 2))
        
        for i in range(iters):
            
            # Forward propagate
            cache = self.forward_prop(X, weights)
            
            # Backward propagate
            derivs = self.backward_prop(X, Y, weights, cache)
            
            # Updating weights
            weights = self.update_weights(derivs, weights, lr)
            
            
            A2 = cache['A2']
            cost = self.cost(Y, A2)
            # Saving cost in each iteration
            save_cost[i] = np.array([[i, cost]])

            # Print cost each 1000 iterations
            if print_cost and i % 1000 == 0:
                
                print('{}-th iteration Cost function is: {}'.format(i, cost))

        # Plot cost function vs iterations
        if plot_cost:
            
            plt.plot(save_cost[:, 0], save_cost[:, 1])
            
            # Set plot labels
            plt.ylabel('Cost function')
            plt.xlabel('# of Iterations')
            plt.grid(True)
            
        return A2, weights



    def accuracy(self, A2, Y=None):
        ''' Calculate accuracy of the model '''
        
        if Y is None:
            Y = self.Y
        
        # Setting values of the output unit to 1's and 0's
        A2 = (A2 >= 0.5)
        
        return (np.sum(Y * A2) + np.sum((1 - Y) * (1 - A2))) / Y.shape[1]


    
    def predict(self, X, weights):
        ''' Predict test set values '''
        
        cache = self.forward_prop(X, weights)
        
        # Output of the network
        A2 = cache['A2']
        
        return A2