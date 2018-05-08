# -*- coding: utf-8 -*-
"""
Created on Mon Jan 01 15:29:03 2018

@author: Ganesh
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Conmpute the sigmoid of a vector
def sigmoid(z):  
    a=1/(1+np.exp(-z))    
    return a

# Compute the model shape given the dataset
def getModelShape(X,Y):
    numTraining= X.shape[1] # No of training examples
    numFeats=X.shape[0]     # No of input features
    numHidden=4             # No of units in hidden layer
    numOutput=Y.shape[0]    # No of output units
    # Create a dcitionary of values
    modelParams={"numTraining":numTraining,"numFeats":numFeats,"numHidden":numHidden,"numOutput":numOutput}
    return(modelParams)


# Initialize the model 
# Input : number of features
#         number of hidden units
#         number of units in output
# Returns: Weight and bias matrices and vectors
def initializeModel(numFeats,numHidden,numOutput):
    np.random.seed(2)
    W1=np.random.randn(numHidden,numFeats)*0.01 #  Multiply by .01 
    b1=np.zeros((numHidden,1))
    W2=np.random.randn(numOutput,numHidden)*0.01
    b2=np.zeros((numOutput,1))
    
    # Create a dictionary of the neural network parameters
    nnParameters={'W1':W1,'b1':b1,'W2':W2,'b2':b2}
    return(nnParameters)

# Compute the forward propoagation through the neural network
# Input : Features
#         Weight and bias matrices and vectors
# Returns : The Activation of 2nd layer
#         : Output and activation of layer 1 & 2

def forwardPropagation(X,nnParameters):
    # Get the parameters
    W1=nnParameters["W1"]
    b1=nnParameters["b1"]
    W2=nnParameters["W2"]
    b2=nnParameters["b2"]

    # Compute Z1 of the input layer
    Z1=np.dot(W1,X)+b1
    # Compute the output A1 with the tanh activation function. The tanh activation function
    # performs better than the sigmoid function
    A1=np.tanh(Z1)
    
    # Compute Z2 of the 2nd  layer
    Z2=np.dot(W2,A1)+b2
    # Compute the output A1 with the tanh activation function. The tanh activation function
    # performs better than the sigmoid function
    A2=sigmoid(Z2)    
    cache={'Z1':Z1,'A1':A1,'Z2':Z2,'A2':A2}
    return A2,cache

# Compute the cost
# Input : Activation of 2nd layer
#       : Output from data
# Output: cost
def computeCost(A2,Y):
    m= float(Y.shape[1])
    # Element wise multiply for logprobs
    cost=-1/m *np.sum(Y*np.log(A2) + (1-Y)*(np.log(1-A2)))
    cost = np.squeeze(cost)
    return cost

# Compute the backpropoagation for 1 cycle
# Input : Neural Network parameters - weights and biases
#       # Z and Activations of 2 layers
#       # Input features
#       # Output values Y
# Returns: Gradients
def backPropagation(nnParameters, cache, X, Y):
    numtraining=float(X.shape[1])
    # Get parameters
    W1=nnParameters["W1"]
    W2=nnParameters["W2"]

    #Get the NN cache
    A1=cache["A1"]
    A2=cache["A2"] 

    dZ2 = A2 - Y
    dW2 = 1/numtraining *np.dot(dZ2,A1.T)
    db2 = 1/numtraining *np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T,dZ2), (1 - np.power(A1, 2)))
    dW1 = 1/numtraining* np.dot(dZ1,X.T)
    db1 = 1/numtraining *np.sum(dZ1,axis=1,keepdims=True)

    gradients = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}   
    return gradients

# Perform Gradient Descent
# Input : Weights and biases
#       : gradients
#       : learning rate
#output : Updated weights after 1 iteration
def gradientDescent(nnParameters, gradients, learningRate):
    W1 = nnParameters['W1']
    b1 = nnParameters['b1']
    W2 = nnParameters['W2']
    b2 = nnParameters['b2']
    dW1 = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]
    W1 = W1-learningRate*dW1
    b1 = b1-learningRate*db1
    W2 = W2-learningRate*dW2
    b2 = b2-learningRate*db2
    updatedNNParameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return updatedNNParameters

# Compute the Neural Network  by minimizing the cost 
# Input : Input data X,
#         Output Y
#         No of hidden units in hidden layer
#         No of iterations
# Returns  Updated weight and bias vectors of the neural network
def computeNN(X, Y, numHidden, learningRate, numIterations = 10000):
    np.random.seed(3)
    modelParams = getModelShape(X, Y)
    numFeats=modelParams['numFeats']
    numOutput=modelParams['numOutput']
    
    costs=[]

    nnParameters = initializeModel(numFeats,numHidden,numOutput)
    W1 = nnParameters['W1']
    b1 = nnParameters['b1']
    W2 = nnParameters['W2']
    b2 = nnParameters['b2']  
    # Perform gradient descent
    for i in range(0, numIterations):
        # Evaluate forward prop to compute activation at output layer
        A2, cache =  forwardPropagation(X, nnParameters)        
        # Compute cost from Activation at output and Y
        cost = computeCost(A2, Y)
        # Perform backprop to compute gradients
        gradients = backPropagation(nnParameters, cache, X, Y) 
        # Use gradients to update the weights for each iteration.
        nnParameters = gradientDescent(nnParameters, gradients,learningRate)     
        # Print the cost every 1000 iterations
        if  i % 1000 == 0:
            costs.append(cost)
            print ("Cost after iteration %i: %f" %(i, cost))
    return nnParameters,costs

# Compute the predicted value for a given input
# Input : Neural Network parameters
#       : Input data
def predict(nnParameters, X):
    A2, cache = forwardPropagation(X, nnParameters)
    predictions = (A2>0.5)    
    return predictions



def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1   
    colors=['black','gold']
    cmap = matplotlib.colors.ListedColormap(colors)   
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=cmap)
    plt.show()

