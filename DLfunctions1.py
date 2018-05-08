# -*- coding: utf-8 -*-
"""
Created on Tue May  8 19:32:14 2018

@author: Ganesh
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define the sigmoid function
def sigmoid(z):  
    a=1/(1+np.exp(-z))    
    return a

# Initialize
def initialize(dim):
    w = np.zeros(dim).reshape(dim,1)
    b = 0   
    return w

# Compute the loss
def computeLoss(numTraining,Y,A):
    loss=-1/numTraining *np.sum(Y*np.log(A) + (1-Y)*(np.log(1-A)))
    return(loss)

# Execute the forward propagation
def forwardPropagation(w,b,X,Y):
    # Compute Z
    Z=np.dot(w.T,X)+b
    # Determine the number of training samples
    numTraining=float(len(X))
    # Compute the output of the sigmoid activation function 
    A=sigmoid(Z)
    #Compute the loss
    loss = computeLoss(numTraining,Y,A)
    # Compute the gradients dZ, dw and db
    dZ=A-Y
    dw=1/numTraining*np.dot(X,dZ.T)
    db=1/numTraining*np.sum(dZ)
    
    # Return the results as a dictionary
    gradients = {"dw": dw,
             "db": db}
    loss = np.squeeze(loss)
    return gradients,loss

# Compute Gradient Descent    
def gradientDescent(w, b, X, Y, numIerations, learningRate):
    losses=[]
    idx =[]
    # Iterate 
    for i in range(numIerations):
        gradients,loss=forwardPropagation(w,b,X,Y)
        #Get the derivates
        dw = gradients["dw"]
        db = gradients["db"]
        w = w-learningRate*dw
        b = b-learningRate*db

        # Store the loss
        if i % 100 == 0:
            idx.append(i)
            losses.append(loss)      
        # Set params and grads
        params = {"w": w,
                  "b": b}  
        grads = {"dw": dw,
                 "db": db}
    
    return params, grads, losses,idx

# Predict the output for a training set 
def predict(w,b,X):
    size=X.shape[1]
    yPredicted=np.zeros((1,size))
    Z=np.dot(w.T,X)
    # Compute the sigmoid
    A=sigmoid(Z)
    for i in range(A.shape[1]):
        #If the value is > 0.5 then set as 1
        if(A[0][i] > 0.5):
            yPredicted[0][i]=1
        else:
        # Else set as 0
            yPredicted[0][i]=0

    return yPredicted

#Normalize the data   
def normalize(x):
    x_norm = None
    x_norm = np.linalg.norm(x,axis=1,keepdims=True)
    x= x/x_norm
    return x

   