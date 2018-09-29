#Implementation of various Activation functions in python 
#Inputs are expressed as x
import numpy as np 

#Sigmoid Activation function 
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
	return x * (1 - x)


#ReLU Activation function 
def ReLU(x):
	return np.maximum(0.0, x)

def ReLU_derivative(x):
	if x <= 0: 
		return 0
	else: 
		return 1


#LReLU Activation function 
def LReLU(x):
	a = 0.01
	if x >= 0: 
		return x
	else:
		return a * x

def LReLU_derivative(x):
	a = 0.01
	if x >= 0: 
		return 1
	else:
		return a

#Tanh Activation function 
def tanh(x):
#	return 2 * sigmoid(2x) - 1
	return np.tanh(x)

def dtanh(x):
    return 1. - x * x

#ELU Activation function 
def ELU(x):
	a = 0.01
	if x >= 0:
		return x
	else: 
		return a(np.exp(x) - 1)

def ELU_derivative(x):
	a = 0.01
	if x >= 0:
		return 1 
	else: 
		return ELU(x) + a 


#Swish Activation function 
def SWISH(x):
	b = 1.0
	return x * sigmoid(b * x)

def SWISH_derivative(x):
	b = 1.0
	return b * SWISH(x) + sigmoid(b * x)* (1 - b * SWISH(x))



#softmax Activation function 

def softmax(x):
    e = np.exp(x - np.max(x))  
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:  
        return e / np.array([np.sum(e, axis=1)]).T  
