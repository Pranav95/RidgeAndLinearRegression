

#%matplotlib inline
import numpy as np
from read_dataset import mnist
import pdb
import math
import matplotlib.pyplot as plt




def sigmoid(scores):
    
    return 1/(1 + np.exp(-scores))
        

def step(X, Y, w, b):
    

    scores = np.dot(w.T, X) + b
    A = sigmoid(scores)
    # compute the gradients and cost 
    m = X.shape[1]  # number of samples in the batch\

    cost = -1/m * np.sum(Y * np.log(A) + (1-Y) * (np.log(1-A)))
    cost = np.squeeze(cost)

    temp = (1/m)*(A-Y)
    dw = np.dot(X, temp.T)
    db = np.sum(temp)


    gradients = {"dw": dw,
                 "db": db}
    return cost, gradients

def optimizer(X, Y, w, b, learning_rate, num_iterations):
    
    costs = []

    # update weights by gradient descent
   

    for ii in range(num_iterations):
        cost, gradients = step(X, Y, w, b)
        dw = gradients["dw"]
        db = gradients["db"]

        w = w - learning_rate*dw 
        b = b - learning_rate*db

        if ii % 10 == 0:
            costs.append(cost)
        if ii % 100 == 0:
            print("Cost at iteration %i is: %f" %(ii, cost))
    parameters = {"w": w, "b": b}
    return parameters, gradients, costs

def classify(X, w, b):
    
    scores = np.dot(w.T, X) + b
    A = sigmoid(scores)
    YPred = np.zeros((1,X.shape[1]))

    YPred = 1. * (A > 0.5)
    return YPred
    
def main():
   
    train_data, train_label, test_data, test_label = mnist()

    
    learning_rate = 0.1
    num_iterations = 2000
   
    w = np.zeros((train_data.shape[0],1))
    
    b = 0

   
    parameters, gradients, costs = optimizer(train_data, \
                    train_label, w, b, learning_rate, num_iterations)
    w = parameters["w"]
    b = parameters["b"]



   
    train_Pred = classify(train_data,w,b)
    test_Pred = classify(test_data,w,b)
    
    temp = np.equal(train_Pred,train_label)
    temp2 = np.equal(test_Pred,test_label)


    trAcc = (np.sum(temp)/train_label.shape[1])*100
    teAcc = (np.sum(temp2)/test_label.shape[1])*100
    print("Accuracy for training set is {} %".format(trAcc))
    print("Accuracy for testing set is {} %".format(teAcc))

    h = np.arange(1,201)



    plt.plot(h,costs)
    plt.savefig("Error vs iterations")


if __name__ == "__main__":
    main()
