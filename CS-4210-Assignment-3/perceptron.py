#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd
import os

# I was having problems with the program not being to open the file because it could not find it
# so I had to define this function.
def path(filename: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, filename)

# Highest accuracy of Perceptron and MLP
highest_acc = {"Perceptron": 0, "MLP" : 0}

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv(path('optdigits.tra'), sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv(path('optdigits.tes'), sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

for lr in n: #iterates over n

    for s in r: #iterates over r

        #iterates over both algorithms
        #-->add your Python code here
        algos = ["Perceptron", "MLP"]

        for algo in algos: #iterates over the algorithms
            # accuracy values of the current algorithm
            accuracy = 0.0
            total_samples = 0
            correct_predictions = 0
            
            #Create a Neural Network classifier
            #if Perceptron then
            #   clf = Perceptron()    #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
            #else:
            #   clf = MLPClassifier() #use those hyperparameters: activation='logistic', learning_rate_init = learning rate,
            #                          hidden_layer_sizes = number of neurons in the ith hidden layer - use 1 hidden layer with 25 neurons,
            #                          shuffle = shuffle the training data, max_iter=1000
            #-->add your Python code here
            if algo == "Perceptron":
                clf = Perceptron(eta0 = lr, shuffle = s, max_iter = 1000)
            else:
                clf = MLPClassifier(activation = 'logistic', learning_rate_init = lr, hidden_layer_sizes = 25, shuffle = s, max_iter=1000)        

            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            #for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #--> add your Python code here
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                total_samples += 1
                if clf.predict([x_testSample]) == y_testSample:
                    correct_predictions += 1

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            #--> add your Python code here
            accuracy = correct_predictions / total_samples
            if accuracy > highest_acc[algo]:
                print(f"Highest {algo} accuracy so far: {accuracy:.2f}, Parameters: learning rate={lr}, shuffle={s}")
                highest_acc[algo] = accuracy











