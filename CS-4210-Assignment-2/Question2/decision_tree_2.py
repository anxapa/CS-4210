#-------------------------------------------------------------------------
# AUTHOR: Chelsea Kathleen Ocampo
# FILENAME: decision_tree2.py
# SPECIFICATION: This program uses the decision tree algorithm trained by given training data
#       to classify the given instances by a certain class. This program uses three training data
#       with different amount of samples and runs them 10 times each to get the average
#       accuracy of each of them. 
# FOR: CS 4210- Assignment #2
# TIME SPENT: 30 mins - 1 hr
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import os

# I was having problems with the program not being to open the file because it could not find it
# so I had to define this function.
def path(filename: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, filename)

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

#Reading the test data in a csv file using pandas
dbTest = []
df_test = pd.read_csv(path('contact_lens_test.csv'))
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []
    total_accuracy_list = []

    #Reading the training data in a csv file using pandas
    # --> add your Python code here
    df_training = pd.read_csv(path(ds)) 
    for _, row in df_training.iterrows():
        dbTraining.append(row.tolist())

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    feature_dict = {"Young": 1, "Prepresbyopic": 2, "Presbyopic": 3, 
                "Myope": 1, "Hypermetrope": 2,
                "No": 1, "Yes": 2,
                "Reduced": 1, "Normal": 2,
                }
    
    for vector in dbTraining:
        test_vector = []
        for i in range(len(vector)):
            # Skip the last value (class)
            if i == 4:
                continue
            # Create a feature vector with all the feature values of the instance
            test_vector.append(feature_dict[vector[i]])
        # Append the created feature vector to X
        X.append(test_vector)
    
    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    class_dict = {"No": 1, "Yes": 2}
    for vector in dbTraining:
        Y.append(class_dict[vector[4]])
    
    #Loop your training and test tasks 10 times here
    for i in range (10):
        accuracy_list = []
        # fitting the decision tree to the data using entropy as your impurity measure and maximum depth = 5
        # --> add your Python code here
        clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
        clf = clf.fit(X, Y)

        # Read the test data and add this data to dbTest
        #--> add your Python code here

        for data in dbTest:
            # Transform the features of the test instances to numbers following the same strategy done during training,
            # and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            # where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            # --> add your Python code here
            test_vector = []
            for i in range(len(vector)):
                # Skip the last value (class)
                if i == 4:
                    continue
                # Create a feature vector with all the feature values of the instance
                test_vector.append(feature_dict[vector[i]])
            
            class_predicted = clf.predict([test_vector])[0]
            # Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            accuracy_list.append(class_predicted == class_dict[data[4]])
        
        total_accuracy_list.append(accuracy_list.count(True) / len(accuracy_list))

    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    
    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print(f"final accuracy when training on {ds}: {sum(total_accuracy_list) / len(total_accuracy_list)}")




