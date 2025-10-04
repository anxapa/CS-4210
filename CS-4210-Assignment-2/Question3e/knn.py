#-------------------------------------------------------------------------
# AUTHOR: Chelsea Kathleen Ocampo
# FILENAME: knn.py
# SPECIFICATION: This program uses the K-Nearest Neighbor algorithm (k = 1) to classify
#       a given instance of a certain class given some training data. It then uses the
#       Cross Validation Leave-One-Out method of estimation to calculate the overall
#       error rate of the algorithm trained by the data.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 30 mins - 1 hr
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
import os
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# I was having problems with the program not being to open the file because it could not find it
# so I had to define this function.
def path(filename: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, filename)

# Used later to see if the "trained" algorithm predicts the test set or not
accuracy_list = []

#Reading the data in a csv file using pandas
db = []
df = pd.read_csv(path('email_classification.csv'))
for _, row in df.iterrows():
    db.append(row.tolist())

#Loop your data to allow each instance to be your test set
for i in db:
    X = []
    Y = []
    
    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    for j in db:
        if i == j:
            continue
        
        vector = []
        for k in range(20):
            vector.append(float(j[k]))

        X.append(vector)
        
    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    class_dict = {"ham": 0.0, "spam": 1.0}
    for j in db:
        if i == j:
            continue
        Y.append(class_dict[j[20]])
    
    #Store the test sample of this iteration in the vector testSample
    test_sample = []
    for k in range(20):
        test_sample.append(float(i[k]))

    #Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    #--> add your Python code here
    clf = KNeighborsClassifier(n_neighbors=1, metric="euclidean").fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([test_sample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    accuracy_list.append(class_predicted == class_dict[i[20]])
    

#Print the error rate
#--> add your Python code here
print(f"Error rate: {accuracy_list.count(False) / len(accuracy_list)}")





