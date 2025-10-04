#-------------------------------------------------------------------------
# AUTHOR: Chelsea Kathleen Ocampo
# FILENAME: naive_bayes.py
# SPECIFICATION: This program uses the Naive Bayes algorithm to get the probabilities 
#       of a given instance being classified as a certain class by the model trained
#       by given data. It uses a smoothing of 0.1 for training the data.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 30 mins - 1 hr
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
import os
from sklearn.naive_bayes import GaussianNB
import pandas as pd

# I was having problems with the program not being to open the file because it could not find it
# so I had to define this function.
def path(filename: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, filename)

dbTraining = []
dbTest = []
X = []
Y = []

#Reading the training data using Pandas
df = pd.read_csv(path('weather_training.csv'))
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
features_dict = {"Sunny": 1, "Overcast": 2, "Rain": 3,  # Outlook
                 "Cool": 1, "Mild": 2, "Hot": 3,        # Temperature
                 "Normal": 1, "High": 2,                # Humidity
                 "Weak": 1, "Strong": 2,}               # Wind

for instance in dbTraining:
    feature_vector = []
    for i in range(len(instance)):
        # Skip the first and last element
        if i == 0 or i == 5:
            continue
        
        feature_vector.append(features_dict[instance[i]])
    
    X.append(feature_vector)

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
classes_dict = {"Yes": 1, "No": 2}
Y = [classes_dict[instance[5]] for instance in dbTraining]

#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB(var_smoothing=0.1).fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv(path('weather_test.csv'))
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
#--> add your Python code here
for test in dbTest:
    test_vector = test[1:-1]
    print(f"Test vector: {test_vector}")
    test_vector = [features_dict[feature] for feature in test_vector]
    
#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
    prediction = clf.predict_proba([test_vector])[0]
    print(f"Prediction: [Yes = {prediction[0] * 100:.2f}%, No = {prediction[1] * 100:.2f}%]")
    print("")