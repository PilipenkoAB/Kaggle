# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:00:52 2020

@author: Александр
"""
#score = pd.read_csv('score.csv')





import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train.csv')

#dataset_test = pd.read_csv('test.csv')
#                   VAR 1
# DELETE AGE EMPTY ROWS
#dataset = dataset.dropna(subset=['Age'])  
#dataset['Age'] = dataset['Age'].fillna(0)
dataset['Age'] = dataset['Age'].fillna(28)


dataset['Cabin'] = dataset['Cabin'].fillna(0) #No information about cabin

#dataset['Cabin'][61] = 1
#dataset['Cabin'][829] = 1

for i in range(len(dataset)):
    if dataset['Cabin'][i] != 0 :
        dataset['Cabin'][i] = 1


#add embarked
dataset = dataset.dropna(subset=['Embarked']) 

#X = dataset.iloc[:, [2,4,5,6,7,9]].values #pclass
#y = dataset.iloc[:, 1].values   # Dependent variable (result)


#add embarked
X = dataset.iloc[:, [2,4,5,6,7,9,10,11]].values #pclass
y = dataset.iloc[:, 1].values   # Dependent variable (result)


# NORMAL
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) # number is column number to encode


labelencoder_X_2 = LabelEncoder()
X[:, 7] = labelencoder_X_2.fit_transform(X[:, 7]) # number is column number to encode

#---
# Splitting the dataset into the Training set and Test set !(only change test_size to desirable)!
#---
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#
# Feature Scaling 
#
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#X[1] = scaler.fit_transform(X[1])



#------
#------
#Part 2 - making the ANN
#------
#------

#Impoting the Keras packadges and modules
import keras
from keras.models import Sequential #Initial ANN
from keras.layers import Dense      #Create layers in ANN

from keras.layers import Dropout      #Dropout (for part 4) - fixing the variance to input and hidden layers
#--
#Creating the ANN
#--

#Initialising the ANN
classifier = Sequential()

#Adding Imput layer and the first hidden layer
classifier.add(Dense(output_dim = 8,    #strategy - take (amount of colums + 1)/2  || units = 6
                     init = 'uniform',  #randomly initiate start weights close to zero || kernel_initilizer = "uniform"
                     activation = 'relu', # rectifier function for 1st hidden layer || activation = "relu"
                     input_dim = 8 # amount of Independent Variables || input_dim = 11
                     )
                )
#classifier.add(Dropout(p=0.5))  #p = from 0 to 1. start from 0.1..0.2.. 0.5 to fix variance
#Addinng the second hidden layer
classifier.add(Dense(output_dim = 8,    #strategy - take (amount of colums + 1)/2  || units = 6
                     init = 'uniform',  #randomly initiate start weights close to zero || kernel_initilizer = "uniform"
                     activation = 'relu' # rectifier function for 2st hidden layer || activation = "relu"
                     )
                )
   
    
#classifier.add(Dropout(p=0.5))  #p = from 0 to 1. start from 0.1..0.2.. 0.5 to fix variance
#classifier.add(Dense(output_dim = 8,    #strategy - take (amount of colums + 1)/2  || units = 6
#                     init = 'uniform',  #randomly initiate start weights close to zero || kernel_initilizer = "uniform"
#                     activation = 'relu' # rectifier function for 2st hidden layer || activation = "relu"
#                     )
#                )
#classifier.add(Dropout(p=0.1))  #p = from 0 to 1. start from 0.1..0.2.. 0.5 to fix variance
#classifier.add(Dense(output_dim = 5,    #strategy - take (amount of colums + 1)/2  || units = 6
#                     init = 'uniform',  #randomly initiate start weights close to zero || kernel_initilizer = "uniform"
#                     activation = 'relu' # rectifier function for 2st hidden layer || activation = "relu"
#                     )
#                )
#Adding the output layer
classifier.add(Dense(output_dim = 1,    #need to get only 1 output value  || units = 1     \\ if more than 0\1 result add number 2(0\1\2) or 3(0\1\2\3) etc
                     init = 'uniform',  #randomly initiate start weights close to zero || kernel_initilizer = "uniform"
                     activation = 'sigmoid' # sigmoid activation function for output layer || activation = "sigmoid"  \\ if more than 0\1 result - "softmax"
                     )
                )
#Compiling the ANN
classifier.compile(optimizer = 'adam',  #Optimisation function for finding optimal weights
                   loss = 'binary_crossentropy', # SSR function \\ if more than 2 dependent variables loss = 'categorical_crossentropy'
                   metrics = ['accuracy']  #evaluate function to improve the model performance
                   )


#--
#Using the ANN
#--

#Putting the ANN to the test set (!can change batch_size and epochs to find betterresults!)
classifier.fit(X_train, y_train, batch_size=10, epochs=1000)



from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


def build_classifier(optimizer):  # Structure of the ANN - the SAME that was above (copy) #if you want to tune parameters except batch and epoch - add them in () of function like optimizer
    classifier = Sequential()
    classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu', input_dim = 6))
    classifier.add(Dropout(p=0.15)) 
    classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p=0.1)) 
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

#(!can change batch_size and epochs to find betterresults! and inside function - ANN structure)
classifier = KerasClassifier(build_fn = build_classifier)

# change parameters to tune
parameters = {'batch_size': [ 25, 32],
              'epochs': [500],
              'optimizer': ['adam', 'rmsprop']}

# don't change anything
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)

#result - best parameters and accuracy
best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_





#

predictions = pd.read_csv('test.csv')
#predictions = predictions.dropna(subset=['Age'])
#predictions['Age'] = predictions['Age'].fillna(0)
#predictions['Age'] = predictions['Age'].fillna(predictions['Age'].median())
predictions['Age'] = predictions['Age'].fillna(28)

predictions['Cabin'] = predictions['Cabin'].fillna(0) #No information about cabin
for i in range(len(predictions)):
    if predictions['Cabin'][i] != 0 :
        predictions['Cabin'][i] = 1

#X_pred = predictions.iloc[:, [1,3,4,5,6,8]].values #pclass
X_pred = predictions.iloc[:, [1,3,4,5,6,8,9,10]].values #pclass

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X_pred[:, 1] = labelencoder_X_1.fit_transform(X_pred[:, 1]) # number is column number to encode


labelencoder_X_2 = LabelEncoder()
X_pred[:, 7] = labelencoder_X_2.fit_transform(X_pred[:, 7]) # number is column number to encode

X_pred = sc.transform(X_pred)


y_pred = classifier.predict(X_pred)  
y_pred = (y_pred > 0.5)  #translating float 0-1 numbers to true\false when if >0.5 = true
 
pd.DataFrame(y_pred, columns =['Survived'])

result =  pd.concat([predictions['PassengerId'], pd.DataFrame(y_pred, columns =['Survived'])], axis=1, sort=False)
result.to_csv('out_3.csv',index=False)






