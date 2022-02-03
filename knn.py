# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:21:08 2022

@author: Basheer
"""

"K-Nearest Neighbour"

import numpy as np
import pandas as pd
from statistics import mode

train=pd.read_csv('trainingdata.csv')
x=(train.iloc[:,0])
y=(train.iloc[:,1])
c=train.iloc[:,3]

k=5
success=0
test=pd.read_csv('testdata.csv')
xtest=test.iloc[:,0]
ytest=test.iloc[:,1]

for j in range(len(xtest)):
    dist=[[0]*len(x), [0]*len(x)]
    npx=xtest[j]
    npy=ytest[j]
    for i in range(1,len(x)):
        dist[0][i]=(npx-float(x[i]))**2+(npy-float(y[i]))**2
        dist[1][i]=str(c.iloc[i])
    #dist = dist[dist[0, :].argsort()]
    list1, list2 = zip(*sorted(zip(dist[0], dist[1])))
    klist=list2[:k]
    npc=mode(klist)
    print('Atom: ',test.iloc[j,2],'Classified as: ',npc)
    print('The original classification is: ',test.iloc[j,3])
    if (npc==test.iloc[j,3]):
        print('Succesdfull Prediction')
        success=success+1
    else:
        print('Failed Prediction')
Accuracy=success/len(xtest)
print('Thus Accuracy of our model is :',Accuracy*100,'%')
if (Accuracy<0.9):
    print('We need more training data to train our model accurately')
else:
    print('Our Model is Accurate enough!')