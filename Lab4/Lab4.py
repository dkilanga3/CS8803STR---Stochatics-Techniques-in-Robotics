#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 20:27:06 2017

@author: dkilanga
"""


from __future__ import print_function
import os, sys
import numpy as np
import numpy.linalg as LA
from math import pow
from math import floor
from math import sqrt, exp
np.random.seed(2017)

def read_data_from_file(input_file):
    data = []
    if not os.path.isfile(input_file):
        print ("Could not open file: ", input_file)
        return None
    with open(input_file, 'r') as f:
        for line in f:
            float_casted_data = []
            line = (' '.join(line.strip().split())).split()
            if '#' in line:
                continue
            elif not line:
                continue
            else:
                for value in enumerate([float(j) for j in line]):
                    float_casted_data.append(value[1])
                data.append(float_casted_data)
    data = np.array(data)
    return data

def get_binary_class_data(file_name):
    parsed_data = read_data_from_file(file_name)
    X = parsed_data[:,5:]
    y = parsed_data[:,4]
    # Find indices where data correspond to Ground or Facade
    index0 = y==1004
    index1 = y==1200
    X_binary = np.concatenate((X[index0,:], X[index1,:]), axis = 0)
    y_binary = np.concatenate((-np.ones(((y[index0]).shape[0], 1 )), np.ones(((y[index1]).shape[0], 1))), axis = 0)
    # Suffle all the data so that they are random
    data_binary = np.random.permutation(np.concatenate((X_binary, y_binary), axis = 1))
    X_binary = data_binary[:, 0:10]
    y_binary = (data_binary[:,-1]).reshape(-1,1)
    return X_binary, y_binary

def rbf_kernel(x_i, x_j, l):
    k = exp(-pow(LA.norm(x_i-x_j),2)/pow(l,2))
    return k

def generate_Kxx(X, kernel, l):
    number_data = X.shape[0]
    Kxx = np.zeros((number_data, number_data))
    if kernel == 'rbf':
        for i in range(number_data):
            for j in range(number_data):
                Kxx[i,j] = rbf_kernel(np.transpose(X[i,:]), np.transpose(X[j,:]), l)
    return Kxx


def generate_R(X, n, l):
    X = np.transpose(X)
    dimno = X.shape[0]  
    r = np.random.normal(size=(n,dimno))
    print(r.shape)
    s = 1/pow(l,2)
    phix = np.concatenate((np.cos(np.multiply(sqrt(2*s), np.dot(r,X))), np.sin(np.multiply(sqrt(2*s), np.dot(r,X)))), axis=0)
    phix = np.divide(phix, sqrt(n))
    return phix
    

file_name = "oakland_part3_an_rf.node_features"
file_name1 = "oakland_part3_am_rf.node_features"

X,y = get_binary_class_data(file_name)
total_number_data = X.shape[0]
number_train_data = int(floor(total_number_data*0.75))
X_train = X[0:number_train_data,:]
y_train = y[0:number_train_data]
X_val = X[number_train_data:,:]
y_val = y[number_train_data:]
#R_X = generate_R(X_train, 2, 4)
#R = generate_R(X_test, 2, 4)

R_X = generate_R(X_train, 2, 0.15)
R = generate_R(X_val, 2, 0.15)
print (R.shape, R_X.shape)
print(np.dot(R_X, R_X.T).shape)
inv_part = LA.inv(np.dot(R_X, R_X.T) + 0.02*np.identity(R_X.shape[0]))
y_predict = np.dot(np.dot(np.dot(y_train.T, R_X.T), inv_part), R)
y_predict = y_predict.T
ground_success = 0
ground_fail = 0
veg_success = 0
veg_fail = 0
tot_ground = 0
tot_veg = 0

X,y = get_binary_class_data(file_name1)
X_test = X[0:10000,:]
y_test = y[0:10000]

for i in range(y_predict.shape[0]):
    if (y_predict[i] >= 0) and (y_val[i] == 1):
        ground_success += 1
    if (y_predict[i] < 0) and (y_val[i] == -1):
        veg_success += 1
    if (y_predict[i] >= 0) and (y_val[i] == -1):
        veg_fail += 1
    if (y_predict[i] < 0) and (y_val[i] == 1):
        ground_fail += 1
    tot_ground = ground_success + ground_fail
    tot_veg = veg_success + veg_fail
#    print("ground success ", ground_success/(tot_ground+0.00001), " veg success ", veg_success/(tot_veg+0.00001))
    
ground_success = 0
ground_fail = 0
veg_success = 0
veg_fail = 0
tot_ground = 0
tot_veg = 0

R_X = generate_R(X_train, 2, 0.15)
R = generate_R(X_test, 2, 0.15)

inv_part = LA.inv(np.dot(R_X, R_X.T) + 0.02*np.identity(R_X.shape[0]))
y_predict = np.dot(np.dot(np.dot(y_train.T, R_X.T), inv_part), R)
y_predict = y_predict.T        

for i in range(y_predict.shape[0]):
    if (y_predict[i] >= 0) and (y_test[i] == 1):
        ground_success += 1
    elif (y_predict[i] < 0) and (y_test[i] == -1):
        veg_success += 1
    elif (y_predict[i] >= 0) and (y_test[i] == -1):
        veg_fail += 1
    elif (y_predict[i] < 0) and (y_test[i] == 1):
        ground_fail += 1
    tot_ground = ground_success + ground_fail
    tot_veg = veg_success + veg_fail
#    print("ground success ", ground_success/(tot_ground+0.00001), " veg success ", veg_success/(tot_veg+0.00001))


