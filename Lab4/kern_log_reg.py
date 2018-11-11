#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:41:08 2017

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
    index0 = y==1200
    index1 = y==1004
    X_binary = np.concatenate((X[index0,:], X[index1,:]), axis = 0)
    y_binary = np.concatenate((-np.ones(((y[index0]).shape[0], 1 )), np.ones(((y[index1]).shape[0], 1))), axis = 0)
    # Suffle all the data so that they are random
    data_binary = np.random.permutation(np.concatenate((X_binary, y_binary), axis = 1))
    X_binary = data_binary[:, 0:10]
    y_binary = (data_binary[:,-1]).reshape(-1,1)
    return X_binary, y_binary

def rbf_kernel(x_i, X_train, l):
    diff = np.subtract(x_i, X_train)
    norm_diff = LA.norm(diff, axis=1)
    k = np.exp(-np.divide(np.square(norm_diff), pow(l,2)))
    return k

def generate_k(x_s, X, kernel, l):
    number_data = X.shape[0]
#    k = np.zeros((number_data,1))
#    for i in range(number_data):
#        k[i] = rbf_kernel(x_s.T, X, l)
    k = rbf_kernel(x_s.T, X, l)
    k = k.reshape(-1,1)
    return k


def f(x_i,X_train, alpha):
    func_val = np.dot(alpha.T, generate_k(x_i, X_train, 'rbf', 4))
    return func_val

def gamma(y_train, X_train, alpha):
    gamma = np.empty(alpha.shape)
    for i in range(gamma.shape[0]):
        gamma[i] = y_train[i]/(1+exp(-y_train[i]*f(X_train[i,:].T, X_train, alpha)))
    return gamma




file_name = "oakland_part3_an_rf.node_features"
file_name1 = "oakland_part3_am_rf.node_features"


X,y = get_binary_class_data(file_name)
total_number_data = X.shape[0]
number_train_data = int(floor(total_number_data*0.75))
number_tain_data_used = int(floor(number_train_data*0.01))
X_train = X[0:number_train_data,:]
y_train = y[0:number_train_data]
X_test = X[number_train_data:,:]
y_test = y[number_train_data:]
alpha = np.zeros((number_tain_data_used, 1))
y_predict = np.empty(y_test.shape)
Lambda = 0.1
eta = 0.1
ground_success = 0
ground_fail = 0
veg_success = 0
veg_fail = 0
tot_ground = 0
tot_veg = 0
count = 0
for i in range(y_test.shape[0]):
    idx = np.random.randint(number_train_data, size=number_tain_data_used)
    X_train_used = X_train[idx,:]
    print(X_train_used.shape)
    y_train_used = y_train[idx]
    y_predict[i] = f(X_test[i,:].T,X_train_used, alpha)
    alpha = alpha - eta*(Lambda*alpha + gamma(y_train_used, X_train_used, alpha))
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
#    print("ground success ", ground_success, " out of ", tot_ground, " veg success", veg_success, "out of ", tot_veg)
    print("ground success ", ground_success/(tot_ground+0.00000000001), " veg success", veg_success/(tot_veg+0.00000000001))

X,y = get_binary_class_data(file_name1)
X_test = X[0:10000,:]
y_test = y[0:10000]

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
    print("ground success ", ground_success/(tot_ground+0.00001), " veg success ", veg_success/(tot_veg+0.00001))



#k = generate_k(X_test[1,:].T, X_train, 'rbf', 4)
#print(k.shape)