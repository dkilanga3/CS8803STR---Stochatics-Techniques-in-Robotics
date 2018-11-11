#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 14:55:02 2017

@author: pushyami
"""


from __future__ import print_function
import os
import numpy as np
from math import pow
from math import floor
from math import sqrt
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
np.random.seed(2017)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

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
#    print(data[:,0], data[:,1], data[:,2])

    return data

def Visualize (data):
    xc = data[:,0]
    yc = data[:,1]
    zc = data[:,2]
    node_id = data[:,3]
    y = data[:,4]
#    X = data[:,5:]
        
    xc = np.floor(xc)
    yc = np.floor(yc)
    zc = np.floor(zc)
    
    min_x = int(min(xc))
    max_x = int(max(xc))
    min_y = int(min(yc))
    max_y = int(max(yc))
    min_z = int(min(zc))
    max_z = int(max(zc))
    
    ax.set_xlabel('X')
    ax.set_xlim(min_x, max_x)
    ax.set_ylabel('Y')
    ax.set_ylim(min_y, max_y)
    ax.set_zlabel('Z')
    ax.set_zlim(min_z, max_z)
    
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 100))
                
    for i in node_id:
#        if i%5 == 0:
        if i%50 == 0:
            if y[i] == 1004:
                ax.scatter(xc[i], yc[i], zc[i], c='g', marker='^')
            elif y[i] == 1100:
                ax.scatter(xc[i], yc[i], zc[i], c='r', marker='o')
            elif y[i] == 1103:
                ax.scatter(xc[i], yc[i], zc[i], c='b', marker='o')
            elif y[i] == 1200:
                ax.scatter(xc[i], yc[i], zc[i], c=colors[57], marker='o')
            elif y[i] == 1400:
                ax.scatter(xc[i], yc[i], zc[i], c=colors[25], marker='^')

#    plt.show()
    
    
    
    
def Visualize_single (data, dt, name):
    data = np.array(data)
    xc = np.floor(data[0])
    yc = np.floor(data[1])
    zc = np.floor(data[2])
    y = np.floor(data[4])
#    print(y)
#    X = data[:,5:]
        
#    print(xc, yc, zc)
    
    xc = np.floor(xc)
    yc = np.floor(yc)
    zc = np.floor(zc)
    
#    print(xc, yc, zc)
    
    ax.set_xlabel('X')
    ax.set_xlim(170, 250)
    ax.set_ylabel('Y')
    ax.set_ylim(170, 270)
    ax.set_zlabel('Z')
    ax.set_zlim(-5, 30)
    
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 100))
    
#    plt.ion()
    
#    print (y)
                
    if y == 1004:
        ax.scatter(xc, yc, zc, c='g', marker='^')
    elif y == 1100:
        ax.scatter(xc, yc, zc, c='r', marker='o')
    elif y == 1103:
        ax.scatter(xc, yc, zc, c='b', marker='o')
    elif y == 1200:
        ax.scatter(xc, yc, zc, c=colors[57], marker='o')
    elif y == 1400:
        ax.scatter(xc, yc, zc, c=colors[25], marker='^')  
#    plt.draw()
    if name == 'gd':
        name_file = "images_GD"
    if name == 'lsvm':
        name_file = "images_LSVM"
    if name == 'blr':
        name_file = "images_BLR"
    plt.savefig(os.path.join(basePath, name_file, 'fig{0}.jpg'.format(dt)))
#    plt.show() 
    

def get_binary_class_data(file_name, class1, class2):
    parsed_data = read_data_from_file(file_name)
    xc = parsed_data[:,0]
    yc = parsed_data[:,1]
    zc = parsed_data[:,2]
    node_id = parsed_data[:,3]
    y = parsed_data[:,4]
    X = parsed_data[:,5:]
    
#    print(np.floor(xc))

    class1_id, class2_id = 0, 0
    
    # Find indices where data correspond to class1 or Facade
    if class1 == 'Veg':
        class1_id = 1004
    elif class1 == 'Wire':
        class1_id = 1100
    elif class1 == 'Pole':
        class1_id = 1103
    elif class1 == 'Ground':
        class1_id = 1200
    elif class1 == 'Facade':
        class1_id = 1400

    if class2 == 'Veg':
        class2_id = 1004
    elif class2 == 'Wire':
        class2_id = 1100
    elif class2 == 'Pole':
        class2_id = 1103
    elif class2 == 'Ground':
        class2_id = 1200
    elif class2 == 'Facade':
        class2_id = 1400        
    
    index0 = y==class1_id
    index1 = y==class2_id
    
    
    xc_bin = np.concatenate((xc[index0], xc[index1]), axis = 0)
    xc_bin = np.reshape(xc_bin, (xc_bin.shape[0], 1))
#    print (xc_bin.shape)
    yc_bin = np.concatenate((yc[index0], yc[index1]), axis = 0)
    yc_bin = np.reshape(yc_bin, (yc_bin.shape[0], 1))
#    print (yc_bin.shape)
    zc_bin = np.concatenate((zc[index0], zc[index1]), axis = 0)
    zc_bin = np.reshape(zc_bin, (zc_bin.shape[0], 1))
#    print (zc_bin.shape)
    node_id_bin = np.concatenate((node_id[index0], node_id[index1]), axis = 0)
    node_id_bin = np.reshape(node_id_bin, (node_id_bin.shape[0], 1))
#    print (node_id_bin.shape)
#    y_bin = np.concatenate((y[index0], y[index1]), axis = 0)
#    y_bin = np.reshape(y_bin, (y_bin.shape[0], 1))
    y_bin = np.concatenate((-np.ones(((y[index0]).shape[0], 1 )), np.ones(((y[index1]).shape[0], 1))), axis = 0)
    y_bin = np.reshape(y_bin, (y_bin.shape[0], 1))
#    print (y_bin.shape)
    X_bin = np.concatenate((X[index0,:], X[index1,:]), axis = 0)
#    print (X_bin.shape)
    
    
    # Suffle all the data so that they are random
    
    data_bin = np.zeros((xc_bin.shape[0], parsed_data.shape[1]))
    for i in np.arange(node_id_bin.shape[0]):
        data_bin[i, 0] = xc_bin[i]
        data_bin[i, 1] = yc_bin[i]
        data_bin[i, 2] = zc_bin[i]
        data_bin[i, 3] = node_id_bin[i]
        data_bin[i, 4] = y_bin[i]
        for j in range(10):
            data_bin[i, 5+j] = X_bin[i, j]
            
#    print (data_bin.shape)
    
    data_bin = np.random.permutation(data_bin)
        
    
#    data_bin = np.hstack((xc_bin, yc_bin, zc_bin, node_id_bin, y_bin))
    
    
#    data_bin = np.random.permutation(np.concatenate((xc_bin, yc_bin, zc_bin, node_id_bin, y_bin, X_bin)))
    class1_id = -1
    class2_id = 1
    X_bin = data_bin[:, 5:]
    y_bin = (data_bin[:,4])
    return X_bin, y_bin, data_bin, class1_id, class2_id

def lsvm_predict(omega, x, y):
    if y*np.dot(omega, x) > 1:
        return True
    else:
        return False

def iteration_over_data(number_train_data, X, y, omega, class1, class2, data_vis): 
    Lambda = 0.001
    dt = 0
    losses = []
    count = 1
    c1_success = 0
    c1_fail = 0
    c2_success = 0
    c2_fail = 0    
    while dt < number_train_data:
        # Predict
        alpha = 1/(sqrt(1+dt))
        if lsvm_predict(omega, X[dt,:], y[dt]):
            omega = omega - alpha*Lambda*omega
            success = True
            if y[dt] == class1:
                c1_success += 1
            else:
                c2_success += 1
        else:
            omega = omega - alpha*Lambda*omega + alpha*y[dt]*X[dt,:]
            success = False
            if y[dt] == class1:
                c1_fail += 1
            else:
                c2_fail += 1
        loss_t = 1/2*Lambda*np.dot(omega, omega.T) + max(0,1-y[dt]*np.dot(omega, X[dt,:]))
        if loss_t != 0:
            if count == 1:
                min_loss = loss_t
                count += 1
            else:
                if loss_t < min_loss:
                    min_loss = loss_t
        if dt%50 == 0:
            Visualize_single(data_vis[dt], dt, 'lsvm')
        #print ('Label: ', y_tuning[dt], 'success: ', success, 'g: ', class1_success, 'v: ', class2_success)
        print('class1: ', c1_success, 'out of', c1_success+c1_fail, class2, c2_success, 'out of', c2_success+c2_fail)
        losses.append(loss_t)
        dt += 1
#        c1_total = c1_success + c1_fail
#        c2_total = c2_success + c2_fail
    index0 = y==class1
    index1 = y==class2
    c1_total = y[index0].shape[0]
    c2_total = y[index1].shape[0]
    return omega, c1_success, c1_total, c2_success, c2_total

def lsvm_classification(file_name, class1, class2):
    X, y, data, c1, c2 = get_binary_class_data(file_name, class1, class2)
    # We will use 75% of the data for training and 25% for testing later
    total_number_data = X.shape[0]
    number_train_data = int(floor(total_number_data*0.75))
    X_train = X[0:number_train_data,:]
    y_train = y[0:number_train_data]
    X_test = X[number_train_data:,:]
    y_test = y[number_train_data:]
    # initialize omega
    feature_size = X_train.shape[1]
    initial_omega = np.zeros((1, feature_size))
    trained_omega, class1_success_train, class1_total_train, class2_success_train, class2_total_train =  \
        iteration_over_data(number_train_data, X_train, y_train, initial_omega, c1, c2, data)
    
    ###################################################################################################
    ###################################################################################################
    # Let's now use the weights obtained from trainning and use them to make predictions on test data
    number_test_data = X_test.shape[0]
    post_test_omega, class1_success_test, class1_total_test, class2_success_test, class2_total_test = \
        iteration_over_data(number_test_data, X_test, y_test, trained_omega, c1, c2, data)
    ###################################################################################################
    ###################################################################################################
    # Let's add random features
    X_train_rand = np.concatenate((X_train, np.random.rand(number_train_data,5)), axis = 1)
    X_test_rand = np.concatenate((X_test, np.random.rand(number_test_data,5)), axis = 1)
    feature_size_rand = X_train_rand.shape[1] 
    initial_omega_rand = np.zeros((1, feature_size_rand))
    number_train_data_rand = X_train_rand.shape[0]
    trained_omega_rand, class1_success_train_rand, class1_total_train_rand, class2_success_train_rand, class2_total_train_rand =  \
        iteration_over_data(number_train_data_rand, X_train_rand, y_train, initial_omega_rand, c1, c2, data)
        
    number_test_data_rand = X_test_rand.shape[0]
    post_test_omega_rand, class1_success_test_rand, class1_total_test_rand, class2_success_test_rand, class2_total_test_rand = \
        iteration_over_data(number_test_data_rand, X_test_rand, y_test, trained_omega_rand, c1, c2, data)
    
    class1_success_ratio_train = class1_success_train/float(class1_total_train)
    class1_success_ratio_test = class1_success_test/float(class1_total_test)
    class2_success_ratio_train = class2_success_train/float(class2_total_train)
    class2_success_ratio_test = class2_success_test/float(class2_total_test)
    
    class1_success_ratio_train_rand = class1_success_train_rand/float(class1_total_train_rand)
    class2_success_ratio_train_rand = class2_success_train_rand/float(class2_total_train_rand)
    class1_success_ratio_test_rand = class1_success_test_rand/float(class1_total_test_rand)
    class2_success_ratio_test_rand = class2_success_test_rand/float(class2_total_test_rand)
    
    return class1_success_ratio_train, class1_success_ratio_test, class2_success_ratio_train, class2_success_ratio_test, \
        class1_success_ratio_train_rand, class2_success_ratio_train_rand, class1_success_ratio_test_rand, class2_success_ratio_test_rand

def moment_to_natural(mu, Sigma):
    P = np.linalg.inv(Sigma)
    J = np.dot(P, mu)
    return J, P
    
def natural_to_moment(J, P):
    Sigma = np.linalg.inv(P)
    mu = np.dot(Sigma, J)
    return mu, Sigma

def update(J, P, sigma, y, x):
    J = J + 1/pow(sigma, 2)*y*x
    P = P + 1/pow(sigma, 2)*np.dot(x, x.T)
    return J, P

def predict(mu, Sigma, x):
    mu_y = np.dot(mu.T, x)
    Sigma_y = np.dot(np.dot(x.T, Sigma), x)
    return mu_y, Sigma_y

def iterate_over_data(x,y,number_iter, mu, Sigma, sigma_noise, class1, class2, data_vis):
    dt = 0
    class1_success = 0
    class1_fail = 0
    class2_success = 0
    class2_fail = 0 
    while dt < number_iter:
        mu_y, Sigma_y = predict(mu, Sigma, (x[dt,:]).T)
        #print(mu.shape, Sigma.shape, ((x[dt,:]).T).shape)
        if (mu_y >= 0):
            if (y[dt] == class2):
                class2_success += 1
            else:
                class2_fail += 1
        else:
            if (y[dt] == class1):
                class1_success += 1
            else:
                class1_fail += 1
        
        J, P = moment_to_natural(mu, Sigma)
        J, P = update(J, P, sigma_noise, y[dt], ((x[dt,:]).T).reshape(-1,1))
        mu, Sigma = natural_to_moment(J, P)
        
        if dt%50 == 0:
            Visualize_single(data_vis[dt], dt, 'blr')
        
#        print(class1, class1_success, 'out of', class1_success+class1_fail, class2, class2_success, 'out of', class2_success+class2_fail)
#        print('class1: ', class1_success, 'out of', class1_success+class1_fail, 'class2: ', class2_success, 'out of', class2_success+class2_fail)
        dt += 1
        #class1_total = class1_success + class1_fail
        #class2_total = class2_success + class2_fail
        index0 = y==class1
        index1 = y==class2
        class1_total = y[index0].shape[0]
        class2_total = y[index1].shape[0]
    return mu, Sigma, class1_success, class1_total, class2_success, class2_total

def blr_classification(file_name, class1, class2):
    X,y, data, c1, c2 = get_binary_class_data(file_name, class1, class2)
    # We will use 75% of the data for training and 25% for testing later
    total_number_data = X.shape[0]
    number_train_data = int(floor(total_number_data*0.75))
    X_train = X[0:number_train_data,:]
    y_train = y[0:number_train_data]
    X_test = X[number_train_data:,:]
    y_test = y[number_train_data:]
    # Initial prior
    feature_size = X_train.shape[1]
    mu_theta = np.zeros((feature_size,1))
    cov_theta = 0.5*np.identity(feature_size)
    sigma_noise = 1
    trained_mu, trained_cov, class1_success_train, class1_total_train, class2_success_train, class2_total_train = \
        iterate_over_data(X_train, y_train, number_train_data, mu_theta, cov_theta, sigma_noise, c1, c2, data)
    ###################################################################################################
    ###################################################################################################
    # Let's now use the weights obtained from trainning and use them to make predictions on test data
    number_test_data = X_test.shape[0]
    post_test_mu, post_test_Sigma, class1_success_test, class1_total_test, class2_success_test, class2_total_test = \
        iterate_over_data(X_test, y_test, number_test_data, trained_mu, trained_cov, sigma_noise, c1, c2, data)
        
    X_train_rand = np.concatenate((X_train, np.random.rand(number_train_data,5)), axis = 1)
    X_test_rand = np.concatenate((X_test, np.random.rand(number_test_data,5)), axis = 1)
    feature_size_rand = X_train_rand.shape[1] 
    mu_theta_rand = np.zeros((feature_size_rand,1))
    cov_theta_rand = 0.5*np.identity(feature_size_rand)
    number_train_data_rand = X_train_rand.shape[0]
    trained_mu_rand, trained_cov_rand, class1_success_train_rand, class1_total_train_rand, class2_success_train_rand, class2_total_train_rand = \
        iterate_over_data(X_train_rand, y_train, number_train_data_rand, mu_theta_rand, cov_theta_rand, sigma_noise, c1, c2, data)
    
    number_test_data_rand = X_test_rand.shape[0]
    post_test_mu_rand, post_test_Sigma_rand, class1_success_test_rand, class1_total_test_rand, class2_success_test_rand, class2_total_test_rand = \
        iterate_over_data(X_test_rand, y_test, number_test_data_rand, trained_mu_rand, trained_cov_rand, sigma_noise, c1, c2, data)
    
    class1_success_ratio_train = class1_success_train/float(class1_total_train)
    class1_success_ratio_test = class1_success_test/float(class1_total_test)
    class2_success_ratio_train = class2_success_train/float(class2_total_train)
    class2_success_ratio_test = class2_success_test/float(class2_total_test)
    
    class1_success_ratio_train_rand = class1_success_train_rand/float(class1_total_train_rand)
    class1_success_ratio_test_rand = class1_success_test_rand/float(class1_total_test_rand)
    class2_success_ratio_train_rand = class2_success_train_rand/float(class2_total_train_rand)
    class2_success_ratio_test_rand = class2_success_test_rand/float(class2_total_test_rand)
    
    return class1_success_ratio_train, class1_success_ratio_test, class2_success_ratio_train, class2_success_ratio_test,\
             class1_success_ratio_train_rand, class2_success_ratio_train_rand, class1_success_ratio_test_rand, class2_success_ratio_test_rand
    
def gd_loss (omega, x, y):
    return pow((np.dot(x.T,omega) - y), 2)

def gd_predict(omega, x):
#    x = x.reshape(-1,1)
    if(np.dot(omega.T,x)) >= 0:
        return 1
    else:
        return -1
    
def gd_iterate_over_data(x, y, number_iter, omega, class1, class2, data_vis):
    dt = 0
    c1_success = 0
    c1_fail = 0
    c2_success = 0
    c2_fail = 0
    alpha = 0.01
    while dt < number_iter:
        if gd_predict(omega, x[dt,:]) == class1:
            if y[dt] == class1:
                c1_success += 1
            else:
                c1_fail += 1
        else:
            if y[dt] == class2:
                c2_success += 1
            else:
                c2_fail += 1
        omega = omega - alpha*(np.dot(omega.T, ((x[dt,:]).reshape(-1,1))) - y[dt])*((x[dt,:]).reshape(-1,1))

#        if dt%50 == 0:
#            Visualize_single(data_vis[dt], dt, 'blr')
#        print('class1: ', c1_success, 'out of', c1_success+c1_fail, 'class2: ', c2_success, 'out of', c2_success+c2_fail)
        dt += 1
#    c1_total = c1_success + c1_fail
#    c2_total = c2_success + c2_fail
    index0 = y==class1
    index1 = y==class2
    c1_total = y[index0].shape[0]
    c2_total = y[index1].shape[0]
    return omega, c1_success, c1_total, c2_success, c2_total

def gd_classification(file_name, class1, class2):
    
    X,y, data, c1, c2 = get_binary_class_data(file_name, class1, class2)
    # We will use 75% of the data for training and 25% for testing later
    total_number_data = X.shape[0]
    number_train_data = int(floor(total_number_data*0.75))
    X_train = X[0:number_train_data,:]
    y_train = y[0:number_train_data]
    X_test = X[number_train_data:,:]
    y_test = y[number_train_data:]
    number_test_data = X_test.shape[0]
    
    omega = np.zeros((10,1))
    omega_train, c1_success_train, c1_total_train, c2_success_train, c2_total_train = gd_iterate_over_data(X_train, y_train, number_train_data, omega, c1, c2, data)
    omega, c1_success_test, c1_total_test, c2_success_test, c2_total_test = gd_iterate_over_data(X_test, y_test, number_test_data, omega_train, c1, c2, data)
    
    X_train_rand = np.concatenate((X_train, np.random.rand(number_train_data,5)), axis = 1)
    X_test_rand = np.concatenate((X_test, np.random.rand(number_test_data,5)), axis = 1)
    feature_size_rand = X_train_rand.shape[1] 
    omega_rand = np.zeros((feature_size_rand,1))
    number_train_data_rand = X_train_rand.shape[0]
    omega_train_rand, c1_success_train_rand, c1_total_train_rand, c2_success_train_rand, c2_total_train_rand = gd_iterate_over_data(X_train_rand, y_train, number_train_data_rand, omega_rand, c1, c2, data)
    
    number_test_data_rand = X_test_rand.shape[0]
    omega_test_rand, c1_success_test_rand, c1_total_test_rand, c2_success_test_rand, c2_total_test_rand = gd_iterate_over_data(X_test_rand, y_test, number_test_data_rand, omega_train_rand, c1, c2, data)
    
    
    class1_success_ratio_train = c1_success_train/float(c1_total_train)
    class1_success_ratio_test = c1_success_test/float(c1_total_test)
    class2_success_ratio_train = c2_success_train/float(c2_total_train)
    class2_success_ratio_test = c2_success_test/float(c2_total_test)
    
    class1_success_ratio_train_rand = c1_success_train_rand/float(c1_total_train_rand)
    class2_success_ratio_train_rand = c2_success_train_rand/float(c2_total_train_rand)
    
    class1_success_ratio_test_rand = c1_success_test_rand/float(c1_total_test_rand)
    class2_success_ratio_test_rand = c2_success_test_rand/float(c2_total_test_rand)
    
    
    return class1_success_ratio_train, class1_success_ratio_test, class2_success_ratio_train, class2_success_ratio_test,\
        class1_success_ratio_train_rand, class2_success_ratio_train_rand, class1_success_ratio_test_rand, class2_success_ratio_test_rand



#file_name = "oakland_part3_am_rf.node_features"

file_name = "oakland_part3_am_rf.node_features"

basePath = os.path.dirname(__file__)


#parsed_data = read_data_from_file(file_name)
#Visualize(parsed_data)

#class1_svm = 'Ground'
#class2_svm = 'Veg'
#
#print ('LSVM')
#class1_train, class1_test, class2_train, class2_test, class1_train_rand, class2_train_rand, class1_test_rand, class2_test_rand = lsvm_classification(file_name, class1_svm, class2_svm)
#print ('Class 1 success ratio train: ', class1_train, 'Class1 success ratio test: ',  class1_test)
#print('class2 success ratio train: ', class2_train, 'class2 success ratio test: ', class2_test)
#print('Class 1 success ratio train with random feat: ', class1_train_rand, 'Class 1 success ratio test with random feat: ', class1_test_rand)
#print('class 2 success ratio train with random feat: ', class2_train_rand, 'Class 2 success ratio test with random feat: ', class2_test_rand)

class1_blr = 'Veg'
class2_blr = 'Facade'

print ('BLR')
class1_ratio_train, class1_ratio_test, class2_ratio_train, class2_ratio_test, class1_ratio_train_rand, class2_ratio_train_rand, \
      class1_ratio_test_rand, class2_ratio_test_rand  = blr_classification(file_name, class1_blr, class2_blr)
print ('class1 success ratio train: ', class1_ratio_train, 'class1 success ratio test: ',  class2_ratio_test)
print('class2 success ratio train: ', class2_ratio_train, 'class2 success ratio test: ', class2_ratio_test)
print('class1 success ratio train with random feat: ', class1_ratio_train_rand, 'class1 success ratio test with randomness: ', class1_ratio_test_rand)
print('class2 success ratio train with random feat: ', class2_ratio_train_rand, 'class2 success ration test with random feat: ', class2_ratio_test_rand )


#file_name = "oakland_part3_am_rf.node_features"
#basePath = os.path.dirname(__file__)
#
#class1_gd = 'Veg'
#class2_gd = 'Facade'
#
#class1_ratio_train, class1_ratio_test, class2_ratio_train, class2_ratio_test, class1_ratio_train_rand, \
#    class2_ratio_train_rand, class1_ratio_test_rand, class2_ratio_test_rand = gd_classification(file_name, class1_gd, class2_gd)
#
#print ('class1 success ratio train: ', class1_ratio_train, 'class1 success ratio test: ',  class2_ratio_test)
#print('class2 success ratio train: ', class2_ratio_train, 'class2 success ratio test: ', class2_ratio_test)
#print('class1 success ratio train with random feat: ', class1_ratio_train_rand, 'class1 success ratio test with random feat: ', class1_ratio_test_rand )
#print('class2 success ratio train with random feat: ', class2_ratio_train_rand, 'class2 success ratio test with random feat: ', class2_ratio_test_rand )
#
#
#
