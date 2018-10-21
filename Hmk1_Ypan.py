#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 17:12:59 2018

@author: yuranpan
"""

# get current working directory to make sure files are saved in the same directory
import os
#os.chdir('/Users/yuranpan/Desktop/Fordham/Data_Mining/Hmk1/Hmk1')

# read csv files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

train_100_10 = pd.read_csv('train-100-10.csv', encoding = 'utf-8')
test_100_10 = pd.read_csv('test-100-10.csv')

train_100_100 = pd.read_csv('train-100-100.csv')
test_100_100 = pd.read_csv('test-100-100.csv')

train_1000_100 = pd.read_csv('train-1000-100.csv')
test_1000_100 = pd.read_csv('test-1000-100.csv')

#create additional training files
train_50_1000_100 = train_1000_100.loc[0:49]
train_100_1000_100 = train_1000_100.loc[0:99]
train_150_1000_100 = train_1000_100.loc[0:149]

train_50_1000_100.to_csv('train-50(1000)-100.csv')
train_100_1000_100.to_csv('train-100(1000)-100.csv')
train_150_1000_100.to_csv('train-150(1000)-100.csv')


# Question 1


def matrix_X(dataset):
    temp_X = np.array(dataset[dataset.columns[0:-1]])
    temp_ones = np.ones((np.shape(temp_X)[0],1))
    X = np.hstack((temp_ones, temp_X))
    return X


def matrix_y(dataset):
    y = np.array(dataset[dataset.columns[-1]])
    return y


def LLR_coef(dataset, lamd):
    X = matrix_X(dataset)
    y = matrix_y(dataset)
    X_T = X.T
    XTX = X.T.dot(X)
    XTX_dim = XTX.shape
    I_dim = XTX_dim[0]
    I = np.eye(I_dim)
    w = np.linalg.inv(XTX + lamd * I).dot(X_T).dot(y)
    return w 

def MSE_train(train_dataset, lamd):
    temp_resi = sum((matrix_X(train_dataset).dot(LLR_coef(train_dataset, lamd))- matrix_y(train_dataset))**2)
    n = matrix_X(train_dataset).shape[0]
    MSE_train = (temp_resi)/n
    return MSE_train

def MSE_test(train_dataset, test_dataset, lamd):
    temp_resi = sum((matrix_X(test_dataset).dot(LLR_coef(train_dataset, lamd))- matrix_y(test_dataset))**2)
    n = matrix_X(test_dataset).shape[0]
    MSE_test = (temp_resi)/n
    return MSE_test


def all_MSE_train(train_dataset,lamd_start, lamd_end):  
    MSE_list = list()
    for l in range(lamd_start, lamd_end + 1):
        meanSE = MSE_train(train_dataset,l)
        MSE_list.append(meanSE) 
    return MSE_list

def all_MSE_test(train_dataset,test_dataset,lamd_start,lamd_end):  
    MSE_list = list()
    for l in range(lamd_start, lamd_end + 1):
        meanSE = MSE_test(train_dataset,test_dataset,l)
        MSE_list.append(meanSE) 
    return MSE_list


def plot_train_test(train_dataset, test_dataset,lamd_start,lamd_end):
    plt.plot(all_MSE_train(train_dataset, lamd_start, lamd_end), label = "Train")
    plt.xlabel('Lambda')
    plt.ylabel('MSE')
    plt.plot(all_MSE_test(train_dataset, test_dataset, lamd_start, lamd_end), label = "Test")
    plt.legend()
    plt.show()
    plt.close()

#plot_train_test(train_100_10,test_100_10,0,150)

#plot_train_test(train_1000_100,test_1000_100,0,150) 

#plot_train_test(train_100_100,test_100_100,0,150)

#plot_train_test(train_50_1000_100,test_1000_100,0,150)

#plot_train_test(train_100_1000_100,test_1000_100,0,150)

#plot_train_test(train_150_1000_100,test_1000_100,0,150)



# Which lambda gives the max MSE for each dataset
def lamb_minMSE(train_dataset, test_dataset, l_start, l_end):
    array_MSE = np.array(all_MSE_test(train_dataset, test_dataset, l_start, l_end))
    return array_MSE.argmin(), array_MSE.min()

print(lamb_minMSE(train_100_10,test_100_10,0,150))
print(lamb_minMSE(train_100_100,test_100_100,0,150))
print(lamb_minMSE(train_1000_100,test_1000_100,0,150))
print(lamb_minMSE(train_50_1000_100,test_1000_100,0,150))
print(lamb_minMSE(train_100_1000_100,test_1000_100,0,150))
print(lamb_minMSE(train_150_1000_100,test_1000_100,0,150))



# additional graph with lambda range from 1 - 150
#plot_train_test(train_100_100, test_100_100,1,150)

#plot_train_test(train_50_1000_100,test_1000_100,1,150)

#plot_train_test(train_100_1000_100,test_1000_100,1,150)


#plot_train_test(train_150_1000_100,test_1000_100,1,150)


##########################
#####CROSS VALIDATION#####
##########################

# split the data set into 10 folds
from sklearn.model_selection import KFold

def lambda_cross_validation(train_set,max_l,fold):
    kf = KFold(n_splits = fold, random_state = 1, shuffle = True)
    lamd_validation_best_list = list()
    MSE_avg = np.array([])
    for l in range(0,max_l + 1):
        MSE_val = list()
        for train_index, validation_index in kf.split(train_set):
            train_subset = train_set.iloc[train_index,:]
            validation_subset = train_set.iloc[validation_index,:]
            MSE_val.append(MSE_test(train_subset, validation_subset,l)) # 10 MSE values for 10 folds
        MSE_avg = np.append(MSE_avg, np.mean(MSE_val))  # avg MSE for each 150 lambda
    lamd_validation_best = MSE_avg.argmin() # the best lambda that gives minial avg MSE 
    return lamd_validation_best

def MSE_CV(train_set, test_set, max_l, fold):
    best_lambda = lambda_cross_validation(train_set,max_l,fold)
    MSE_test_with_best_lambda = MSE_test(train_set,test_set,best_lambda)
    return best_lambda, MSE_test_with_best_lambda



print('100_10 CV:', MSE_CV(train_100_10, test_100_10, 150, 10))
print('100_100 CV:', MSE_CV(train_100_100, test_100_100, 150, 10))
print('1000_100 CV:', MSE_CV(train_1000_100, test_1000_100, 150, 10))
print('50_1000_100 CV:', MSE_CV(train_50_1000_100, test_1000_100, 150, 10))
print('100_1000_100 CV:', MSE_CV(train_100_1000_100, test_1000_100, 150, 10))
print('150_1000_100 CV:', MSE_CV(train_150_1000_100, test_1000_100, 150, 10))






##########################
#####LEARNING CURVE#######
##########################
    
def learning_curve(train_dataset, test_dataset,max_sample_size, lamd):
    sample_size_list = list(range(10,max_sample_size + 1,10))
    Error_in_list = list()
    Error_out_list = list()
    for sample_size in sample_size_list:
        MSE_sample_train_list = list()
        MSE_sample_test_list = list()
        sample_draw = 1 
        while sample_draw <=100:    
            sample_train = train_dataset.sample(sample_size)  
            sample_test = test_dataset.sample(sample_size)
            MSE_sample_train = MSE_train(sample_train,lamd)
            MSE_sample_test = MSE_test(sample_train, sample_test,lamd)
            MSE_sample_train_list.append(MSE_sample_train)
            MSE_sample_test_list.append(MSE_sample_test)
            sample_draw = sample_draw + 1
        avg_MSE_sample_train = np.mean(MSE_sample_train_list)
        avg_MSE_sample_test = np.mean(MSE_sample_test_list)
        Error_in_list.append(avg_MSE_sample_train)
        Error_out_list.append(avg_MSE_sample_test)
    plt.plot(sample_size_list, Error_in_list, label = "Error_in")
    plt.plot(sample_size_list, Error_out_list, label = "Error_out")
    plt.title('Learning Curve')
    plt.xlabel('Sample Size')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    plt.close()


#learning_curve(train_1000_100, test_1000_100,1000, 1)
#learning_curve(train_1000_100, test_1000_100,1000, 25)
#learning_curve(train_1000_100, test_1000_100,1000, 150)



