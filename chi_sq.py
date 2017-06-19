#!/usr/bin/env python
# classification in presence of imbalance and overlap
# matching the variable names as in the corresponding R code 

import os, sys, pickle
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import gamma
from scipy.special import beta

def main():
    if sys.argv[1]<3:     
        print "Usage: python chi_sq.py trainfile testfile"
        exit(0)
    trainfile = sys.argv[1]
    testfile = sys.argv[2]
    (dirName, trainfileName) = os.path.split(trainfile)
    (trainfileBaseName, trainfileExtension)=os.path.splitext(trainfileName)

    # Read the file in csv format
    input_data = np.loadtxt(trainfile, delimiter=',', dtype=float)
#    pca = PCA(n_components = 6)
#    pca.fit(X)
#    pca_X = pca.transform(X)
#    pca_data = np.concatenate((data[:, 0:2], pca_X), axis=1)
    data = input_data[:, 1:]
    label = data[:, 1]
    train_alive = data[data[:, 0]==0]
    print train_alive
    train_dead = data[data[:, 0]==1]
    print train_dead

    train_alive_features = train_alive[:, 1:]
    train_dead_features = train_dead[:, 1:]
    print train_alive_features
   
    test = np.loadtxt(testfile, delimiter=',', dtype=float)

    mean_alive = np.mean(train_alive_features, axis=0)
    mean_dead = np.mean(train_dead_features, axis=0)
    print mean_alive
    print mean_dead 

    var_alive = np.cov(train_alive_features.T)
    var_dead = np.cov(train_dead_features.T)
    print var_alive

    var_alive_inv = np.linalg.pinv(np.matrix(var_alive))
    var_dead_inv = np.linalg.pinv(np.matrix(var_dead))

    print var_alive_inv
    print var_dead_inv
    print train_alive_features.shape
    print np.zeros((2, 3))
    mat_0 = (np.zeros((train_alive_features.shape[0], train_alive_features.shape[0])))
    mat_1 = (np.zeros((train_dead_features.shape[0], train_dead_features.shape[0])))
    print mat_0.shape
    print mat_1.shape


    for i in range(mat_0.shape[0]):
        for j in range(mat_0.shape[0]):
            mat_0[i][j] = (np.matrix(train_alive_features[i, :]-mean_alive)*var_alive_inv*np.matrix(train_alive_features[j, :]-mean_alive).T)**3
    
    print mat_0
    for i in range(mat_1.shape[0]):
        for j in range(mat_1.shape[0]):
            mat_1[i][j] = (np.matrix(train_dead_features[i, :]-mean_dead)*var_dead_inv*np.matrix(train_dead_features[j, :]-mean_dead).T)**3
    print mat_1

    u0 = sum(sum(mat_0))/(6*mat_0.shape[0])
    u1 = sum(sum(mat_1))/(6*mat_1.shape[0])

    print u0
    print u1
# ********** verified
    
    test_features = test[:, 2:]
    print test_features

    pred = np.zeros((test_features.shape[0], 1))
    print pred

    for i in range(test_features.shape[0]):
        z = test_features[i, :]
        z0 = 1/6.0*(np.matrix(z-mean_alive)*var_alive_inv*np.matrix(z-mean_alive).T)**3
        z1 = 1/6.0*(np.matrix(z-mean_dead)*var_dead_inv*np.matrix(z-mean_dead).T)**3
        print 'inside loop' 
        z0 = z0.item(0, 0)
        z1 = z1.item(0, 0)
        print('z0 - u0') 
        print(abs(z0-u0)) 
        # NOTE: check the beta and gamma functions!!!
        if(abs(z0-u0)>1):
            print('1 if')
            if(z0-u0>0):
                print('1 1 if')
                p0 = 2**(-28)*55*gamma.cdf((z0-u0)/2.0, 27)
                print 'p0', p0
            else:
                print('1 1 else')
                p0 = 1-2**(-28)*55*gamma.cdf((z0-u0)/2.0, 27)
                print 'p0', p0
        
        if(abs(z0-u0)<1):
            print('2 if')
            lambda1 = 2**(-56)/beta(28, 28) * 4/110.0
            print 'lambda1', lambda1
            a = lambda1*2
            b = 3*lambda1
            p0 = 0.5+a+b+np.exp(-(u0-z0))
            print 'p0', p0
        
        print('z1 - u1') 
        print(abs(z1-u1))
        if(abs(z1-u1)>1):
            print('3 if')
            if(z1-u1>0):
                print('3 1 if')
                p1 = 2**(-28) *55*gamma.cdf((z1-u1)/2, 27)
                print 'p1', p1
            else:
                print('3 1 else')
                p1 = 1-2**(-28)*55*gamma.cdf((z1-u1)/2, 27)
                print 'p1', p1

        if(abs(z1-u1)<1):
            lambda1 = 2**(-56)/beta(28, 28)*4/100
            print 'lambda1', lambda1
            a = lambda1*2
            b = 3*lambda1
            p1 = 0.5+a+b+np.exp(-(u0-z0))
            print 'p1', p1

        if(p0>0.975):
            pred[i] = 0
        elif(p1>0.975):
            pred[i] = 1
        else:
            l0 = 1/train_alive_features.shape[0] * (sum(sum(train_alive_features)) - sum(z))
            l1 = 1/train_dead_features.shape[0] *(sum(sum(train_dead_features))-sum(z))
            print 'l0, l1', l0, l1
            if(l0>l1):
                pred[i]=1
            else:
                pred[i]=0
        print p0, pred[i]   
# save the output
    output = np.concatenate((pred, test_features[:, 0:2]), axis=1)
    np.savetxt('output_result_'+testfile, output, delimiter=',')

if __name__=="__main__":
    main()
