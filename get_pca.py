#!/usr/bin/env python


import os, sys, pickle
import numpy as np
from sklearn.decomposition import PCA


def main():
    if sys.argv[1]<2:     
        print "Usage: python chi_sq.py filename"
        exit(0)
    filename = sys.argv[1]
    (dirName, fileName) = os.path.split(filename)
    (fileBaseName, fileExtension)=os.path.splitext(filename)

    # Read the file in csv format
    data = np.loadtxt(filename, delimiter=',', dtype=float)
    X = data[:, 2:]
    y = data[:, 1] 
    pca = PCA(n_components = 6)
    pca.fit(X)
    pca_X = pca.transform(X)
    pca_data = np.concatenate((data[:, 0:2], pca_X), axis=1)

    np.savetxt('pca_6_'+fileBaseName +'.csv', pca_data, delimiter=',')

if __name__=="__main__":
    main()
