# -*- coding: utf-8 -*-
"""
Created on Fri May 11 15:34:15 2018

@author: nurooool
"""

#import os,cv2
#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib import offsetbox
import h5py
import time
import pylab
from six.moves import cPickle
import tensorflow as tf
from sklearn.manifold import TSNE
#from MulticoreTSNE import MulticoreTSNE as TSNE
tf.__version__

#%%
#open the feature extraction file 520
# feature = feature extracted
# feature labels = filename

ff = h5py.File('features_inceptionresnetv2_avgpool.hdf5', 'r')
feature = ff['feature']

#try with 100k first for each
test = feature[0:5000]
test = (test - test.min()) / (test.max() - test.min())

#%%


print("Computing t-SNE embedding")
time_start = time.time()
tsne = TSNE(n_components=2, perplexity= 30, n_iter= 5000, verbose= 2, init='pca')

x_tsne = tsne.fit_transform(test)

print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)

xy_coor = open('test.pkl', 'wb')
cPickle.dump(x_tsne, xy_coor, protocol=cPickle.HIGHEST_PROTOCOL)
xy_coor.close()

pylab.scatter(x_tsne[:, 0], x_tsne[:, 1])
pylab.show()

#to open the coordinate
#test_xy = 'Mxy_coor_avg150k_50per.pkl'
#with open(test_xy,'rb') as fid:
#    xy = cPickle.load(fid)
#
#pylab.scatter(xy[:, 0], xy[:, 1])
#pylab.show()