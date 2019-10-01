#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:50:40 2019

@author: samaneh
"""

# Clustering handwritten digits with k-means
import numpy as np
import matplotlib.pyplot as plt
# import data
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
digits = load_digits()
data =  scale(digits.data)
print(digits.keys())
# print data 
def print_digits(images, y, max_n=10):
    fig = plt.figure(figsize=(12, 12)) # set up the figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1,
             hspace=0.05, wspace=0.05)
    i = 0
    while i <max_n and i <images.shape[0]:
        p = fig.add_subplot(20, 20, i + 1, xticks=[],
                            yticks=[]) # plot the images in a matrix of 20x20
        p.imshow(images[i], cmap=plt.cm.bone)
        p.text(0, 14, str(y[i])) # label the image with the target value
        i = i+ 1
print_digits(digits.images, digits.target, max_n=10)
# seperate train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images,  test_size=0.25,random_state=42)
n_samples, n_features = x_train.shape # digits 8*8 pixles= 64 features
n_digits = len(np.unique(y_train))
labels = y_train
# using K-means
from sklearn import cluster
clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)
clf.fit(x_train)

print_digits(images_train, clf.labels_, max_n=10)

# predict
y_pred = clf.predict(x_test)

def print_cluster(images, y_pred, cluster_number):
    images = images[y_pred==cluster_number]
    y_pred = y_pred[y_pred==cluster_number]
    print_digits(images, y_pred,max_n=10)

for i in range(10):
    print_cluster(images_test, y_pred, i)
# accuracy index: rand index    
from sklearn import metrics
print("Adjusted rand score: {:.2}".format(metrics.adjusted_rand_score(y_test, y_pred)))
print(metrics.confusion_matrix(y_test, y_pred))

# plotting process: dimensionlaity reduction with PCA, construct a meshgrid of points, calculate their assigned cluster, and plot them
from sklearn import decomposition
pca = decomposition.PCA(n_components=2).fit(x_train)
reduced_x_train = pca.transform(x_train)
h = .01 # Step size of the mesh
x_min, x_max = reduced_x_train[:, 0].min() + 1, reduced_x_train[:, 0].max() - 1 # point in the mesh [x_min, m_max]x[y_min, y_max]
y_min, y_max = reduced_x_train[:, 1].min() + 1, reduced_x_train[:, 1].max() - 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
kmeans = cluster.KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_x_train)
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape) # Put the result into a color plot
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(),yy.max()), cmap=plt.cm.Paired, aspect='auto', origin='lower')
plt.plot(reduced_x_train[:, 0], reduced_x_train[:, 1], 'k.',
       markersize=2)
centroids = kmeans.cluster_centers_ # Plot the centroids as a white X
plt.scatter(centroids[:, 0], centroids[:,1], marker='.', s=169, linewidths=3, color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA reduced data)\n Centroids are marked with white dots')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

# kernel K-means




























