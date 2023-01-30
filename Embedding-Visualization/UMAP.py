# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 12:03:53 2023

@author: ken3


intorduction:
    Uniform Manifold Approximation and Projection (UMAP) is a 
    dimension reduction technique that can be used for 
    visualisation similarly to t-SNE, but also for 
    general non-linear dimension reduction. 
    
    The goal of UMAP is to preserve the local and global structure of the data,
    meaning that similar data points should be close to each other in 
    the reduced space, and dissimilar data points should be far apart. 
    UMAP achieves this by approximating the data manifold, 
    which is a low-dimensional surface that the data lies on, 
    and projecting the data onto it. 
    This results in a more interpretable and visually appealing 
    representation of the data in lower dimensions

Require:
    UMAP packadge
        conda install -c conda-forge umap-learn
    
reference:
    1. https://github.com/lmcinnes/umap
    2. McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation and Projection
        for Dimension Reduction, ArXiv e-prints 1802.03426, 2018 
"""

import umap
import umap.plot
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


digits = load_digits()

mapper = umap.UMAP().fit(digits.data)
umap.plot.points(mapper, labels=digits.target)
plt.savefig("UMAP_mnist.png")

'''
Plotting
    UMAP includes a subpackage umap.plot for plotting the results of UMAP embeddings.
    This package needs to be imported separately since 
    it has **extra requirements (matplotlib, datashader and holoviews)**. 
    It allows for fast and simple plotting and attempts to make 
    sensible decisions to avoid overplotting and other pitfalls.
'''