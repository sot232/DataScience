#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:27:28 2018

@author: jeong-ugim


# K-Armed Bandit Problem

Exercise 2.5

Desmonstrate and plot the difficulties 
that sample-average methods for nonstationary problems.

q*(a) : start out equally
mean : zero
standard deviation : 0.01
alpha : 0.1
epsilon : 0.1
N : 10,000 steps
"""

import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 0, 0.1
s = np.random.normal(mu, sigma, 1000)

count, bins, ignored = plt.hist(s, 30, normed=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * \
        np.exp( - (bins - mu)**2 / (2 * sigma**2) ),\
        linewidth=2, color='r')
plt.show()

value = 