#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Editor : Jeong Kim
Origin : Seong Cho

Gershgorin Region

Goal:
    Set a matrix(M)
    Compute the Isospectral Reduction
    Compute the spectrum of M
    Solve the regions for the matrix using Gershgorin's Theorem
    Plot the region
    ! Reduce the matrix along the row 1 and then 1 & 3
    
"""

import numpy as np
import scipy.linalg as la # scipy.linalg for linear algebra functions
import sympy as sy # for symbolic mathematics

M = np.array([[1,0,1,-1],[2,1,-1,1],[1,0,0,2],[1,1,1,1]])
eigs = la.eigvals(M) # Computer eigenvalues

# We need this process because we have lamda x
x = sy.symbols('x') # Transform strings into symbols
S = sy.Matrix(M) # Contruct a matrix
M_in = sy.Matrix([M[3,3] - x])
M_in_inv = M_in.inv()
R = S[1:,1:] - S[1:,0]@M_in_inv@S[0,1:] # @ means multiplication

# Spectrum of M : the solutions of det(M-lamda*I) = 0 
char_poly = (R - x*sy.eye(3)).det() # eye makes identity matrix
sy.solve(char_poly, x) # solve it in terms of 'x'

"""

# Mij is an element
# let d = lamda
Region:
    R1 = |d-1| <= |0|+|1|+|-1|
    R2 = |d-1| <= |2|+|-1|+|1|
    R3 = |d-0| <= |1|+|0|+|2|
    R4 = |d-1| <= |1|+|1|+|1|
    Region = (R1) U (R2) U (R3) U (R4)
    
"""

import matplotlib.pyplot as plt
from pylab import *

"""

# j : imaginary number, 1j : one imaginary number
# r : real number

domain:
    One dimensional array list containing 40000 values
    200 X 200 = 40000
    x-axis : real number
    y-axis : imaginary number

"""

# Generate evenly spaced 200 numbers from -5 to 5
domain = np.linspace(-5,5,200) 
# faltten : return one dimensional array
domain = np.array([r+1j*np.linspace(-5,5,200) for r in domain]).flatten()

r1 = []
r2 = []
r3 = []
r4 = []
for c in domain: # manually putting in
    if np.abs(c-1) <= 2:
        r1.append(c)
    if np.abs(c-1) <= 4:
        r2.append(c)
    if np.abs(c) <= 3:
        r3.append(c)
    if np.abs(c-1) <= 3:
        r4.append(c)

# Plot
plt.figure(figsize=(10,10))
plt.scatter(np.real(r1), np.imag(r1), alpha=1, s=30, c='b', label='r1')
plt.scatter(np.real(r2), np.imag(r2), alpha=0.1, s=30, label='r2')
plt.scatter(np.real(r3), np.imag(r3), alpha=0.2, s=30, label='r3')
plt.scatter(np.real(r4), np.imag(r4), alpha=0.1, s=30, label='r4')
plt.scatter(np.real(eigs), np.imag(eigs), s=50, c='r', label='eigenvalues')

plt.legend()
plt.ylim((-5,5))
plt.xlim((-5,5))
plt.show()