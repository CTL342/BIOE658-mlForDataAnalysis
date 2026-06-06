# Name: Christopher A. Lee
# Date: 02/07/2025
# Prof: Dr. Azarnoosh
# Course: Machine Learning for Data Analysis - ENBC 321

import numpy as np
import pandas as pd
import sympy as sp
# Problem 2
print("Problem 2")
M = np.array([14, -7, -10, 2, 9, 5, 14, -22, 16]).reshape(3, 3)
v1 = np.array([2, 8, 6]).reshape(3, 1)
v2 = np.array([13, 21, 16]).reshape(3, 1)
print(M)
print(v1)
print(v2, "\n")

print("Part a")
v1Tv2 = np.dot(v1.T, v2)
print(v1Tv2, "\n")

print("Part b")
Mv1 = np.dot(M, v1)
print(Mv1, "\n")

print("Part c")
MTM = np.dot(M.T, M)
print(MTM, "\n")

print("Part d")
print("False because M transposed is not equal to M", "\n")

print("Part e")
v1L2 = np.linalg.norm(v1, ord=2)
v1L1 = np.linalg.norm(v1, ord=1)
v1LI = np.linalg.norm(v1, ord=np.inf)
print("v1 L2 Norm:", v1L2)
print("v1 L1 Norm:", v1L1)
print("v1 LInfinity Norm:", v1LI)

v2L2 = np.linalg.norm(v2, ord=2)
v2L1 = np.linalg.norm(v2, ord=1)
v2LI = np.linalg.norm(v2, ord=np.inf)
print("v2 L2 Norm:", v2L2)
print("v2 L1 Norm:", v2L1)
print("v2 LInfinity Norm:", v2LI, "\n")