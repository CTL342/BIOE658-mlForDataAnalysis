# Name: Christopher A. Lee
# Date: 02/07/2025
# Prof: Dr. Azarnoosh
# Course: Machine Learning for Data Analysis - ENBC 321

import numpy as np
import pandas as pd
import sympy as sp
# Problem 1
print("Problem 1")
A = np.array([2, 1, -1, 3, 4, 2, 1, -2, 5]).reshape(3, 3)
b = np.array([3, 7, 4]).reshape(3, 1)
print(A)
print(b, "\n")

print("Part a")
sol = np.linalg.solve(A, b)
print(sol, "\n")

print("Part b")
det = np.linalg.det(A)
print(det, "\n")

print("Part c")
tra = A.T
print(tra, "\n")

print("Part d")
ran = np.trace(A)
print(ran, "\n")

print("Part e")
ran = np.linalg.matrix_rank(A)
print(ran, "\n")

print("Part f")
lin_indep_col = np.linalg.matrix_rank(A)
print(lin_indep_col, "=", ran, "? True" "\n")

print("Part g")
orth = np.dot(tra, A)
print(orth)
print("False, because A^T * A is not equal to identity matrix", "\n")

print("Part h")
inv = np.linalg.inv(A)
print("A Inverse:\n", inv)
print("A Transpose\n", A.T)
print("False, A Inverse is not equal to A Transpose", "\n")

print("Part i")
eig_val, eig_vec = np.linalg.eig(A)
print("Eiganvalues:", eig_val)
print("Eiganvectors:", eig_vec, "\n")

print("Part j")
x, y, z = sp.symbols('x y z')
func = x**2 + 3*x*y + y**2 + 2*x*z + z**2
v = list(sp.ordered(func.free_symbols))
gradient = sp.Matrix([func]).jacobian(v)
hessian = sp.hessian(func, v)
print("Graident:\n", gradient)
print("Hessian:\n", hessian, "\n")