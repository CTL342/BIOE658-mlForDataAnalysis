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

# Problem 3
# Kaggle Dataset Used: Cafe Sales - Dirty Data for Cleaning Training, Dirty Cafe Sales Dataset
# Author: Ahmed Mohamed
# Link: https://www.kaggle.com/datasets/ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training
print("Problem 3")
df = pd.read_csv('dirty_cafe_sales.csv')
print(df.head())
print("Chosen Column: Total Spent")

print("\n", "Part a")
total_spent_remove = pd.to_numeric(df['Total Spent'], errors='coerce').dropna()

df_remove = pd.DataFrame({
    'Before': [0.0, 0.0, 0.0, 0],
    'After': [0.0, 0.0, 0.0, 0]
}, index=['Mean', 'Variance', 'Standard Deviation', '# Rows'])

df_remove.loc['Mean', 'Before'] = total_spent_remove.mean()
df_remove.loc['Variance', 'Before'] = total_spent_remove.var()
df_remove.loc['Standard Deviation', 'Before'] = total_spent_remove.std()
df_remove.loc['# Rows', 'Before'] = total_spent_remove.shape[0]

mask = df.isin(['ERROR', 'UNKNOWN'])
df_clean = df[~mask.any(axis=1)].dropna()
total_spent_clean = pd.to_numeric(df_clean['Total Spent'], errors='coerce')

df_remove.loc['Mean', 'After'] = total_spent_clean.mean()
df_remove.loc['Variance', 'After'] = total_spent_clean.var()
df_remove.loc['Standard Deviation', 'After'] = total_spent_clean.std()
df_remove.loc['# Rows', 'After'] = total_spent_clean.shape[0]

print(df_remove, "\n")
df_copy = pd.to_numeric(df['Total Spent'], errors='coerce')

print("Part b")
df_mean = df_remove.__deepcopy__()
total_spent_mean = pd.to_numeric(df['Total Spent'], errors='coerce').fillna(df_mean.at['Mean', 'Before'])

df_mean.loc['Mean', 'After'] = total_spent_mean.mean()
df_mean.loc['Variance', 'After'] = total_spent_mean.var()
df_mean.loc['Standard Deviation', 'After'] = total_spent_mean.std()
df_mean.loc['# Rows', 'After'] = total_spent_mean.shape[0]

print(df_mean, "\n")

print("Part c")
df_median = df_remove.__deepcopy__()
total_spent_median = pd.to_numeric(df['Total Spent'], errors='coerce').fillna(total_spent_remove.median())

df_median.loc['Mean', 'After'] = total_spent_median.mean()
df_median.loc['Variance', 'After'] = total_spent_median.var()
df_median.loc['Standard Deviation', 'After'] = total_spent_median.std()
df_median.loc['# Rows', 'After'] = total_spent_median.shape[0]

print(df_median, "\n")

print("Part d")
df_ffill = df_remove.__deepcopy__()
total_spent_ffill = pd.to_numeric(df['Total Spent'], errors='coerce').ffill()

df_ffill.loc['Mean', 'After'] = total_spent_ffill.mean()
df_ffill.loc['Variance', 'After'] = total_spent_ffill.var()
df_ffill.loc['Standard Deviation', 'After'] = total_spent_ffill.std()
df_ffill.loc['# Rows', 'After'] = total_spent_ffill.shape[0]

print(df_ffill, "\n")

print("Part e")
df_bfill = df_remove.__deepcopy__()
total_spent_bfill = pd.to_numeric(df['Total Spent'], errors='coerce').bfill()

df_bfill.loc['Mean', 'After'] = total_spent_bfill.mean()
df_bfill.loc['Variance', 'After'] = total_spent_bfill.var()
df_bfill.loc['Standard Deviation', 'After'] = total_spent_bfill.std()
df_bfill.loc['# Rows', 'After'] = total_spent_bfill.shape[0]

print(df_bfill, "\n")

print("Part f")
df_dup = df_remove.__deepcopy__()
df_no_dup = df.drop(columns=['Transaction ID']).drop_duplicates(keep='first')
total_spent_dup = pd.to_numeric(df_no_dup['Total Spent'], errors='coerce').dropna()

df_dup.loc['Mean', 'After'] = total_spent_dup.mean()
df_dup.loc['Variance', 'After'] = total_spent_dup.var()
df_dup.loc['Standard Deviation', 'After'] = total_spent_dup.std()
df_dup.loc['# Rows', 'After'] = total_spent_dup.shape[0]

print(df_dup, "\n")