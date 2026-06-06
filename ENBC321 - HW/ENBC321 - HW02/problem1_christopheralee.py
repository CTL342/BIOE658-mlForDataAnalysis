# Name: Christopher A. Lee
# Date: 02/11/2025
# Prof: Dr. Azarnoosh
# Course: Machine Learning for Data Analysis

# Libraries
import numpy as np
import pandas as pd
import sklearn
import statsmodels.api as sm

print("Problem 1")
num_of_ads = np.array([1, 2, 3, 4, 5, 6])
monthly_sales = np.array([3, 5, 7, 10, 12, 14])
print("Number of Advertisements:", num_of_ads, "\nMonthly Sales:", monthly_sales, "\n")

print("Part a")
# Part i, Custom Linear Regression Function
def custom_LinearRegression(x, y):
    n = np.size(x)

    m_x = np.mean(x)
    m_y = np.mean(y)

    SS_xy = np.sum(y*x) - n * m_y * m_x
    SS_xx = np.sum(x*x) - n * m_x * m_x

    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_1, b_0)
b_1, b_0 = custom_LinearRegression(num_of_ads, monthly_sales)
print(f"Custom Linear Regression Results\nSlope: {b_1:.4f}\nY-Intercept: {b_0:.4f}")
# Part ii, Scikit Learn Linear Regression Function
num_of_ads = num_of_ads.reshape(-1, 1)
lin_reg_model = sklearn.linear_model.LinearRegression().fit(num_of_ads, monthly_sales)
print(f"Scikit Learn Linear Regression Results\nSlope: {lin_reg_model.coef_[0]:.4f}\nY-Intercept: {lin_reg_model.intercept_:.4f}\n")

print("Part b")
print(f"Sales Predicitons\n4 Months: {(b_1 * 4) + b_0:.2f}\n8 Months: {(b_1 * 8) + b_0:.2f}\n")

print("Part c")
predicted_sales = np.array([(b_1 * ad) + b_0 for ad in num_of_ads])
SST = np.sum(np.square(monthly_sales - monthly_sales.mean()))
SSR = np.sum(np.square(predicted_sales - monthly_sales.mean()))
SSE = SST - SSR
print(f"SST: {SST:.4f}\nSSR: {SSR:.4f}\nSSE: {SSE:.4f}")
print(f"R^2: {SSR/SST:.4f}\n")