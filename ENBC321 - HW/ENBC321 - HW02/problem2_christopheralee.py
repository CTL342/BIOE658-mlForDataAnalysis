# Name: Christopher A. Lee
# Date: 02/11/2025
# Prof: Dr. Azarnoosh
# Course: Machine Learning for Data Analysis

# Libraries
import numpy as np
import pandas as pd
import sklearn
import statsmodels.api as sm

print("Problem 2")
data = pd.DataFrame({
    "Recovery_Time": [10, 8, 15, 12, 7, 14, 9, 11, 6, 13],
    "Treatment": ["Placebo", "Drug", "Placebo", "Drug", "Drug", "Placebo", "Drug", "Placebo", "Drug", "Placebo"],
    "Gender": ["Male", "Female", "Female", "Male", "Male", "Female", "Female", "Male", "Female", "Male"]
})
print(data)

print("\nPart a")
# Creating a dummy variable for categorical variable Gender
df = pd.get_dummies(data, columns=["Treatment", "Gender"], drop_first=True, dtype=int)
print(df)
multi_lin_reg_model = sklearn.linear_model.LinearRegression().fit(df[["Treatment_Placebo", "Gender_Male"]], df["Recovery_Time"])
print(f"Scikit Learn Multi Linear Regression Results\nEquation: {multi_lin_reg_model.coef_[0]:.4f}x1 + {multi_lin_reg_model.coef_[1]:.4f}x2 + {multi_lin_reg_model.intercept_:.4f}\n")

print("Part b")
print("The coefficients represent the magnitude of the slope and the amount of influence the Treatment or Gender has on the recovery time.")
print("More specifically, the bigger the magnitude, the more influence it has on the recovery time of the patient, in days.\n")

print("Part c")
print(f"Recovery Time (Days) Predicitons\nFemale Patient who Recieved the Drug: {(multi_lin_reg_model.coef_[0] * 0) + (multi_lin_reg_model.coef_[1] * 0) + multi_lin_reg_model.intercept_:.4f}\nMale Patient who Recieved the Drug: {(multi_lin_reg_model.coef_[0] * 0) + (multi_lin_reg_model.coef_[1] * 1) + multi_lin_reg_model.intercept_:.4f}\n")