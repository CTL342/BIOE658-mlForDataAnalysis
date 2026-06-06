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


print("Problem 3")
# Given dataset
np.random.seed(123)
Price = np.round(np.random.normal(loc=300, scale=50, size=50), 2)
Size = np.round(np.random.uniform(low=800, high=4000, size=50), 2)
Bedrooms = np.random.choice(range(2, 7), size=50, replace=True)
Age = np.random.choice(range(1, 51), size=50, replace=True)
Distance_to_City_Center = np.round(np.random.uniform(low=1, high=30, size=50), 2)
# Create DataFrame
data2 = pd.DataFrame({
    "Price": Price,
    "Size": Size,
    "Bedrooms": Bedrooms,
    "Age": Age,
    "Distance_to_City_Center": Distance_to_City_Center
})
print(data2.head())
data2 = sm.add_constant(data2)

print("\nPart a")
house_price_model = sm.OLS(data2["Price"], data2[["Size", "Bedrooms", "Age", "Distance_to_City_Center", "const"]]).fit()
print(house_price_model.summary())

print("\nPart b")
print("Based on the p-values in the summary table, the only variable that significantly impacts the house price is Age. This is because Age has a p-value of 0.038, which is less than the significance level of 0.05. All other variables (Size, Bedrooms, Distance) have p-values greater than 0.05, meaning they are not statistically significant.\n")

print("Part c")
print(f"House Sales Prediction\nParameters: Size; 2500ft, Bedrooms; 4, Age; 20 years, Distance to City Center; 10 miles\nCost: ${(house_price_model.params['Size'] * 2500) + (house_price_model.params['Bedrooms'] * 4) + (house_price_model.params['Age'] * 20) + (house_price_model.params['Distance_to_City_Center'] * 10) + house_price_model.params['const']:.4f}")